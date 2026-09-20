"""Qwen2.5-VL-7B (4bit) 로 a/b/c/d 확률을 계산한다 — zero-shot 또는 LoRA 어댑터.

생성(generate) 대신 **assistant 헤더 직후 위치의 로짓에서 a/b/c/d 네 토큰만 비교**합니다.
파싱 실패가 원천적으로 없고, 학습(정답 글자 1토큰에만 loss)과 추론이 정확히 일치합니다.

    # zero-shot
    python infer.py --split holdout --zeroshot
    python infer.py --split test    --zeroshot

    # 파인튜닝 어댑터
    python infer.py --split holdout --adapter checkpoints/lora_7b_ft
    python infer.py --split test    --adapter checkpoints/lora_7b_ft

결과는 outputs/probs_{name}_{split}.npy  (N×4, 행 순서 = CSV 순서).
2분마다 체크포인트를 저장하므로 중단돼도 같은 명령으로 이어서 실행됩니다.

실측 (RTX 5060 Laptop 8GB, batch 2): 0.70 s/샘플 -> test 5,074건 약 60분.
"""
import argparse
import gc
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common import ROOT

# 로컬에 모델 캐시가 있으면 그걸 쓰고 네트워크로 나가지 않는다.
# ★ huggingface_hub 는 import 시점에 이 값을 읽으므로 transformers 보다 먼저 설정해야 한다.
if (ROOT / "models--Qwen--Qwen2.5-VL-7B-Instruct").exists():
    os.environ.setdefault("HF_HUB_CACHE", str(ROOT))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from common import (CH, CHOICES, DATA_DIR, IMAGE_PX, LORA_ALPHA, LORA_DROPOUT, LORA_R,
                    LORA_TARGETS, MODEL_ID, MODEL_REVISION, OUTPUT_DIR, build_messages,
                    get_holdout, is_counting, load_csv)

Image.MAX_IMAGE_PIXELS = None


class VQADataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(DATA_DIR / row["path"]).convert("RGB")
        return {"messages": build_messages(row, img), "image": img}


@dataclass
class Collator:
    processor: Any

    def __call__(self, batch):
        texts = [self.processor.apply_chat_template(s["messages"], tokenize=False,
                                                    add_generation_prompt=True) for s in batch]
        images = [s["image"] for s in batch]
        # left padding -> index -1 이 항상 마지막 실제 토큰
        self.processor.tokenizer.padding_side = "left"
        return self.processor(text=texts, images=images, padding=True, return_tensors="pt")


def load_model(adapter: Path | None, image_px: int = IMAGE_PX):
    import transformers
    from transformers import AutoProcessor, BitsAndBytesConfig

    # transformers 5.x 에서 AutoModelForVision2Seq 가 제거됨 (실험은 4.57 에서 수행)
    AutoVLM = getattr(transformers, "AutoModelForVision2Seq", None) \
        or transformers.AutoModelForImageTextToText

    free, total = torch.cuda.mem_get_info()
    print(f"GPU {torch.cuda.get_device_name()} / 여유 {free / 1e9:.1f} / {total / 1e9:.1f} GB")

    processor = AutoProcessor.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True,
        min_pixels=image_px * image_px, max_pixels=image_px * image_px)

    # T4 등 compute capability 8.0 미만은 bf16 텐서코어가 없어 fp16 이 훨씬 빠르다.
    # (torch.cuda.is_bf16_supported() 는 에뮬레이션도 True 라 판정에 쓰면 안 됨)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    base = AutoVLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True, torch_dtype=dtype,
        device_map={"": 0},          # "auto" 는 VRAM 부족 시 조용히 오프로드한다
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype))

    bad = [n for n, p in base.named_parameters() if p.device.type in ("meta", "cpu")]
    assert not bad, f"GPU 밖 파라미터 {len(bad)}개 — VRAM 부족으로 오프로드됨"
    base.config.use_cache = False

    if adapter is None:
        model = base
        print("어댑터 없음 — zero-shot")
    else:
        from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
        from safetensors.torch import load_file

        model = get_peft_model(base, LoraConfig(
            r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none",
            target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
        # PeftModel.from_pretrained 로 다시 감싸면 LoRA 가 이중 주입된다 -> state_dict 만 로드
        sd = load_file(str(adapter / "adapter_model.safetensors"))
        res = set_peft_model_state_dict(model, sd)
        unexpected = getattr(res, "unexpected_keys", None)
        assert not unexpected, f"어댑터 키 불일치: {list(unexpected)[:3]}"
        # lora_B 는 0 으로 초기화되므로, 0 이 아니면 실제로 로드된 것
        nz = sum(float(p.detach().abs().sum()) for n, p in model.named_parameters() if "lora_B" in n)
        assert nz > 0, "lora_B 가 전부 0 — 어댑터가 반영되지 않았습니다"
        print(f"어댑터 {adapter} ({len(sd)}개 텐서, lora_B 절대합 {nz:.1f})")

    model.eval()
    return model, processor


def last_token_kwarg(model):
    """마지막 위치 logits 만 계산 -> (B, L, V) 대신 (B, 1, V). left padding 이라 결과 동일."""
    import inspect
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    params = inspect.signature(base.forward).parameters
    for name in ("logits_to_keep", "num_logits_to_keep"):
        if name in params:
            return {name: 1}
    return {}


@torch.no_grad()
def predict_probs(model, processor, df, batch_size, ckpt: Path, desc="predict"):
    """(N, 4) 확률. ckpt 에 부분 결과를 저장해 중단 시 이어서 실행한다."""
    n = len(df)
    probs = np.full((n, 4), np.nan)
    done = 0
    if ckpt.exists():
        saved = np.load(ckpt)
        if len(saved) == n:
            probs = saved
            ok = ~np.isnan(probs[:, 0])
            done = n if ok.all() else int(np.argmin(ok))
            print(f"체크포인트에서 재개: {done}/{n}")
    if done >= n:
        return probs

    tok = processor.tokenizer
    ids = [tok.encode(c, add_special_tokens=False) for c in CHOICES]
    assert all(len(x) == 1 for x in ids), "a/b/c/d 가 단일 토큰이 아닙니다"
    cid = torch.tensor([x[0] for x in ids], device=model.device)

    dl = DataLoader(VQADataset(df.iloc[done:]), batch_size=batch_size, shuffle=False,
                    collate_fn=Collator(processor), num_workers=0)
    keep1 = last_token_kwarg(model)

    i, t0, last_save = done, time.time(), time.time()
    for batch in tqdm(dl, desc=desc):
        batch = {k: v.to(model.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        logits = model(**batch, **keep1).logits[:, -1, :]
        p = F.softmax(logits[:, cid].float(), dim=-1).cpu().numpy()
        probs[i:i + len(p)] = p
        i += len(p)
        if time.time() - last_save > 120:
            np.save(ckpt, probs)
            last_save = time.time()

    np.save(ckpt, probs)
    assert not np.isnan(probs).any()
    el = time.time() - t0
    print(f"완료 {n - done}건 / {el:.0f}초 ({el / max(n - done, 1):.2f} s/샘플)")
    return probs


def report(p, df, label):
    gold = df["answer"].astype(str).str.strip().values
    pred = CH[p.argmax(1)]
    m = is_counting(df["question"]).values
    print(f"{label}  전체 {(pred == gold).mean():.4f}"
          f"  (카운팅 {(pred[m] == gold[m]).mean():.4f} / 기타 {(pred[~m] == gold[~m]).mean():.4f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["holdout", "test"], required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--zeroshot", action="store_true")
    g.add_argument("--adapter", type=Path, help="adapter_model.safetensors 가 들어 있는 폴더")
    ap.add_argument("--name", default=None, help="출력 파일 이름 (기본: zs 또는 어댑터 폴더명)")
    ap.add_argument("--batch", type=int, default=2, help="8GB 는 2, 16GB 는 4")
    ap.add_argument("--limit", type=int, default=None, help="앞 N건만 (동작 확인용)")
    args = ap.parse_args()

    name = args.name or ("zs" if args.zeroshot else args.adapter.name)
    df = get_holdout(load_csv("train.csv")) if args.split == "holdout" else load_csv("test.csv")
    if args.limit:
        df = df.head(args.limit).reset_index(drop=True)
        name += f"_limit{args.limit}"

    model, processor = load_model(None if args.zeroshot else args.adapter)
    out = OUTPUT_DIR / f"probs_{name}_{args.split}.npy"
    p = predict_probs(model, processor, df, args.batch,
                      OUTPUT_DIR / f"_ckpt_{name}_{args.split}.npy", desc=f"{name}/{args.split}")
    np.save(out, p)
    print(f"저장: {out}")
    if args.split == "holdout":
        report(p, df, name)
    del model
    gc.collect()


if __name__ == "__main__":
    main()
