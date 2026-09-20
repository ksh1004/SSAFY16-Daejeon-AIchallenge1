# =====================================================================
#  7B QLoRA 파인튜닝 어댑터 -> test 추론  (노트북 RTX 5060 8GB 전용)
#
#  HANDOFF.md ② 단계. 노트북(.ipynb) 셀을 자체 완결형 스크립트로 옮긴 것.
#  8GB 카드에서 몇 시간 걸리는 실행이므로 '재개 가능'이 핵심 설계입니다.
#
#  사용법
#    python infer_ft.py --verify            # 홀드아웃 50건으로 파이프라인 검증 (~3분)
#    python infer_ft.py --holdout           # 홀드아웃 400건 재현 (0.8575 나와야 정상)
#    python infer_ft.py --test              # test 5074건 본 추론 (재개 가능)
#    python infer_ft.py --test --batch 1    # OOM이면 배치를 줄여서 이어서 실행
#
#  중단되면 같은 명령을 다시 실행하세요. 이미 끝난 부분은 건너뜁니다.
# =====================================================================
import os, sys, gc, json, time, argparse
from pathlib import Path

# ★ 모델 캐시(16GB)가 기본 위치가 아니라 프로젝트 폴더에 있습니다.
#   transformers 를 import 하기 전에 지정해야 재다운로드를 피합니다.
_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HUB_CACHE", str(_ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")     # 있는 것만 쓰고 네트워크로 나가지 않음

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Any
from PIL import Image
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent
Image.MAX_IMAGE_PIXELS = None

# ---------- 설정 (노트북 셀 4와 동일하게 유지) ----------
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
USE_4BIT = True
IMAGE_PX = 640          # ★ 어댑터가 640px로 학습됨. 바꾸면 성능 저하
ADAPTER  = ROOT / "lora_7b_ft_check"
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]
SEED = 42
VAL_RATIO = 0.12
CHOICES = list("abcd")
CH = np.array(CHOICES)

CNT_PAT = r"몇 개|몇개|개입니까|개인가요|몇 가지"
def is_counting(s):
    return s.str.contains(CNT_PAT, regex=True)


# ---------- 프롬프트 (노트북 셀 10과 완전히 동일해야 함) ----------
SYSTEM_INSTRUCT = (
    "You are a helpful visual question answering assistant. "
    "Answer using exactly one letter among a, b, c, or d. No explanation."
)

def build_mc_prompt(question, a, b, c, d):
    return (
        f"{question}\n"
        f"(a) {a}\n(b) {b}\n(c) {c}\n(d) {d}\n\n"
        "정답을 반드시 a, b, c, d 중 하나의 소문자 한 글자로만 출력하세요."
    )

def build_messages(row, img):
    o = [str(row[k]) for k in "abcd"]
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_INSTRUCT}]},
        {"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": build_mc_prompt(str(row["question"]), *o)},
        ]},
    ]


# ---------- Dataset / Collator (노트북 셀 16, 추론 경로만) ----------
class VQADataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(ROOT / row["path"]).convert("RGB")
        return {"messages": build_messages(row, img), "image": img}


@dataclass
class Collator:
    processor: Any

    def __call__(self, batch):
        texts, images = [], []
        for s in batch:
            texts.append(self.processor.apply_chat_template(
                s["messages"], tokenize=False, add_generation_prompt=True))
            images.append(s["image"])
        # 추론은 left padding -> index -1 이 항상 마지막 실제 토큰
        self.processor.tokenizer.padding_side = "left"
        return self.processor(text=texts, images=images,
                              padding=True, return_tensors="pt")


# ---------- 모델 로드 ----------
def load_model():
    import transformers
    from transformers import AutoProcessor, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from safetensors.torch import load_file

    # transformers 5.x 에서 AutoModelForVision2Seq 가 제거됨.
    # 데스크탑은 4.x 였으므로 4.x 를 권장하지만, 5.x 에서도 돌아가게 해둔다.
    AutoVLM = getattr(transformers, "AutoModelForVision2Seq", None) \
              or transformers.AutoModelForImageTextToText
    print(f"transformers {transformers.__version__} / {AutoVLM.__name__}")

    free, total = torch.cuda.mem_get_info()
    print(f"GPU     : {torch.cuda.get_device_name()}  cc={torch.cuda.get_device_capability()}")
    print(f"로드 전 : {free/1e9:.1f} / {total/1e9:.1f} GB 여유")
    if free / 1e9 < 6.5:
        print("⚠️ 여유가 6.5GB 미만입니다. 다른 GPU 프로그램을 끄세요 (모델만 5.4GB 필요).")

    processor = AutoProcessor.from_pretrained(
        MODEL_ID, min_pixels=IMAGE_PX * IMAGE_PX, max_pixels=IMAGE_PX * IMAGE_PX,
        trust_remote_code=True)

    load_kw = dict(device_map={"": 0}, trust_remote_code=True,
                   torch_dtype=torch.bfloat16)
    if USE_4BIT:
        load_kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)

    base = AutoVLM.from_pretrained(MODEL_ID, **load_kw)

    # device_map={"":0} 이어도 확인은 해둔다 (오프로드되면 결과가 조용히 틀어짐)
    bad = [(n, str(p.device)) for n, p in base.named_parameters()
           if p.device.type in ("meta", "cpu")]
    assert not bad, f"GPU 밖 파라미터 {len(bad)}개 (예: {bad[:2]}) — VRAM 부족"

    base.config.use_cache = False
    model = get_peft_model(base, LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none",
        target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))

    # ★ PeftModel.from_pretrained 로 다시 감싸면 LoRA 이중 주입 -> state_dict만 로드
    sd = load_file(str(ADAPTER / "adapter_model.safetensors"))
    res = set_peft_model_state_dict(model, sd)
    missing = getattr(res, "unexpected_keys", None)
    print(f"어댑터  : {len(sd)}개 텐서 로드"
          + (f"  (unexpected {len(missing)})" if missing else ""))
    assert not missing, f"어댑터 키 불일치: {list(missing)[:3]}"

    # 실제로 값이 들어갔는지 확인 — lora_B는 0으로 초기화되므로 비0이면 로드 성공
    nz = sum(float(p.abs().sum()) for n, p in model.named_parameters()
             if "lora_B" in n)
    assert nz > 0, "lora_B가 전부 0입니다 — 어댑터가 실제로 반영되지 않았습니다"
    print(f"        lora_B 절대합 {nz:.1f} (>0 이면 반영됨) ✅")

    model.eval()
    free, total = torch.cuda.mem_get_info()
    print(f"로드 후 : {(total-free)/1e9:.1f} GB 사용 / 여유 {free/1e9:.1f} GB")
    return model, processor


# ---------- 마지막 토큰만 계산하는 최적화 ----------
def last_token_kwarg(model):
    """(B, 650, 152064) 대신 (B, 1, 152064) 만 계산해 배치 4 기준 약 800MB를 아낀다.

    left padding이라 index -1 이 마지막 실제 토큰이므로 수학적으로 동일하다.
    PeftModel.forward 는 **kwargs 로 넘기기만 하므로 베이스 모델에서 확인해야 한다.
    """
    import inspect
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    fwd = inspect.signature(base.forward).parameters
    for name in ("logits_to_keep", "num_logits_to_keep"):
        if name in fwd:
            return {name: 1}
    return {}


@torch.no_grad()
def prove_keep1_equivalent(model, processor, df, batch_size):
    """최적화를 켠 것과 끈 것이 같은 확률을 주는지 한 배치로 직접 확인."""
    keep1 = last_token_kwarg(model)
    if not keep1:
        print("최적화 미지원 — 전체 logits 경로만 사용합니다.")
        return
    tok = processor.tokenizer
    cid = torch.tensor([tok.encode(c, add_special_tokens=False)[0]
                        for c in CHOICES]).to(model.device)
    dl = DataLoader(VQADataset(df.head(batch_size)), batch_size=batch_size,
                    shuffle=False, collate_fn=Collator(processor), num_workers=0)
    batch = next(iter(dl))
    batch = {k: v.to(model.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
    out = []
    for kw in (keep1, {}):
        lg = model(**batch, **kw).logits[:, -1, :]
        out.append(F.softmax(lg[:, cid].float(), -1).cpu().numpy())
    d = np.abs(out[0] - out[1]).max()
    print(f"최적화 동등성 확인: 최대 확률차 {d:.2e} "
          + ("✅" if d < 1e-5 else "❌ — 최적화를 끄고 다시 실행하세요"))
    assert d < 1e-5, "logits_to_keep 경로가 결과를 바꿉니다"
    del out; gc.collect(); torch.cuda.empty_cache()


# ---------- 추론 (재개 가능) ----------
@torch.no_grad()
def predict_probs(model, processor, df, batch_size, ckpt: Path, desc="predict"):
    """(N,4) 확률. ckpt에 부분 결과를 저장해 중단 시 이어서 실행."""
    n = len(df)
    probs = np.full((n, 4), np.nan, dtype=np.float64)
    done = 0
    if ckpt.exists():
        saved = np.load(ckpt)
        if len(saved) == n:
            probs = saved
            ok = ~np.isnan(probs[:, 0])
            done = n if ok.all() else int(np.argmin(ok))   # 첫 NaN 위치부터 재개
            print(f"체크포인트 발견 — {done}/{n} 건 완료, 이어서 진행")
        else:
            print(f"체크포인트 크기 불일치 ({len(saved)} != {n}) — 처음부터")

    if done >= n:
        print("이미 완료된 결과입니다.")
        return probs

    tok = processor.tokenizer
    cid = torch.tensor([tok.encode(c, add_special_tokens=False)[0] for c in CHOICES])
    assert len(set(cid.tolist())) == 4
    cid = cid.to(model.device)

    rest = df.iloc[done:].reset_index(drop=True)
    dl = DataLoader(VQADataset(rest), batch_size=batch_size, shuffle=False,
                    collate_fn=Collator(processor), num_workers=0)

    keep1 = last_token_kwarg(model)
    print(f"마지막 토큰만 계산: {'예 (' + list(keep1)[0] + ')' if keep1 else '아니오 (전체 logits)'}")

    i, t0, last_save = done, time.time(), time.time()
    pbar = tqdm(dl, desc=desc, initial=0, total=len(dl))
    for batch in pbar:
        batch = {k: v.to(model.device) for k, v in batch.items()
                 if isinstance(v, torch.Tensor)}
        logits = model(**batch, **keep1).logits[:, -1, :]  # left padding -> -1이 마지막 실제 토큰
        p = F.softmax(logits[:, cid].float(), dim=-1).cpu().numpy()
        probs[i:i + len(p)] = p
        i += len(p)
        el = time.time() - t0
        pbar.set_postfix({"s/샘플": f"{el/max(i-done,1):.2f}",
                          "남은분": f"{(n-i)*el/max(i-done,1)/60:.0f}"})
        if time.time() - last_save > 120:            # 2분마다 저장
            np.save(ckpt, probs); last_save = time.time()

    np.save(ckpt, probs)
    assert not np.isnan(probs).any(), "결측 확률이 남아 있습니다"
    print(f"\n완료 {n}건 / {time.time()-t0:.0f}초 ({(time.time()-t0)/max(n-done,1):.2f} s/샘플)")
    return probs


# ---------- 홀드아웃 재구성 ----------
def get_holdout(train_df):
    """저장된 holdout_ids.csv 를 우선 쓰고, sklearn 재현과 교차 검증한다."""
    ids = pd.read_csv(ROOT / "holdout_ids.csv")["id"].tolist()
    va = train_df.set_index("id").loc[ids].reset_index()

    try:
        from sklearn.model_selection import train_test_split
        strat = (train_df["answer"].astype(str) + "_"
                 + is_counting(train_df["question"]).astype(str))
        _, va_idx = train_test_split(np.arange(len(train_df)), test_size=VAL_RATIO,
                                     random_state=SEED, stratify=strat)
        rep = train_df.iloc[va_idx]["id"].tolist()
        if rep == ids:
            print("홀드아웃: holdout_ids.csv == sklearn 재현 (순서까지 일치) ✅")
        else:
            same = set(rep) == set(ids)
            print(f"⚠️ sklearn 재현과 다릅니다 (집합 일치={same}). "
                  "holdout_ids.csv 를 신뢰합니다 — 저장된 .npy 와 같은 순서여야 하므로.")
    except ImportError:
        print("홀드아웃: holdout_ids.csv 사용 (sklearn 없어 교차 검증 생략)")
    return va


def report(p, gold, questions, label):
    pred = CH[p.argmax(1)]
    m = is_counting(questions).values
    acc = (pred == gold).mean()
    print(f"{label:16s} 전체 {acc:.4f}"
          + (f" / 카운팅 {(pred[m]==gold[m]).mean():.4f}" if m.sum() else "")
          + (f" / 기타 {(pred[~m]==gold[~m]).mean():.4f}" if (~m).sum() else ""))
    return acc


# ---------- main ----------
def main():
    global IMAGE_PX
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify",  action="store_true", help="홀드아웃 앞 N건으로 파이프라인 검증")
    ap.add_argument("--holdout", action="store_true", help="홀드아웃 400건 재현")
    ap.add_argument("--test",    action="store_true", help="test 5074건 본 추론")
    ap.add_argument("--batch",   type=int, default=2)
    ap.add_argument("--n",       type=int, default=50, help="--verify 건수")
    ap.add_argument("--px",      type=int, default=IMAGE_PX,
                    help="입력 해상도. 어댑터는 640px로 학습됐으므로 바꾸면 단독 성능은 "
                         "떨어질 수 있지만, 앙상블 멤버로는 쓸모가 있을 수 있습니다.")
    ap.add_argument("--tag",     type=str, default="",
                    help="출력 파일명 접미사 (예: --px 512 --tag _512)")
    args = ap.parse_args()

    if args.px != IMAGE_PX:
        print(f"⚠️ 해상도 {IMAGE_PX} -> {args.px} (어댑터 학습 해상도와 다릅니다)")
        IMAGE_PX = args.px
    if not (args.verify or args.holdout or args.test):
        ap.error("--verify / --holdout / --test 중 하나를 지정하세요")

    torch.manual_seed(SEED); np.random.seed(SEED)
    train_df = pd.read_csv(ROOT / "train.csv")
    model, processor = load_model()

    if args.verify or args.holdout:
        va = get_holdout(train_df)
        n = args.n if args.verify else 400
        sub = va.head(n).reset_index(drop=True)
        gold = sub["answer"].astype(str).str.strip().values

        tag = f"verify{n}" if args.verify else "holdout400"
        if args.verify:
            prove_keep1_equivalent(model, processor, sub, args.batch)
        p = predict_probs(model, processor, sub, args.batch,
                          ROOT / f"_ckpt_{tag}.npy", desc=tag)

        print()
        ref = np.load(ROOT / "vlm7b_probs_holdout_ft.npy")[:n]
        report(p,   gold, sub["question"], "이번 실행")
        report(ref, gold, sub["question"], "데스크탑 저장본")
        agree = (p.argmax(1) == ref.argmax(1)).mean()
        mad = np.abs(p - ref).mean()
        print(f"\n예측 일치율 {agree:.3f} / 확률 평균절대차 {mad:.4f}")
        if agree > 0.95 and mad < 0.05:
            print("✅ 데스크탑 파이프라인을 정확히 재현했습니다. --test 로 진행하세요.")
        else:
            print("❌ 재현 실패 — 프롬프트/해상도/어댑터 로드를 점검하세요. "
                  "이 상태로 test를 돌리면 안 됩니다.")
            sys.exit(1)

    if args.test:
        test_df = pd.read_csv(ROOT / "test.csv")
        out = ROOT / f"vlm7b_probs_test_ft{args.tag}.npy"
        p = predict_probs(model, processor, test_df, args.batch,
                          ROOT / f"_ckpt_test_ft{args.tag}.npy", desc=f"test(FT{args.tag})")
        np.save(out, p)
        pred = CH[p.argmax(1)]
        print(f"\n저장: {out.name}")
        print("예측 분포:", dict(pd.Series(pred).value_counts().sort_index()))
        print("평균 최대확률 %.3f" % p.max(1).mean())
        print("\n다음: python make_submission.py")


if __name__ == "__main__":
    main()
