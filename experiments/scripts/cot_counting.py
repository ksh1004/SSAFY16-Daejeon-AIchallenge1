# =====================================================================
#  카운팅 문항 Chain-of-Thought — 사전 등록된 단일 구성 실험
#
#  왜: 병목은 카운팅 하나입니다(0.733 vs 0.930). 지금 방식은 마지막 위치의
#      a/b/c/d 로짓 하나만 봅니다 — 모델이 실제로 '세어볼' 기회가 없습니다.
#      추론을 생성하게 한 뒤 정답을 스코어링하면 메커니즘이 달라집니다.
#
#  오늘 배운 것 반영 (HANDOFF §3 채택 규칙):
#    - 구성을 **하나만** 사전 지정한다. 프롬프트 변형을 훑어 최대값을 고르지 않는다.
#    - 파싱·기계적 동작 확인은 dev(버리는 데이터)로 하고,
#      점수는 홀드아웃 카운팅 210건에서 **한 번만** 잰다.
#    - 기준선 카운팅 0.7333, 채택 기준 +1%p (사전 지정 단일 구성).
#
#  사용법
#    .venv/Scripts/python.exe cot_counting.py --smoke      # dev 20건, 파싱 확인만
#    .venv/Scripts/python.exe cot_counting.py --holdout    # 본 평가 (210건, 1회)
#    .venv/Scripts/python.exe cot_counting.py --test       # 채택 시에만
# =====================================================================
import argparse, time, re

# ★ infer_ft 를 transformers 보다 먼저 (HF_HUB_CACHE 설정 시점 문제)
import infer_ft as F
from infer_ft import (ROOT, CHOICES, CH, is_counting, load_model, get_holdout,
                      SYSTEM_INSTRUCT, build_mc_prompt, MODEL_ID)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F_
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoProcessor

IMAGE_PX  = 640          # 어댑터 학습 해상도. 여기서 해상도를 건드리지 않는다.
MAX_NEW   = 128          # 추론 길이 상한
CKPT_EVERY = 60          # 초

# ★ 어시스턴트 응답을 추론으로 강제 시작시킵니다.
#   파인튜닝 어댑터가 '한 글자만 출력'을 학습해서, 그냥 두면 45%가 추론 전에
#   정답 글자부터 뱉습니다(dev 스모크에서 확인). 그러면 뒤따르는 '추론'은
#   이미 정한 답의 사후 합리화라 가설을 검증하지 못합니다.
#   성능 튜닝이 아니라 실험이 의도대로 돌게 하는 구현 수정입니다.
PREFILL = "사진을 보고 세어보겠습니다. "

# ★ 사전 지정한 단 하나의 프롬프트. 변형을 시도하지 않는다.
COT_SUFFIX = "\n먼저 사진에서 해당하는 것을 하나씩 세어 짧게 설명한 뒤, 마지막 줄에 '정답: x' 형식으로 답하세요."
ANSWER_CUE = "\n정답: "


def build_cot_messages(row, img):
    o = [str(row[k]) for k in "abcd"]
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_INSTRUCT}]},
        {"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": build_mc_prompt(str(row["question"]), *o) + COT_SUFFIX},
        ]},
    ]


@torch.no_grad()
def cot_probs(model, proc, df, ckpt, desc="cot", show=0):
    """추론을 생성한 뒤 '정답: ' 다음 토큰의 a/b/c/d 로짓으로 확률을 만든다."""
    n = len(df)
    probs = np.full((n, 4), np.nan)
    texts_out = [""] * n
    done = 0
    if ckpt.exists():
        saved = np.load(ckpt, allow_pickle=True).item()
        if len(saved["p"]) == n:
            probs, texts_out = saved["p"], saved["t"]
            ok = ~np.isnan(probs[:, 0])
            done = n if ok.all() else int(np.argmin(ok))
            print(f"체크포인트 — {done}/{n} 완료, 이어서")
    if done >= n:
        return probs, texts_out

    tok = proc.tokenizer
    cid = torch.tensor([tok.encode(c, add_special_tokens=False)[0]
                        for c in CHOICES]).to(model.device)
    # 생성에는 KV 캐시가 필요합니다 (load_model 이 use_cache=False 로 꺼둠)
    model.config.use_cache = True
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = True

    t0, last = time.time(), time.time()
    for i in tqdm(range(done, n), desc=desc, initial=0, total=n - done):
        row = df.iloc[i]
        img = Image.open(ROOT / row["path"]).convert("RGB")
        msgs = build_cot_messages(row, img)
        text = proc.apply_chat_template(msgs, tokenize=False,
                                        add_generation_prompt=True) + PREFILL
        tok.padding_side = "left"
        enc = proc(text=[text], images=[img], return_tensors="pt")
        enc = {k: v.to(model.device) for k, v in enc.items() if isinstance(v, torch.Tensor)}

        gen = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
        new = gen[0, enc["input_ids"].shape[1]:]
        reasoning = PREFILL + tok.decode(new, skip_special_tokens=True)
        texts_out[i] = reasoning

        # 생성된 추론을 붙이고 "정답: " 다음 한 토큰의 a/b/c/d 로짓을 본다
        # text 에 이미 PREFILL 이 들어 있으므로 reasoning 에서 그만큼 잘라 중복을 막는다
        body = reasoning[len(PREFILL):].split("정답:")[0].rstrip()
        full = text + body + ANSWER_CUE
        enc2 = proc(text=[full], images=[img], return_tensors="pt")
        enc2 = {k: v.to(model.device) for k, v in enc2.items() if isinstance(v, torch.Tensor)}
        lg = model(**enc2).logits[:, -1, :]
        probs[i] = F_.softmax(lg[:, cid].float(), -1).cpu().numpy()[0]

        if show and i - done < show:
            tqdm.write(f"\n--- {row['id']} ---\n{reasoning[:300]}\n"
                       f"-> {dict(zip(CHOICES, np.round(probs[i], 3)))}")
        if time.time() - last > CKPT_EVERY:
            np.save(ckpt, {"p": probs, "t": texts_out}); last = time.time()

    np.save(ckpt, {"p": probs, "t": texts_out})
    el = time.time() - t0
    print(f"\n완료 {n - done}건 / {el:.0f}초 ({el / max(n - done, 1):.1f} s/건)")
    return probs, texts_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="dev 20건 — 파싱 동작 확인용(버리는 데이터)")
    ap.add_argument("--holdout", action="store_true", help="본 평가: 홀드아웃 카운팅 210건, 1회")
    ap.add_argument("--test", action="store_true", help="채택된 경우에만")
    args = ap.parse_args()
    if not (args.smoke or args.holdout or args.test):
        ap.error("--smoke / --holdout / --test 중 하나")

    model, _ = load_model()
    proc = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=IMAGE_PX*IMAGE_PX,
                                         max_pixels=IMAGE_PX*IMAGE_PX,
                                         trust_remote_code=True)

    if args.smoke:
        dev = pd.read_csv(ROOT / "dev.csv")
        sub = dev[is_counting(dev["question"])].head(20).reset_index(drop=True)
        print(f"\n[smoke] dev 카운팅 {len(sub)}건 — 생성/파싱이 되는지만 봅니다\n")
        p, txt = cot_probs(model, proc, sub, ROOT / "_ckpt_cot_smoke.npy",
                           desc="smoke", show=3)
        bad = int(np.isnan(p[:, 0]).sum())
        flat = int((p.max(1) < 0.30).sum())
        print(f"\n확률 생성 실패 {bad}건 / 무의미하게 평평한 분포(<0.30) {flat}건")
        print(f"평균 최대확률 {p.max(1).mean():.3f}  (직접 스코어링은 보통 0.82)")
        print(f"추론 평균 길이 {np.mean([len(t) for t in txt]):.0f}자")
        print("\n생성·파싱이 정상이면 --holdout 으로 본 평가를 1회 하세요.")
        return

    train_df = pd.read_csv(ROOT / "train.csv")
    if args.holdout:
        va = get_holdout(train_df)
        mc = is_counting(va["question"]).values
        sub = va[mc].reset_index(drop=True)
        gold = sub["answer"].astype(str).str.strip().values
        print(f"\n[본 평가] 홀드아웃 카운팅 {len(sub)}건 — 1회만 잽니다\n")
        p, txt = cot_probs(model, proc, sub, ROOT / "_ckpt_cot_holdout.npy", desc="cot holdout")

        base = np.load(ROOT / "vlm7b_probs_holdout609_ft.npy").astype(float)
        zs   = np.load(ROOT / "vlm_probs_holdout_full_7B_4bit_640px_zs_nodev.npy").astype(float)
        txtp = np.load(ROOT / "text_probs_holdout.npy").astype(float)
        def norm(q): return q / q.sum(1, keepdims=True)
        def bl(pv, pt, a=0.75):
            return norm(np.clip(pv,1e-9,1)**a * np.clip(pt,1e-9,1)**(1-a))
        cur = bl(norm(base + zs), txtp)[mc]
        a_cur = (CH[cur.argmax(1)] == gold).mean()
        a_cot = (CH[p.argmax(1)] == gold).mean()
        a_mix = (CH[bl(norm(p + base[mc] + zs[mc]), txtp[mc]).argmax(1)] == gold).mean()

        print("\n" + "="*58)
        print(f"기준선 (현재 파이프라인 카운팅)  {a_cur:.4f}")
        print(f"CoT 단독                        {a_cot:.4f}   ({(a_cot-a_cur)*100:+.2f}%p)")
        print(f"CoT + 기존 앙상블 + 텍스트       {a_mix:.4f}   ({(a_mix-a_cur)*100:+.2f}%p)")
        print(f"{len(sub)}건 표준오차 ±{100*(0.25/len(sub))**0.5:.1f}%p")
        print("채택 기준: 사전 지정 단일 구성이므로 +1%p 이상")
        best = max(a_cot, a_mix)
        print("-> " + ("채택 — test 적용을 검토하세요" if (best-a_cur)*100 >= 1
                       else "기각. 프롬프트를 바꿔가며 재시도하지 마세요(그게 오늘의 실수였습니다)"))
        print("="*58)
        np.save(ROOT / "vlm7b_probs_holdout_cnt_cot.npy", p)
        return

    if args.test:
        te = pd.read_csv(ROOT / "test.csv")
        mc = is_counting(te["question"]).values
        sub = te[mc].reset_index(drop=True)
        print(f"\n[test] 카운팅 {len(sub)}건\n")
        p, _ = cot_probs(model, proc, sub, ROOT / "_ckpt_cot_test.npy", desc="cot test")
        full = np.full((len(te), 4), np.nan); full[mc] = p
        np.save(ROOT / "vlm7b_probs_test_cnt_cot.npy", full)
        print("저장: vlm7b_probs_test_cnt_cot.npy")


if __name__ == "__main__":
    main()
