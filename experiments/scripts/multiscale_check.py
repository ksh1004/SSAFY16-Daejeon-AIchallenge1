# =====================================================================
#  앙상블 축 탐색 — 해상도 다중 스케일 · 좌우반전 TTA (홀드아웃 609건)
#
#  근거: 이 프로젝트에서 실제로 통한 건 '앙상블'이었습니다.
#        같은 모델의 ft/zs 확률을 평균낸 것만으로 +0.66%p (0.8555 -> 0.8621).
#        둘은 상관이 매우 높은데도 그만큼 나왔으므로, 더 안 겹치는 축을 찾습니다.
#
#  기각된 것과의 차이: 선지 순서 TTA는 '선지 구조' 축이라 파인튜닝이 이미 흡수했습니다.
#        해상도와 이미지 반전은 건드리는 축이 다릅니다.
#
#  모델은 한 번만 로드하고 processor만 바꿔가며 잽니다 (ViT가 동적 해상도라 가능).
#
#  사용법
#    .venv/Scripts/python.exe multiscale_check.py --batch 2
#    .venv/Scripts/python.exe multiscale_check.py --batch 2 --only 512   # 하나만
# =====================================================================
import argparse

# ★ infer_ft 를 반드시 transformers 보다 먼저 import 해야 합니다.
#   infer_ft 가 HF_HUB_CACHE(모델 16GB가 프로젝트 폴더에 있음)와 HF_HUB_OFFLINE 을
#   설정하는데, huggingface_hub 는 import 시점에 이 값들을 읽어 고정합니다.
#   순서가 뒤바뀌면 기본 캐시(비어 있음)를 보고 16GB를 새로 받으려 합니다.
import infer_ft as F
from infer_ft import (ROOT, CH, is_counting, load_model, predict_probs, get_holdout,
                      VQADataset, build_messages, MODEL_ID)

import numpy as np
import pandas as pd
from PIL import Image
from transformers import AutoProcessor

ALPHA = 0.75

def norm(p): return p / p.sum(1, keepdims=True)
def bl(pv, pt, a=ALPHA):
    return norm(np.clip(pv, 1e-9, 1) ** a * np.clip(pt, 1e-9, 1) ** (1 - a))


class FlipDataset(VQADataset):
    """이미지를 좌우반전해서 넣는다. 개수·재질은 반전에 불변이므로 정답이 바뀌지 않는다."""
    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(ROOT / row["path"]).convert("RGB").transpose(Image.FLIP_LEFT_RIGHT)
        return {"messages": build_messages(row, img), "image": img}


def make_processor(px):
    return AutoProcessor.from_pretrained(MODEL_ID, min_pixels=px*px, max_pixels=px*px,
                                         trust_remote_code=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--only", type=str, default=None, help="512 / 768 / flip 중 하나만")
    args = ap.parse_args()

    train_df = pd.read_csv(ROOT / "train.csv")
    va = get_holdout(train_df)
    gold = va["answer"].astype(str).str.strip().values
    mcnt = is_counting(va["question"]).values
    model, _ = load_model()

    jobs = [("512",  dict(px=512, flip=False)),
            ("768",  dict(px=768, flip=False)),
            ("flip", dict(px=640, flip=True))]
    if args.only:
        jobs = [j for j in jobs if j[0] == args.only]

    for tag, cfg in jobs:
        out = ROOT / f"vlm7b_probs_holdout609_ft_{tag}.npy"
        if out.exists():
            print(f"\n[{tag}] 기존 결과 사용: {out.name}")
            continue
        print(f"\n{'='*62}\n[{tag}] px={cfg['px']} flip={cfg['flip']}\n{'='*62}")
        proc = make_processor(cfg["px"])
        if cfg["flip"]:
            _orig = F.VQADataset
            F.VQADataset = FlipDataset          # predict_probs 내부가 쓰는 것을 갈아끼움
        try:
            p = predict_probs(model, proc, va, args.batch,
                              ROOT / f"_ckpt_h609_{tag}.npy", desc=tag)
        finally:
            if cfg["flip"]:
                F.VQADataset = _orig
        np.save(out, p)

    # ---------------- CPU 분석 ----------------
    p_ft  = np.load(ROOT / "vlm7b_probs_holdout609_ft.npy").astype(np.float64)
    p_zs  = np.load(ROOT / "vlm_probs_holdout_full_7B_4bit_640px_zs_nodev.npy").astype(np.float64)
    p_txt = np.load(ROOT / "text_probs_holdout.npy").astype(np.float64)
    extra = {}
    for tag in ["512", "768", "flip"]:
        f = ROOT / f"vlm7b_probs_holdout609_ft_{tag}.npy"
        if f.exists(): extra[tag] = np.load(f).astype(np.float64)

    def acc(p, m=None):
        pr = CH[p.argmax(1)]
        return (pr == gold).mean() if m is None else (pr[m] == gold[m]).mean()

    base = bl(norm(p_ft + p_zs), p_txt)
    a0 = acc(base)
    print("\n" + "=" * 62)
    print("홀드아웃 609건 (LB 오프셋 ~0 — 이 숫자를 LB로 읽으세요)")
    print("=" * 62)
    print(f"  기준선 ft+zs+텍스트        {a0:.4f}   (LB 0.86322)")
    print(f"  파인튜닝 640px 단독        {acc(p_ft):.4f}")
    for tag, p in extra.items():
        print(f"  파인튜닝 {tag:4s} 단독        {acc(p):.4f}"
              f"   (640px와 예측 불일치 {100*(p.argmax(1)!=p_ft.argmax(1)).mean():.1f}%)")

    print("\n앙상블 조합 (전부 텍스트 α=0.75 적용 후)")
    combos = [("기준선 ft+zs", norm(p_ft + p_zs))]
    for tag, p in extra.items():
        combos.append((f"ft+zs+{tag}", norm(p_ft + p_zs + p)))
        combos.append((f"ft+{tag}", norm(p_ft + p)))
    if len(extra) >= 2:
        combos.append(("ft+zs+전부", norm(p_ft + p_zs + sum(extra.values()))))
    best = ("기준선", a0)
    for name, p in combos:
        P = bl(p, p_txt); a = acc(P)
        d = (a - a0) * 100
        print(f"  {name:16s} {a:.4f}  ({d:+.2f}%p)"
              f"  카운팅 {acc(P,mcnt):.4f} / 기타 {acc(P,~mcnt):.4f}")
        if a > best[1]: best = (name, a)

    print("\n" + "=" * 62)
    print(f"최선: {best[0]}  {best[1]:.4f}  ({(best[1]-a0)*100:+.2f}%p)")
    print(f"609건 표준오차 ±{100*(0.25/len(va))**0.5:.1f}%p")
    print("판단 기준 (HANDOFF §3): 탐색으로 고른 값이므로 +1.5%p 이상이어야 채택.")
    if (best[1] - a0) * 100 >= 1.5:
        print(f"  -> 채택. test에 적용하면 약 60분/스케일. 예상 LB {0.86322+(best[1]-a0):.4f}")
    else:
        print("  -> 기각. 이득이 노이즈 수준입니다.")
    print("=" * 62)


if __name__ == "__main__":
    main()
