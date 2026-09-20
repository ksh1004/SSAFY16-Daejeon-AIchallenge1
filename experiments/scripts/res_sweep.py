# =====================================================================
#  해상도 스윕 — 카운팅 문항 전용 (홀드아웃) / test 적용
#
#  발견: 파인튜닝 어댑터를 768px로 돌리면 카운팅이 0.7190 -> 0.7810 (+6.2%p),
#        짝지은 검정 p=0.0146 로 유의합니다. 기타 문항은 오히려 약간 손해라
#        **카운팅 문항에만** 고해상도를 쓰는 것이 최선이었습니다 (0.8621 -> 0.8801).
#
#  이 스크립트는 카운팅 문항만 골라 원하는 해상도로 추론합니다.
#  전체를 돌릴 필요가 없어 test도 29분이면 끝납니다.
#
#  사용법
#    .venv/Scripts/python.exe res_sweep.py --px 896 --holdout --batch 1
#    .venv/Scripts/python.exe res_sweep.py --px 768 --test --batch 2
# =====================================================================
import argparse

# ★ infer_ft 를 transformers 보다 먼저 import (HF_HUB_CACHE 를 먼저 설정해야 함)
import infer_ft as F
from infer_ft import (ROOT, CH, is_counting, load_model, predict_probs, get_holdout,
                      MODEL_ID)

import numpy as np
import pandas as pd
from transformers import AutoProcessor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--px", type=int, required=True)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    if not (args.holdout or args.test):
        ap.error("--holdout 또는 --test 를 지정하세요")

    train_df = pd.read_csv(ROOT / "train.csv")
    model, _ = load_model()
    proc = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=args.px * args.px,
                                         max_pixels=args.px * args.px,
                                         trust_remote_code=True)

    for which in (["holdout"] if args.holdout else []) + (["test"] if args.test else []):
        df = get_holdout(train_df) if which == "holdout" else pd.read_csv(ROOT / "test.csv")
        mc = is_counting(df["question"]).values
        sub = df[mc].reset_index(drop=True)
        print(f"\n{'='*62}\n[{which}] {args.px}px — 카운팅 {len(sub)}/{len(df)}건 "
              f"({100*mc.mean():.1f}%)\n{'='*62}")

        p = predict_probs(model, proc, sub, args.batch,
                          ROOT / f"_ckpt_{which}_cnt_{args.px}.npy",
                          desc=f"{which} cnt {args.px}px")

        # 전체 길이 배열에 카운팅 위치만 채우고 나머지는 NaN (쓰는 쪽에서 구분)
        full = np.full((len(df), 4), np.nan)
        full[mc] = p
        out = ROOT / f"vlm7b_probs_{which}_cnt{args.px}.npy"
        np.save(out, full)
        print(f"저장: {out.name}  (카운팅 위치만 채움, 나머지는 NaN)")

        if which == "holdout":
            gold = df["answer"].astype(str).str.strip().values
            print(f"카운팅 {len(sub)}건 정확도 {(CH[p.argmax(1)] == gold[mc]).mean():.4f}")
            for tag, f in [("640(어댑터 학습)", "vlm7b_probs_holdout609_ft.npy"),
                           ("768", "vlm7b_probs_holdout609_ft_768.npy")]:
                q = ROOT / f
                if q.exists():
                    r = np.load(q).astype(float)
                    print(f"  참고 {tag:16s} {(CH[r[mc].argmax(1)] == gold[mc]).mean():.4f}")


if __name__ == "__main__":
    main()
