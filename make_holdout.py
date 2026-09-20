"""train을 (정답 위치 × 문항 유형)으로 층화 분할해 홀드아웃 609건의 id를 저장한다.

    python make_holdout.py        ->  data/holdout_ids.csv

train 5,073건 -> 학습 4,464 / 홀드아웃 609. 모든 실험 판단은 이 609건으로 했습니다.
홀드아웃 확률 파일의 행 순서가 이 파일의 순서와 같아야 하므로, 한 번 만든 뒤에는
다시 만들지 마세요 (같은 seed면 순서까지 동일하게 재현되긴 합니다).
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import DATA_DIR, SEED, VAL_RATIO, is_counting, load_csv


def main():
    train_df = load_csv("train.csv")
    strat = train_df["answer"].astype(str) + "_" + is_counting(train_df["question"]).astype(str)
    _, va_idx = train_test_split(np.arange(len(train_df)), test_size=VAL_RATIO,
                                 random_state=SEED, stratify=strat)
    va = train_df.iloc[va_idx]
    out = DATA_DIR / "holdout_ids.csv"
    va[["id"]].to_csv(out, index=False)
    print(f"학습 {len(train_df) - len(va)} / 홀드아웃 {len(va)}")
    print(f"홀드아웃 카운팅 비율 {100 * is_counting(va['question']).mean():.1f}%")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
