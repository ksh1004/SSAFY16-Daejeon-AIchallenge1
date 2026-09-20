"""확률 파일들을 합쳐 최종 제출 파일을 만든다 (CPU, 수 초).

최종 제출(Public 0.86322 / Private 0.85494) 구성:

    P_vlm   = (P_finetuned + P_zeroshot) / 2          # 같은 모델의 두 상태 평균
    P_final = P_vlm^0.75 · P_text^0.25                # 텍스트 전용 모델과 로그공간 블렌딩
    + train/test 중복 이미지 24건은 train 정답으로 덮어쓰기

    python make_submission.py \\
        --ft   outputs/probs_lora_7b_ft_test.npy \\
        --zs   outputs/probs_zs_test.npy \\
        --text outputs/text_probs_test.npy

같은 이름의 holdout 확률 파일이 있으면 홀드아웃 609건 정확도도 함께 출력합니다.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import ALPHA, CH, DATA_DIR, OUTPUT_DIR, blend, get_holdout, is_counting, load_csv, norm


def combine(p_ft, p_zs, p_text, alpha):
    vlm = p_ft if p_zs is None else norm(p_ft + p_zs)
    return vlm if p_text is None else blend(vlm, p_text, alpha)


def load(path):
    return None if path is None else np.load(path).astype(np.float64)


def holdout_twin(path):
    """outputs/probs_x_test.npy -> outputs/probs_x_holdout.npy"""
    if path is None or not path.name.endswith("_test.npy"):
        return None
    p = path.with_name(path.name[: -len("_test.npy")] + "_holdout.npy")
    return p if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft", type=Path, required=True, help="파인튜닝 모델 test 확률")
    ap.add_argument("--zs", type=Path, default=None, help="zero-shot test 확률 (생략 가능)")
    ap.add_argument("--text", type=Path, default=None, help="텍스트 모델 test 확률 (생략 가능)")
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--dup", type=Path, default=OUTPUT_DIR / "dup_answers.csv")
    ap.add_argument("--out", type=Path, default=OUTPUT_DIR / "submission.csv")
    args = ap.parse_args()

    # ---------- 홀드아웃 점검 (파일이 있을 때만) ----------
    h_ft, h_zs, h_tx = holdout_twin(args.ft), holdout_twin(args.zs), holdout_twin(args.text)
    if h_ft is not None and (args.zs is None or h_zs is not None) \
            and (args.text is None or h_tx is not None):
        va = get_holdout(load_csv("train.csv"))
        gold = va["answer"].astype(str).str.strip().values
        m = is_counting(va["question"]).values
        P = combine(load(h_ft), load(h_zs), load(h_tx), args.alpha)
        pred = CH[P.argmax(1)]
        print(f"홀드아웃 {len(va)}건  전체 {(pred == gold).mean():.4f}"
              f"  (카운팅 {(pred[m] == gold[m]).mean():.4f} / 기타 {(pred[~m] == gold[~m]).mean():.4f})")

    # ---------- test ----------
    test = load_csv("test.csv")
    P = combine(load(args.ft), load(args.zs), load(args.text), args.alpha)
    assert len(P) == len(test) and not np.isnan(P).any(), "확률 파일 행 수/결측 확인"
    sub = pd.DataFrame({"id": test["id"].values, "answer": CH[P.argmax(1)]})

    if args.dup.exists():
        dup = pd.read_csv(args.dup).set_index("id")["answer"]
        hit = sub["id"].isin(dup.index)
        changed = int((sub.loc[hit, "answer"] != sub.loc[hit, "id"].map(dup)).sum())
        sub.loc[hit, "answer"] = sub.loc[hit, "id"].map(dup)
        print(f"중복 이미지 정답 {int(hit.sum())}건 적용 ({changed}건 변경)")

    # ---------- 제출 형식 검증 ----------
    sample_path = DATA_DIR / "sample_submission.csv"
    if sample_path.exists():
        sample = pd.read_csv(sample_path)
        sub = sample[["id"]].merge(sub, on="id", how="left")
        assert (sub["id"].values == sample["id"].values).all(), "id 순서 불일치"
    assert sub["answer"].notna().all() and sub["answer"].isin(CH).all()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.out, index=False)
    print("예측 분포:", sub["answer"].value_counts().sort_index().to_dict())
    print(f"저장: {args.out}  ({len(sub)}행)")


if __name__ == "__main__":
    main()
