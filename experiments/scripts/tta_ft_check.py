# =====================================================================
#  선지 순서 TTA — 파인튜닝 확률에서도 유효한가 (GPU 약 12분)
#
#  기존 tta_1_check.py 는 zero-shot 확률에서 +0.98%p 를 봤습니다.
#  파인튜닝이 선지 순서 편향을 이미 줄였을 수 있으므로 다시 확인합니다.
#
#  덤: 홀드아웃 FT 확률을 400 -> 609건으로 확장합니다. 609건 홀드아웃은
#      LB 오프셋이 ~0 이라(400건은 -1.1%p) 이후 판단이 정확해집니다.
#
#  사용법
#    .venv/Scripts/python.exe tta_ft_check.py --batch 2
# =====================================================================
import argparse
import numpy as np
import pandas as pd

from infer_ft import (ROOT, CHOICES, CH, is_counting, load_model, predict_probs,
                      get_holdout)

ALPHA = 0.75
PERM  = (1, 2, 3, 0)          # 화면 위치 j 에 원래 선지 PERM[j] 를 표시
K_LIST = [20, 40, 60, 100]    # 저신뢰 하위 K% 에만 TTA 적용

def norm(p): return p / p.sum(1, keepdims=True)
def bl(pv, pt, a=ALPHA):
    return norm(np.clip(pv, 1e-9, 1) ** a * np.clip(pt, 1e-9, 1) ** (1 - a))


def tta_probs(model, processor, df, batch, ckpt, perm=PERM, desc="tta"):
    """선지 순서를 바꿔 재추론하고, 원래 선지 기준으로 되돌린 확률을 반환."""
    d = df.copy().reset_index(drop=True)
    orig = df[["a", "b", "c", "d"]].astype(str).values
    for j, src in enumerate(perm):
        d[CHOICES[j]] = orig[:, src]          # 화면 위치 j <- 원래 선지 src
    p = predict_probs(model, processor, d, batch, ckpt, desc=desc)
    out = np.zeros_like(p)
    for j, src in enumerate(perm):
        out[:, src] = p[:, j]                 # 화면 위치 j -> 원래 선지 src
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=2)
    args = ap.parse_args()

    train_df = pd.read_csv(ROOT / "train.csv")
    va = get_holdout(train_df)                        # 609건, 저장된 .npy 와 같은 순서
    gold = va["answer"].astype(str).str.strip().values
    mcnt = is_counting(va["question"]).values
    n = len(va)

    model, processor = load_model()

    # ---------- 1. 홀드아웃 609건 FT 확률 (400 -> 609 확장) ----------
    ft_path = ROOT / "vlm7b_probs_holdout609_ft.npy"
    if ft_path.exists():
        p_ft = np.load(ft_path)
        print(f"기존 {ft_path.name} 사용")
    else:
        p_ft = predict_probs(model, processor, va, args.batch,
                             ROOT / "_ckpt_holdout609_ft.npy", desc="holdout609 FT")
        np.save(ft_path, p_ft)

    p_zs  = np.load(ROOT / "vlm_probs_holdout_full_7B_4bit_640px_zs_nodev.npy").astype(np.float64)
    p_txt = np.load(ROOT / "text_probs_holdout.npy").astype(np.float64)
    assert len(p_zs) == len(p_txt) == n

    def acc(p, m=None):
        pr = CH[p.argmax(1)]
        return (pr == gold).mean() if m is None else (pr[m] == gold[m]).mean()

    # 현재 최고 제출(LB 0.86322)에 대응하는 구성 = 기준선
    p_base = norm(p_ft + p_zs)
    P_base = bl(p_base, p_txt)
    print("\n" + "=" * 62)
    print("홀드아웃 609건 — 400건보다 LB와 잘 맞습니다 (오프셋 ~0)")
    print("=" * 62)
    print(f"  zero-shot + 텍스트   {acc(bl(p_zs, p_txt)):.4f}   (제출 4: LB 0.84430)")
    print(f"  파인튜닝 + 텍스트     {acc(bl(p_ft, p_txt)):.4f}")
    print(f"  ft+zs + 텍스트       {acc(P_base):.4f}   (제출 6: LB 0.86322)  ← 기준선")
    print(f"    카운팅 {acc(P_base, mcnt):.4f} / 기타 {acc(P_base, ~mcnt):.4f}")

    # ---------- 2. TTA — 전체 609건에 1회 치환 ----------
    # 하위 K% 만 돌리면 K를 바꿔 재평가할 수 없으므로, 검증에서는 전체를 돌리고
    # 사후에 K를 잘라 봅니다 (609건 = 약 7분, 어차피 싸다).
    q_ft = tta_probs(model, processor, va, args.batch,
                     ROOT / "_ckpt_holdout609_ft_tta.npy", desc="holdout609 TTA")
    np.save(ROOT / "vlm7b_probs_holdout609_ft_tta.npy", q_ft)

    conf = P_base.max(1)
    print("\n" + "=" * 62)
    print("TTA 효과 — 저신뢰 하위 K% 에만 적용 (VLM 확률만, 텍스트는 뒤에 재적용)")
    print("=" * 62)
    a0 = acc(P_base)
    best = (0, a0)
    for K in K_LIST:
        idx = np.where(conf <= np.percentile(conf, K))[0] if K < 100 else np.arange(n)
        p_new = p_ft.copy()
        p_new[idx] = norm(0.5 * p_ft[idx] + 0.5 * q_ft[idx])
        P_new = bl(norm(p_new + p_zs), p_txt)
        a = acc(P_new)
        s_old = (CH[P_base[idx].argmax(1)] == gold[idx]).mean()
        s_new = (CH[P_new[idx].argmax(1)] == gold[idx]).mean()
        print(f"  K={K:3d}%  대상 {len(idx):3d}건 {s_old:.4f}->{s_new:.4f}"
              f"   전체 {a0:.4f}->{a:.4f}  ({(a-a0)*100:+.2f}%p)")
        if a > best[1]:
            best = (K, a)

    print("\n" + "=" * 62)
    print(f"최선 K={best[0]}%  전체 {best[1]:.4f}  ({(best[1]-a0)*100:+.2f}%p)")
    print(f"609건 표준오차 ±{100*(0.25/n)**0.5:.1f}%p")
    print("판단 기준 (HANDOFF §3): 탐색으로 고른 값이므로 +1.5%p 이상이어야 채택.")
    if (best[1] - a0) * 100 >= 1.5:
        print(f"  -> 채택. test 하위 {best[0]}% 에 TTA 적용 "
              f"(약 {5074*best[0]/100*0.7/60:.0f}분). 예상 LB {0.86322+(best[1]-a0):.4f}")
    else:
        print("  -> 기각. 파인튜닝 확률에서는 TTA 이득이 노이즈 수준입니다.")
    print("=" * 62)


if __name__ == "__main__":
    main()
