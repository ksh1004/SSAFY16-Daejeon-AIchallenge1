# =====================================================================
#  마감 전 마무리 — 노트북 맨 아래에 붙여넣고 순서대로 실행
#  GPU는 셀 1(12분)만 씁니다. 셀 2·3은 전부 CPU라 즉시 끝납니다.
# =====================================================================
import numpy as np, pandas as pd

CH = np.array(CHOICES)
gold_va = va_part["answer"].astype(str).str.strip().values
is_cnt_va = is_counting(va_part["question"]).values

# ---------------------------------------------------------------- 셀 1  (GPU 12분)
# 7B 홀드아웃 확률 확보 — 이거 하나로 아래 개선 3종을 전부 CPU에서 검증합니다
p_va = predict_probs(va_part, desc="7B holdout")
np.save(ROOT / "vlm7b_probs_holdout.npy", p_va)
np.save(ROOT / "vlm7b_probs_test.npy", probs)      # probs = 셀 32의 테스트 확률
acc0 = (CH[p_va.argmax(1)] == gold_va).mean()
print(f"7B 홀드아웃 {acc0:.4f}  (셀 23에서 0.8276이었으므로 일치해야 정상)")


# ---------------------------------------------------------------- 셀 2  (CPU, 즉시)
def norm(p): return p / p.sum(1, keepdims=True)
def acc(p, m=None):
    pr = CH[p.argmax(1)]
    return (pr == gold_va).mean() if m is None else (pr[m] == gold_va[m]).mean()

print(f"기준          전체 {acc(p_va):.4f} / 카운팅 {acc(p_va,is_cnt_va):.4f} / 기타 {acc(p_va,~is_cnt_va):.4f}")
print(f"예측 분포     {dict(pd.Series(CH[p_va.argmax(1)]).value_counts().sort_index())}")

# --- (A) 사전분포 교정: train 정답이 a~d 각 25%인데 모델은 c를 과다 선택 ---
print("\n[A] 사전분포 교정  (tau=0 무보정, 1 완전제거)")
bestA = (0.0, acc(p_va))
for tau in np.arange(0, 1.01, 0.1):
    q = norm(p_va / p_va.mean(0, keepdims=True) ** tau)
    s = acc(q)
    print(f"  tau={tau:.1f}  {s:.4f}")
    if s > bestA[1]: bestA = (tau, s)
print(f"  -> 최적 tau={bestA[0]:.1f}  {bestA[1]:.4f}  ({(bestA[1]-acc0)*100:+.2f}%p)")

# --- (B) 텍스트 모델 블렌딩 ---
p_txt_va = np.load(ROOT / "text_probs_holdout.npy")
p_txt_te = np.load(ROOT / "text_probs_test.npy")
assert len(p_txt_va) == len(va_part) and len(p_txt_te) == len(probs)

def blend(pv, pt, a):
    return norm(np.clip(pv,1e-9,1)**a * np.clip(pt,1e-9,1)**(1-a))

print("\n[B] 텍스트 블렌딩")
bestB = (1.0, acc(p_va))
for a in np.arange(0.65, 1.001, 0.05):
    s = acc(blend(p_va, p_txt_va, a))
    print(f"  α={a:.2f}  {s:.4f}   (카운팅 {acc(blend(p_va,p_txt_va,a),is_cnt_va):.4f})")
    if s > bestB[1]: bestB = (a, s)
print(f"  -> 최적 α={bestB[0]:.2f}  {bestB[1]:.4f}  ({(bestB[1]-acc0)*100:+.2f}%p)")

# --- (C) A+B 결합 ---
print("\n[C] 교정 후 블렌딩")
bestC = (0.0, 1.0, acc0)
for tau in np.arange(0, 1.01, 0.2):
    pc = norm(p_va / p_va.mean(0, keepdims=True) ** tau)
    for a in np.arange(0.65, 1.001, 0.05):
        s = acc(blend(pc, p_txt_va, a))
        if s > bestC[2]: bestC = (tau, a, s)
print(f"  -> tau={bestC[0]:.1f}, α={bestC[1]:.2f}  {bestC[2]:.4f}  ({(bestC[2]-acc0)*100:+.2f}%p)")

print(f"\n{'='*56}")
print(f"기준(7B 단독)  {acc0:.4f}")
print(f"A 교정만       {bestA[1]:.4f}   ({(bestA[1]-acc0)*100:+.2f}%p)")
print(f"B 블렌딩만     {bestB[1]:.4f}   ({(bestB[1]-acc0)*100:+.2f}%p)")
print(f"C 둘 다        {bestC[2]:.4f}   ({(bestC[2]-acc0)*100:+.2f}%p)")
print("⚠️ 홀드아웃 609건에서 고른 값이라 낙관 편향(약 +0.5~1%p)이 있습니다.")
print("   상승폭이 1.5%p 미만이면 7B 단독으로 제출하세요.")


# ---------------------------------------------------------------- 셀 3  (CPU, 즉시)
# 위 결과를 보고 직접 지정 — 개선이 없으면 TAU=0, ALPHA=1.0 (=7B 단독)
TAU, ALPHA = bestC[0], bestC[1]

p_te = norm(probs / probs.mean(0, keepdims=True) ** TAU)
p_te = blend(p_te, p_txt_te, ALPHA)
preds_f = CH[p_te.argmax(1)]

# 중복 이미지 확정 정답 덮어쓰기 (train 라벨 기반, 무조건 옳음)
sub = pd.DataFrame({"id": test_df["id"].values, "answer": preds_f})
dup = pd.read_csv(ROOT / "dup_answers.csv").set_index("id")["answer"]
hit = sub["id"].isin(dup.index)
changed = (sub.loc[hit, "answer"].values != sub.loc[hit, "id"].map(dup).values).sum()
sub.loc[hit, "answer"] = sub.loc[hit, "id"].map(dup)
print(f"중복 이미지 확정답 {hit.sum()}건 적용 (그중 {changed}건 수정)")

print("예측 분포:", dict(sub.answer.value_counts().sort_index()))
OUT = ROOT / f"submission_7B_final_tau{TAU:.1f}_a{ALPHA:.2f}.csv"
write_submission(sub["id"].values, sub["answer"].values, OUT)
print("✅ 저장:", OUT.name)
