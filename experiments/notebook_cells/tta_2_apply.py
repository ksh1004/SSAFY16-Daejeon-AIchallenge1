# =====================================================================
#  TTA 적용 — test 저신뢰 구간 (tta_1_check.py 를 같은 커널에서 실행한 뒤)
#  K=40 -> 약 2,029건 / 약 24분
# =====================================================================
K = 40

P_te = bl(probs, t_te)
conf_t = P_te.max(1)
idx_t = np.where(conf_t <= np.percentile(conf_t, K))[0]
print(f"test 하위 {K}% = {len(idx_t)}건 -> 예상 {len(idx_t)*0.71/60:.0f}분")

q_te = tta_pass(test_df, idx_t)                      # VLM 확률만 TTA

p_te_new = probs.copy()
p_te_new[idx_t] = norm(0.5 * probs[idx_t] + 0.5 * q_te)
P_te_new = bl(p_te_new, t_te)                        # 텍스트 블렌딩 다시 적용
preds = CH[P_te_new.argmax(1)]
np.save(ROOT / "vlm7b_probs_test_tta.npy", p_te_new)

sub = pd.DataFrame({"id": test_df["id"].values, "answer": preds})
dup = pd.read_csv(ROOT / "dup_answers.csv").set_index("id")["answer"]
hit = sub["id"].isin(dup.index)
sub.loc[hit, "answer"] = sub.loc[hit, "id"].map(dup)

prev = pd.read_csv(ROOT / "submission_7B_blend_a0.75.csv")
print(f"직전 제출(LB 0.8443) 대비 변경 {(sub.answer.values != prev.answer.values).sum()}건")
print("예측 분포:", dict(sub.answer.value_counts().sort_index()))

write_submission(sub["id"].values, sub["answer"].values, ROOT / f"submission_7B_tta{K}.csv")
print(f"✅ 저장: submission_7B_tta{K}.csv")
