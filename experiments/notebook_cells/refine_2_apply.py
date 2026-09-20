# 셀 1 검증 후 실행. (셀 1을 같은 커널에서 이미 돌렸다면 함수/변수가 살아 있습니다)
# ---------------------------------------------------------------- 셀 2  적용 (약 35분)
conf_t = P_te.max(1)
idx_t = np.where(conf_t <= np.percentile(conf_t, K))[0]
print(f"test 하위 {K}% = {len(idx_t)}건 -> 약 {len(idx_t)*2.05/60:.0f}분")

P_te_ref = refine(test_df, P_te, idx_t)
preds = CH[P_te_ref.argmax(1)]
np.save(ROOT / "vlm7b_probs_test_refined.npy", P_te_ref)

sub = pd.DataFrame({"id": test_df["id"].values, "answer": preds})
dup = pd.read_csv(ROOT / "dup_answers.csv").set_index("id")["answer"]
hit = sub["id"].isin(dup.index); sub.loc[hit, "answer"] = sub.loc[hit, "id"].map(dup)
prev = pd.read_csv(ROOT / "submission_7B_blend_a0.75.csv")
print(f"직전 제출 대비 변경 {(sub.answer.values != prev.answer.values).sum()}건")
write_submission(sub["id"].values, sub["answer"].values, ROOT / "submission_7B_refine.csv")
print("✅ 저장: submission_7B_refine.csv")
