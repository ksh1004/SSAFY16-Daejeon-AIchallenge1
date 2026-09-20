# =====================================================================
#  실행 중인 v2 노트북 "맨 아래에 새 셀로 붙여넣어" 사용하세요.
#  노트북을 다시 로드할 필요 없습니다. (셀 32까지 끝난 뒤 실행)
#
#  아이디어: VLM(0.7504)과 텍스트 전용 모델(0.5714)은 서로 다른 신호를 씁니다.
#           텍스트 모델은 VLM이 무시하는 데이터셋 편향(카운팅 정답이 작은 수에
#           쏠림 등)을 잡으므로, 확률을 블렌딩하면 둘 다보다 높아질 수 있습니다.
#  비용: 홀드아웃 확률 1회 계산(약 4분) 외에는 GPU 사용 없음.
# =====================================================================

# ---------------------------------------------------------------- 셀 A
# VLM 확률을 확보하고 디스크에 저장 (커널이 죽어도 잃지 않도록)
import numpy as np

CH = np.array(CHOICES)
gold_va = va_part["answer"].astype(str).str.strip().values

p_va_vlm = predict_probs(va_part, desc="holdout probs")      # 약 4분
np.save(ROOT / f"vlm_probs_holdout_{run_tag()}.npy", p_va_vlm)

# probs 는 셀 32에서 만들어진 테스트 확률 (5074, 4)
np.save(ROOT / f"vlm_probs_test_{run_tag()}.npy", probs)

acc_vlm = (CH[p_va_vlm.argmax(1)] == gold_va).mean()
print(f"VLM 홀드아웃 정확도 {acc_vlm:.4f}   (저장 완료)")


# ---------------------------------------------------------------- 셀 B
# 텍스트 확률 로드 + α 탐색
p_va_txt = np.load(ROOT / "text_probs_holdout.npy")
p_te_txt = np.load(ROOT / "text_probs_test.npy")
assert len(p_va_txt) == len(va_part) == len(p_va_vlm), "홀드아웃 길이 불일치"
assert len(p_te_txt) == len(probs) == len(test_df),    "테스트 길이 불일치"

def blend(pv, pt, a):
    """로그 공간 가중 기하평균. a=1이면 VLM 단독."""
    q = np.clip(pv, 1e-9, 1) ** a * np.clip(pt, 1e-9, 1) ** (1 - a)
    return q / q.sum(1, keepdims=True)

is_cnt = is_counting(va_part["question"]).values
print(f"{'α':>6} {'전체':>8} {'카운팅':>8} {'기타':>8}")
rows = []
for a in np.arange(0.60, 1.001, 0.025):
    pred = CH[blend(p_va_vlm, p_va_txt, a).argmax(1)]
    ov = (pred == gold_va).mean()
    ct = (pred[is_cnt] == gold_va[is_cnt]).mean()
    ot = (pred[~is_cnt] == gold_va[~is_cnt]).mean()
    rows.append((a, ov, ct, ot))
    print(f"{a:6.3f} {ov:8.4f} {ct:8.4f} {ot:8.4f}")

best_a, best_ov, _, _ = max(rows, key=lambda r: r[1])
gain = best_ov - acc_vlm
print(f"\n최적 α={best_a:.3f}  홀드아웃 {best_ov:.4f}   VLM 단독 {acc_vlm:.4f}   ({gain*100:+.2f}%p)")
print("⚠️ α를 홀드아웃에서 고른 만큼 낙관 편향이 있습니다(약 +0.5~1%p).")
print("   상승폭이 1.5%p 미만이면 블렌딩하지 말고 VLM 단독으로 제출하세요.")


# ---------------------------------------------------------------- 셀 C (선택)
# 문항 유형별로 α를 따로: 텍스트 모델은 카운팅에서 상대적으로 더 쓸모 있습니다
#   (텍스트 카운팅 0.5190 vs 기타 0.5990 / VLM 카운팅 0.6190 vs 기타 0.8195)
best2, sc2 = None, -1
for a_c in np.arange(0.60, 1.001, 0.025):
    for a_o in np.arange(0.60, 1.001, 0.025):
        pred = np.empty(len(va_part), dtype="<U1")
        pred[is_cnt]  = CH[blend(p_va_vlm[is_cnt],  p_va_txt[is_cnt],  a_c).argmax(1)]
        pred[~is_cnt] = CH[blend(p_va_vlm[~is_cnt], p_va_txt[~is_cnt], a_o).argmax(1)]
        s = (pred == gold_va).mean()
        if s > sc2: best2, sc2 = (a_c, a_o), s
print(f"유형별 α — 카운팅 {best2[0]:.3f} / 기타 {best2[1]:.3f}  홀드아웃 {sc2:.4f}")
print("⚠️ 2차원 탐색이라 과적합 위험이 더 큽니다. 단일 α보다 +1%p 이상일 때만 쓰세요.")


# ---------------------------------------------------------------- 셀 C2 (선택)
# 신뢰도 적응형 블렌딩:
#   VLM이 확신하는 문항은 VLM을 그대로 믿고,
#   헷갈려하는 문항(최대확률이 낮음)에서만 텍스트 모델 쪽으로 기울입니다.
#   α(x) = a_hi 일 때 VLM 확신, a_lo 일 때 불확실
conf = p_va_vlm.max(1)
print(f"VLM 확신도 분포: 하위25% {np.percentile(conf,25):.3f} / 중앙 {np.percentile(conf,50):.3f}"
      f" / 상위25% {np.percentile(conf,75):.3f}")
print(f"확신도 상위 절반 정확도 {(CH[p_va_vlm[conf>=np.median(conf)].argmax(1)]==gold_va[conf>=np.median(conf)]).mean():.4f}")
print(f"확신도 하위 절반 정확도 {(CH[p_va_vlm[conf<np.median(conf)].argmax(1)]==gold_va[conf<np.median(conf)]).mean():.4f}")

best3, sc3 = None, -1
for a_lo in np.arange(0.55, 1.001, 0.05):
    for a_hi in np.arange(a_lo, 1.001, 0.05):
        w = (conf - conf.min()) / (conf.max() - conf.min() + 1e-9)
        a = a_lo + (a_hi - a_lo) * w                     # 확신할수록 VLM 비중↑
        q = np.clip(p_va_vlm,1e-9,1)**a[:,None] * np.clip(p_va_txt,1e-9,1)**(1-a[:,None])
        s = (CH[(q/q.sum(1,keepdims=True)).argmax(1)] == gold_va).mean()
        if s > sc3: best3, sc3 = (a_lo, a_hi), s
print(f"신뢰도 적응형 — α_low {best3[0]:.2f} → α_high {best3[1]:.2f}   홀드아웃 {sc3:.4f}")
print("⚠️ 2차원 탐색이라 단일 α보다 +1%p 이상 나을 때만 채택하세요.")


# ---------------------------------------------------------------- 셀 D
# 채택하기로 했다면 제출 파일 생성 (α는 위 결과를 보고 직접 지정하세요)
ALPHA = best_a          # 또는 직접 숫자 입력
p_te_blend = blend(probs, p_te_txt, ALPHA)
preds_blend = CH[p_te_blend.argmax(1)]
print("예측 분포:", dict(pd.Series(preds_blend).value_counts().sort_index()))
print("VLM 단독과 다른 예측:", (preds_blend != preds).sum(), "건")

SUB_BLEND = ROOT / f"submission_{run_tag()}_blend{ALPHA:.3f}.csv"
write_submission(test_df["id"].values, preds_blend, SUB_BLEND)
print("✅ 저장:", SUB_BLEND.name)
