# =====================================================================
#  카운팅 전용 "숫자 직접 스코어링" — 노트북 맨 아래에 새 셀로 붙여넣기
#
#  현재 방식은 모델에게 두 가지를 동시에 시킵니다:
#    (1) 물체를 센다   (2) 그 개수와 일치하는 '글자'를 찾는다
#  (2)는 VLM이 자주 틀리는 간접 단계입니다. 이걸 제거하고 (1)만 시킵니다.
#
#  근거 (train/test 실측):
#    - 카운팅 선지 4개가 모두 "N개" 형태  : train 84.1% / test 84.5%
#    - 선지 숫자가 중복되는 문항          : 0.00%  -> 숫자를 맞히면 선지가 유일하게 결정
#    - 정답 숫자가 1~9 범위               : 96.3%  -> 한 글자 토큰으로 스코어링 가능
#
#  비용: 홀드아웃 카운팅 210건만 평가 = 약 1.5분. 효과 있으면 테스트 1,713건에 적용.
# =====================================================================

import re
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

# ---------------------------------------------------------------- 셀 1
# 숫자 토큰 확인 (0~9가 각각 단일 토큰이어야 함)
DIGIT_IDS = {}
for d in range(10):
    ids = tok.encode(str(d), add_special_tokens=False)
    assert len(ids) == 1, f"'{d}' 가 단일 토큰이 아님: {ids}"
    DIGIT_IDS[d] = ids[0]
print("숫자 토큰 id:", DIGIT_IDS)


def opt_num(v):
    """선지 텍스트 -> 정수. 파싱 불가면 None."""
    s = str(v).strip()
    if s in ("없음", "없다", "0개", "0"):
        return 0
    if "이상" in s or "미만" in s or s.count("개") > 1:   # '4개 이상', '캔 2개, 병 1개'
        return None
    m = re.fullmatch(r"\s*(\d+)\s*개?\s*", s)
    return int(m.group(1)) if m else None


def numeric_row(row):
    """이 문항에 숫자 스코어링을 쓸 수 있는가 + 선지별 숫자."""
    ns = [opt_num(row[k]) for k in "abcd"]
    ok = (all(n is not None and 0 <= n <= 9 for n in ns) and len(set(ns)) == 4)
    return ok, ns


NUM_SYSTEM = ("You are a visual question answering assistant. "
              "Count carefully and answer with a single digit only. No explanation.")


@torch.no_grad()
def predict_counting(df):
    """선지에 등장하는 숫자 토큰들만 비교해 (N,4) 확률 반환."""
    model.eval()
    out = np.full((len(df), 4), np.nan)
    B = INFER_BATCH
    for s in tqdm(range(0, len(df), B), desc="숫자 스코어링"):
        sub = df.iloc[s:s + B]
        texts, images, numsets, rows = [], [], [], []
        for j, (_, row) in enumerate(sub.iterrows()):
            ok, ns = numeric_row(row)
            if not ok:
                continue
            img = Image.open(ROOT / row["path"]).convert("RGB")
            msgs = [
                {"role": "system", "content": [{"type": "text", "text": NUM_SYSTEM}]},
                {"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": f"{row['question']}\n숫자 한 글자만 출력하세요."}]},
            ]
            texts.append(processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            images.append(img); numsets.append(ns); rows.append(s + j)
        if not texts:
            continue
        processor.tokenizer.padding_side = "left"
        enc = processor(text=texts, images=images, padding=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items() if isinstance(v, torch.Tensor)}
        logits = model(**enc).logits[:, -1, :].float()
        for i, (gi, ns) in enumerate(zip(rows, numsets)):
            ids = [DIGIT_IDS[n] for n in ns]
            out[gi] = torch.softmax(logits[i, ids], dim=-1).cpu().numpy()
    return out


# ---------------------------------------------------------------- 셀 2
# 홀드아웃의 카운팅 문항으로 검증 (약 1.5분)
CH = np.array(CHOICES)
is_cnt_va = is_counting(va_part["question"]).values
va_cnt = va_part[is_cnt_va].reset_index(drop=True)
gold_cnt = va_cnt["answer"].astype(str).str.strip().values
print(f"홀드아웃 카운팅 {len(va_cnt)}건")

p_num = predict_counting(va_cnt)
applied = ~np.isnan(p_num).any(1)
print(f"숫자 스코어링 적용 가능: {applied.sum()}건 ({applied.mean():.1%})")

# 기존 글자 방식과 비교 (같은 문항에서만)
p_let = predict_probs(va_cnt, desc="글자 방식")
acc_let = (CH[p_let[applied].argmax(1)] == gold_cnt[applied]).mean()
acc_num = (CH[p_num[applied].argmax(1)] == gold_cnt[applied]).mean()
print(f"\n적용 가능 {applied.sum()}건 기준")
print(f"  글자 스코어링(현행) : {acc_let:.4f}")
print(f"  숫자 스코어링(신규) : {acc_num:.4f}   ({(acc_num-acc_let)*100:+.2f}%p)")
print(f"  표준오차 약 ±{100*(0.25/applied.sum())**0.5:.1f}%p")

# 두 방식 평균도 확인
for w in [0.3, 0.5, 0.7]:
    mix = w * p_num[applied] + (1 - w) * p_let[applied]
    print(f"  혼합 w={w:.1f}          : {(CH[mix.argmax(1)]==gold_cnt[applied]).mean():.4f}")


# ---------------------------------------------------------------- 셀 3
# 효과가 있으면(+2%p 이상) 테스트 전체에 적용
# probs = 셀 32에서 만든 (5074,4) 글자 방식 확률
W_NUM = 1.0          # 셀 2 결과를 보고 조정 (1.0=숫자만, 0.5=혼합)

is_cnt_te = is_counting(test_df["question"]).values
te_cnt = test_df[is_cnt_te].reset_index(drop=True)
print(f"테스트 카운팅 {len(te_cnt)}건 — 약 {len(te_cnt)*242/609/60:.0f}분 예상")

p_num_te = predict_counting(te_cnt)
ok_te = ~np.isnan(p_num_te).any(1)

probs_new = probs.copy()
idx_cnt = np.where(is_cnt_te)[0]
tgt = idx_cnt[ok_te]
probs_new[tgt] = W_NUM * p_num_te[ok_te] + (1 - W_NUM) * probs[tgt]

preds_new = CH[probs_new.argmax(1)]
print(f"적용 {ok_te.sum()}건 / 예측이 바뀐 문항 {(preds_new != preds).sum()}건")

np.save(ROOT / f"vlm_probs_test_{run_tag()}_numcount.npy", probs_new)
SUB_NUM = ROOT / f"submission_{run_tag()}_numcount.csv"
write_submission(test_df["id"].values, preds_new, SUB_NUM)
print("✅ 저장:", SUB_NUM.name)
