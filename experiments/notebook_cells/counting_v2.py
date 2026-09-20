# =====================================================================
#  카운팅 숫자 스코어링 — 검증용 (노트북 맨 아래에 붙여넣기)
#  GPU 약 4분. p_va(홀드아웃 확률)를 재사용하므로 글자 방식은 다시 계산하지 않습니다.
# =====================================================================
import re, numpy as np, pandas as pd, torch
from PIL import Image
from tqdm.auto import tqdm

CH = np.array(CHOICES)

# --- 숫자 토큰 ---
DIGIT_IDS = {}
for d in range(10):
    ids = tok.encode(str(d), add_special_tokens=False)
    assert len(ids) == 1, f"'{d}' 단일 토큰 아님: {ids}"
    DIGIT_IDS[d] = ids[0]
print("숫자 토큰:", DIGIT_IDS)

def opt_num(v):
    s = str(v).strip()
    if s in ("없음", "없다", "0개", "0"): return 0
    if "이상" in s or "미만" in s or s.count("개") > 1: return None
    m = re.fullmatch(r"\s*(\d+)\s*개?\s*", s)
    return int(m.group(1)) if m else None

def numeric_row(row):
    ns = [opt_num(row[k]) for k in "abcd"]
    return (all(n is not None and 0 <= n <= 9 for n in ns) and len(set(ns)) == 4), ns

NUM_SYS = ("You are a visual question answering assistant. "
           "Count carefully and answer with a single digit only. No explanation.")

@torch.no_grad()
def predict_counting(df):
    model.eval()
    out = np.full((len(df), 4), np.nan)
    for s in tqdm(range(0, len(df), INFER_BATCH), desc="숫자 스코어링"):
        sub = df.iloc[s:s + INFER_BATCH]
        texts, images, nss, gis = [], [], [], []
        for j, (_, row) in enumerate(sub.iterrows()):
            ok, ns = numeric_row(row)
            if not ok: continue
            img = Image.open(ROOT / row["path"]).convert("RGB")
            msgs = [{"role": "system", "content": [{"type": "text", "text": NUM_SYS}]},
                    {"role": "user", "content": [{"type": "image", "image": img},
                     {"type": "text", "text": f"{row['question']}\n숫자 한 글자만 출력하세요."}]}]
            texts.append(processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            images.append(img); nss.append(ns); gis.append(s + j)
        if not texts: continue
        processor.tokenizer.padding_side = "left"
        enc = processor(text=texts, images=images, padding=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items() if isinstance(v, torch.Tensor)}
        lg = model(**enc).logits[:, -1, :].float()
        for i, (gi, ns) in enumerate(zip(gis, nss)):
            out[gi] = torch.softmax(lg[i, [DIGIT_IDS[n] for n in ns]], -1).cpu().numpy()
    return out


# ---- 카운팅 부분집합 준비 (글자 확률은 p_va 재사용) ----
mc = is_counting(va_part["question"]).values
va_cnt = va_part[mc].reset_index(drop=True)
gold_c = va_cnt["answer"].astype(str).str.strip().values
p_let_c = p_va[mc]                                   # 이미 계산해둔 글자 방식 확률
p_txt_c = np.load(ROOT / "text_probs_holdout.npy")[mc]
print(f"홀드아웃 카운팅 {len(va_cnt)}건")

p_num_c = predict_counting(va_cnt)                   # ← GPU 약 4분
ok = ~np.isnan(p_num_c).any(1)
print(f"숫자 스코어링 적용 가능 {ok.sum()}건 ({ok.mean():.1%})")

def norm(p): return p / p.sum(1, keepdims=True)
def bl(pv, pt, a): return norm(np.clip(pv,1e-9,1)**a * np.clip(pt,1e-9,1)**(1-a))
def A(p): return (CH[p.argmax(1)] == gold_c[ok]).mean()

L, T, N = p_let_c[ok], p_txt_c[ok], p_num_c[ok]
print(f"\n적용 가능 {ok.sum()}건에서의 비교  (표준오차 ±{100*(0.25/ok.sum())**0.5:.1f}%p)")
print(f"  글자만                 {A(L):.4f}")
print(f"  글자+텍스트 α=0.55 ★현행 {A(bl(L,T,0.55)):.4f}")
print(f"  숫자만                 {A(N):.4f}")
for a in [0.5, 0.6, 0.7, 0.8]:
    print(f"  숫자+텍스트 α={a:.1f}      {A(bl(N,T,a)):.4f}")
for w in [0.3, 0.5, 0.7]:
    print(f"  글자·숫자 평균 w={w:.1f}    {A(norm(w*N+(1-w)*L)):.4f}")
for w in [0.5]:
    for a in [0.6, 0.7]:
        print(f"  (글자·숫자 w={w}) +텍스트 α={a:.1f}  {A(bl(norm(w*N+(1-w)*L), T, a)):.4f}")
print("\n★ 현행(글자+텍스트)을 +2%p 이상 넘지 못하면 적용하지 마세요.")
