# =====================================================================
#  저신뢰 문항 2지선다 재질의 (binary refinement)
#
#  근거 (홀드아웃 609건):
#    신뢰도 하위 20%  ->  top1 정확도 0.525 / top2 포함률 0.836
#    즉 "정답을 2개로 좁히는 것"은 되는데 "둘 중 고르기"에서 틀립니다.
#    -> 오답 2개를 지우고 2지선다로 다시 물으면 개선 여지가 있습니다.
#
#  셀 1(검증) GPU 약 4분 / 셀 2(적용) GPU 약 35분
# =====================================================================
import numpy as np, pandas as pd, torch
from PIL import Image
from tqdm.auto import tqdm

CH = np.array(CHOICES)
if "p_va" not in dir():
    p_va = np.load(ROOT / "vlm_probs_holdout_full_7B_4bit_640px_zs_nodev.npy").astype(np.float64)
if "probs" not in dir():
    probs = np.load(ROOT / "vlm_probs_test_full_7B_4bit_640px_zs_nodev.npy").astype(np.float64)

def norm(p): return p / p.sum(1, keepdims=True)
def bl(pv, pt, a): return norm(np.clip(pv,1e-9,1)**a * np.clip(pt,1e-9,1)**(1-a))

t_va = np.load(ROOT / "text_probs_holdout.npy")
t_te = np.load(ROOT / "text_probs_test.npy")
P_va = bl(p_va, t_va, 0.75)          # 현재 제출과 동일한 블렌딩
P_te = bl(probs, t_te, 0.75)

BIN_SYS = ("You are a visual question answering assistant. "
           "Answer using exactly one letter, a or b. No explanation.")

@torch.no_grad()
def refine(df, P, idx):
    """idx 행들에 대해 top-2만 남긴 2지선다로 재질의. 갱신된 확률 반환."""
    model.eval()
    out = P.copy()
    ab = [tok.encode(x, add_special_tokens=False)[0] for x in ("a", "b")]
    for s in tqdm(range(0, len(idx), INFER_BATCH), desc="2지선다 재질의"):
        chunk = idx[s:s + INFER_BATCH]
        texts, images, pairs_ = [], [], []
        for i in chunk:
            row = df.iloc[i]
            top2 = sorted(np.argsort(-P[i])[:2])      # 원래 순서 유지(위치 편향 방지)
            o1, o2 = str(row[CHOICES[top2[0]]]), str(row[CHOICES[top2[1]]])
            img = Image.open(ROOT / row["path"]).convert("RGB")
            txt = (f"{row['question']}\n(a) {o1}\n(b) {o2}\n\n"
                   "정답을 반드시 a 또는 b 중 하나의 소문자 한 글자로만 출력하세요.")
            msgs = [{"role": "system", "content": [{"type": "text", "text": BIN_SYS}]},
                    {"role": "user", "content": [{"type": "image", "image": img},
                                                 {"type": "text", "text": txt}]}]
            texts.append(processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            images.append(img); pairs_.append((i, top2))
        processor.tokenizer.padding_side = "left"
        enc = processor(text=texts, images=images, padding=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items() if isinstance(v, torch.Tensor)}
        lg = model(**enc).logits[:, -1, :].float()
        for k, (i, top2) in enumerate(pairs_):
            q2 = torch.softmax(lg[k, ab], -1).cpu().numpy()
            mass = P[i, top2].sum()                    # 원래 top-2 질량 유지, 내부 순위만 교체
            out[i, top2[0]] = q2[0] * mass
            out[i, top2[1]] = q2[1] * mass
    return norm(out)


# ---------------------------------------------------------------- 셀 1  검증 (약 4분)
gold = va_part["answer"].astype(str).str.strip().values
conf = P_va.max(1)
K = 20                                                  # 하위 K%
thr = np.percentile(conf, K)
idx = np.where(conf <= thr)[0]
print(f"홀드아웃 하위 {K}% = {len(idx)}건 (현재 정확도 {(CH[P_va[idx].argmax(1)]==gold[idx]).mean():.4f})")

P_ref = refine(va_part, P_va, idx)
a_old = (CH[P_va.argmax(1)] == gold).mean()
a_new = (CH[P_ref.argmax(1)] == gold).mean()
s_old = (CH[P_va[idx].argmax(1)] == gold[idx]).mean()
s_new = (CH[P_ref[idx].argmax(1)] == gold[idx]).mean()
print(f"\n대상 {len(idx)}건 : {s_old:.4f} -> {s_new:.4f}  ({(s_new-s_old)*100:+.1f}%p)")
print(f"전체 609건 : {a_old:.4f} -> {a_new:.4f}  ({(a_new-a_old)*100:+.2f}%p)")
print(f"표준오차 대상 ±{100*(0.25/len(idx))**0.5:.1f}%p / 전체 ±2.0%p")
print(f"참고: 이 구간의 top-2 포함률 0.836 = 이론상 상한")
print("\n★ 전체 +1.5%p 이상이면 셀 2 진행")
