# =====================================================================
#  선지 순서 TTA — 저신뢰 구간 검증 (GPU 약 4분)
#  2지선다 실패와의 차이:
#    - 선지 4개를 그대로 유지하고 '순서'만 바꿈 (문제 형식 불변)
#    - TTA는 VLM 확률에만 적용하고, 텍스트 블렌딩은 그 뒤에 다시 얹음
#      (지난번엔 블렌딩된 확률을 덮어써서 텍스트 기여를 날렸음)
# =====================================================================
import numpy as np, pandas as pd, torch
from PIL import Image
from tqdm.auto import tqdm

CH = np.array(CHOICES)
if "p_va" not in dir():
    p_va = np.load(ROOT / "vlm_probs_holdout_full_7B_4bit_640px_zs_nodev.npy").astype(np.float64)
if "probs" not in dir():
    probs = np.load(ROOT / "vlm_probs_test_full_7B_4bit_640px_zs_nodev.npy").astype(np.float64)
t_va = np.load(ROOT / "text_probs_holdout.npy")
t_te = np.load(ROOT / "text_probs_test.npy")

def norm(p): return p / p.sum(1, keepdims=True)
def bl(pv, pt, a=0.75): return norm(np.clip(pv,1e-9,1)**a * np.clip(pt,1e-9,1)**(1-a))

PERM = (1, 2, 3, 0)      # 화면 위치 j 에 원래 선지 PERM[j] 를 표시

@torch.no_grad()
def tta_pass(df, idx, perm=PERM):
    """idx 행을 선지 순서 바꿔 1회 재추론. 원래 선지 기준 (len(idx),4) 확률."""
    model.eval()
    out = np.zeros((len(idx), 4))
    cid = CHOICE_IDS.to(device)
    for s in tqdm(range(0, len(idx), INFER_BATCH), desc="TTA"):
        chunk = idx[s:s + INFER_BATCH]
        texts, images = [], []
        for i in chunk:
            row = df.iloc[i]
            o = [str(row[CHOICES[perm[j]]]) for j in range(4)]
            img = Image.open(ROOT / row["path"]).convert("RGB")
            msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_INSTRUCT}]},
                    {"role": "user", "content": [{"type": "image", "image": img},
                     {"type": "text", "text": build_mc_prompt(str(row["question"]), *o)}]}]
            texts.append(processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            images.append(img)
        processor.tokenizer.padding_side = "left"
        enc = processor(text=texts, images=images, padding=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items() if isinstance(v, torch.Tensor)}
        lg = model(**enc).logits[:, -1, :]
        q = torch.softmax(lg[:, cid].float(), -1).cpu().numpy()
        for k in range(len(chunk)):
            for j in range(4):
                out[s + k, perm[j]] = q[k, j]          # 화면 j -> 원래 perm[j]
    return out


# ---- 검증: 홀드아웃 하위 20% ----
gold = va_part["answer"].astype(str).str.strip().values
P_va = bl(p_va, t_va)
conf = P_va.max(1)
idx = np.where(conf <= np.percentile(conf, 20))[0]
print(f"홀드아웃 하위 20% = {len(idx)}건, 현재 {(CH[P_va[idx].argmax(1)]==gold[idx]).mean():.4f}")

q_tta = tta_pass(va_part, idx)                          # VLM 확률만 TTA

p_new = p_va.copy()
p_new[idx] = norm(0.5 * p_va[idx] + 0.5 * q_tta)        # 원본 + 치환 평균
P_new = bl(p_new, t_va)                                 # 텍스트 블렌딩 다시 적용

s_old = (CH[P_va[idx].argmax(1)] == gold[idx]).mean()
s_new = (CH[P_new[idx].argmax(1)] == gold[idx]).mean()
a_old = (CH[P_va.argmax(1)] == gold).mean()
a_new = (CH[P_new.argmax(1)] == gold).mean()
print(f"\n대상 {len(idx)}건 : {s_old:.4f} -> {s_new:.4f}  ({(s_new-s_old)*100:+.1f}%p)")
print(f"전체 609건 : {a_old:.4f} -> {a_new:.4f}  ({(a_new-a_old)*100:+.2f}%p)")
print(f"표준오차 대상 ±{100*(0.25/len(idx))**0.5:.1f}%p / 전체 ±2.0%p")
print("\n★ 전체 +1.5%p 이상이면 tta_2_apply.py 진행 (약 35분)")
