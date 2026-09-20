# =====================================================================
#  7B QLoRA 파인튜닝 검증 (자체 완결형 — 노트북 맨 아래 새 셀에 붙여넣기)
#
#  목적: "파인튜닝이 zero-shot(0.8276)을 넘는가?" 하나만 답한다.
#  설계: 이미 로드된 7B / 640px 모델을 그대로 쓰므로 재로드 불필요.
#        검증은 홀드아웃 앞 400건으로 하고, 같은 400건의 zero-shot 점수는
#        저장된 .npy 에서 공짜로 계산해 '짝지은 비교'를 만든다.
#
#  예산: 학습 33분 + 평가 5분 = 약 38분
# =====================================================================
import gc, time, math, numpy as np, torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from tqdm.auto import tqdm

# ---------- 하이퍼파라미터 ----------
FT_SAMPLES   = 4000       # 시간 상한이 실질 제한. 넉넉히 잡고 시간으로 끊는다
FT_BATCH     = 1          # 7B 4bit + 640px + VRAM 여유 없음 -> 1
FT_ACCUM     = 8          # 유효 배치 8
FT_LR        = 1e-4
FT_MAX_MIN   = 33         # ★ 학습 시간 상한
EVAL_N       = 400        # 홀드아웃 앞 400건으로 검증 (표준오차 ±2.5%p)

import torch.nn.functional as F

# 노트북에서는 run_training() 안에 정의돼 있어 전역에 없으므로 여기서 다시 정의
def masked_weighted_loss(logits, labels, last, weight):
    """정답 위치 1개에 대한 CE를 샘플 가중치로 가중평균."""
    B = logits.size(0)
    r = torch.arange(B, device=logits.device)
    sel = logits[r, last - 1, :].float()      # 위치 t의 logits가 t+1을 예측
    tgt = labels[r, last]
    ce = F.cross_entropy(sel, tgt, reduction="none")
    return (ce * weight).sum() / weight.sum().clamp(min=1e-8)

CH = np.array(CHOICES)
gc.collect(); torch.cuda.empty_cache()
print(f"학습 전 VRAM 여유 {torch.cuda.mem_get_info()[0]/1e9:.1f} GB")

# ---------- 짝지은 비교 기준: 같은 400건의 zero-shot ----------
p_va_saved = np.load(ROOT / "vlm_probs_holdout_full_7B_4bit_640px_zs_nodev.npy").astype(np.float64)
va_eval  = va_part.head(EVAL_N).reset_index(drop=True)
gold_e   = va_eval["answer"].astype(str).str.strip().values
zs_e     = (CH[p_va_saved[:EVAL_N].argmax(1)] == gold_e).mean()
mc_e     = is_counting(va_eval["question"]).values
print(f"기준 zero-shot (같은 {EVAL_N}건) : {zs_e:.4f}"
      f"  (카운팅 {(CH[p_va_saved[:EVAL_N].argmax(1)][mc_e]==gold_e[mc_e]).mean():.4f})")

# ---------- 학습 ----------
ft_df = tr_part.head(FT_SAMPLES).reset_index(drop=True)
if "weight" not in ft_df.columns: ft_df["weight"] = 1.0
loader = DataLoader(VQADataset(ft_df, train=True, aug=USE_AUG), batch_size=FT_BATCH,
                    shuffle=True, collate_fn=Collator(processor, True),
                    num_workers=NUM_WORKERS, drop_last=False)
total = math.ceil(len(loader) / FT_ACCUM)
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=FT_LR, weight_decay=0.01)
sch = get_cosine_schedule_with_warmup(opt, max(1, int(total * 0.03)), total)
print(f"학습 샘플 {len(ft_df)} / 최대 {total} step / 상한 {FT_MAX_MIN}분")

model.train(); opt.zero_grad(set_to_none=True)
t0, done, run, seen = time.time(), 0, 0.0, 0
pbar = tqdm(loader, desc="7B QLoRA", unit="b")
for step, batch in enumerate(pbar, 1):
    w    = batch.pop("_weight").to(device)
    last = batch.pop("_last").to(device)
    batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
    lab = batch.pop("labels")
    loss = masked_weighted_loss(model(**batch).logits, lab, last, w)
    (loss / FT_ACCUM).backward()
    run += loss.item(); seen += 1
    if step % FT_ACCUM == 0 or step == len(loader):
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sch.step(); opt.zero_grad(set_to_none=True); done += 1
        pbar.set_postfix({"loss": f"{run/max(seen,1):.3f}", "step": done})
        run, seen = 0.0, 0
        if (time.time() - t0) / 60 > FT_MAX_MIN:
            tqdm.write(f"\n⏰ {FT_MAX_MIN}분 경과 — 학습 종료 (step {done}, 샘플 {step})")
            break
print(f"\n학습 완료: {done} optimizer step / {time.time()-t0:.0f}초")

# ---------- 평가 ----------
gc.collect(); torch.cuda.empty_cache()
p_ft = predict_probs(va_eval, desc="학습후 평가")
ft_e = (CH[p_ft.argmax(1)] == gold_e).mean()

print("\n" + "=" * 54)
print(f"zero-shot   {zs_e:.4f}")
print(f"파인튜닝후   {ft_e:.4f}   ({(ft_e-zs_e)*100:+.2f}%p)")
print(f"  카운팅 {(CH[p_ft.argmax(1)][mc_e]==gold_e[mc_e]).mean():.4f}"
      f" / 기타 {(CH[p_ft.argmax(1)][~mc_e]==gold_e[~mc_e]).mean():.4f}")
print(f"표준오차 ±{100*(0.25/EVAL_N)**0.5:.1f}%p  (짝지은 비교라 실제 검출력은 더 높음)")
print("=" * 54)

np.save(ROOT / "vlm7b_probs_holdout_ft.npy", p_ft)
if ft_e > zs_e:
    model.save_pretrained(ROOT / "lora_7b_ft_check")
    print("어댑터 저장: lora_7b_ft_check  (집에서 이어서 학습 가능)")
else:
    print("zero-shot 미달 — 어댑터를 저장하지 않습니다.")
