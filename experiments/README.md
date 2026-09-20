# experiments — 실험 기록

대회 중에 실제로 실행한 코드를 **수정하지 않고** 보존한 폴더입니다.
최종 파이프라인은 저장소 루트의 스크립트이고, 여기 있는 것은 판단 근거를 남기기 위한 기록입니다.

> ⚠️ 원래 프로젝트 루트에 CSV·이미지·확률 파일이 함께 있는 평면 구조에서 실행됐습니다.
> 경로와 파일명이 루트 스크립트와 다르므로 그대로는 돌지 않습니다.

## notebook_cells/ — 메인 노트북 커널에 붙여넣어 실행한 셀

`notebooks/train_and_zeroshot_desktop.ipynb` 로 모델을 올린 상태에서, 노트북 맨 아래에
새 셀로 붙여넣어 썼습니다. `model`, `processor`, `va_part`, `predict_probs` 등 노트북 전역
변수를 그대로 사용하므로 단독 실행되지 않습니다.

| 파일 | 내용 | 결과 |
|---|---|---|
| `finetune_check.py` | 7B QLoRA 파인튜닝 + 짝지은 검증 (33분 시간 상한) | ✅ 홀드아웃 400건 +2.75%p |
| `blend_cells.py` | 텍스트 전용 모델과의 블렌딩 α 탐색 | ✅ α=0.75, +1.5%p |
| `final_cells.py` | 사전분포 교정 + 블렌딩 + 중복 이미지 정답 적용 | 교정은 기각, 블렌딩 채택 |
| `tta_1_check.py` / `tta_2_apply.py` | 저신뢰 구간 선지 순서 TTA (검증 / 적용) | zero-shot에서 +0.99%p, 파인튜닝 후 무효 |
| `counting_cells.py` / `counting_v2.py` | 카운팅 문항을 숫자 토큰으로 직접 스코어링 | ❌ 0.6818 < 글자 방식 0.6919 |
| `binary_refine.py`, `refine_1_check.py` / `refine_2_apply.py` | 저신뢰 문항 2지선다 재질의 | ❌ −1.81%p |

## scripts/ — 노트북 8GB 환경에서 실행한 단독 스크립트

`infer_ft.py` 를 공통 모듈로 import 합니다 (`infer_ft` 를 반드시 `transformers` 보다 먼저
import 해야 로컬 모델 캐시 경로가 적용됩니다).

| 파일 | 내용 | 결과 |
|---|---|---|
| `infer_ft.py` | 어댑터 추론 원본 (재개 가능, `logits_to_keep` 최적화 동등성 검사 포함) | 루트 `infer.py` 의 원형 |
| `tta_ft_check.py` | 파인튜닝 확률에서 선지 순서 TTA 재검증 | ❌ −0.16%p |
| `multiscale_check.py` | 512/768px · 좌우반전 앙상블 | ❌ 최대 +0.82%p |
| `res_sweep.py` | 카운팅 문항만 고해상도(768/896px) | ❌ 홀드아웃 +1.81%p → **Public −0.16%p** |
| `cot_counting.py` | 카운팅 문항 Chain-of-Thought 후 스코어링 | ❌ −8.57%p (p=0.027) |

`res_sweep.py` 는 **선택 편향의 사례**로 남겨둡니다. 같은 홀드아웃 609건에서 30개 넘는
구성을 평가하고 최대값을 고른 결과, 홀드아웃이 예측한 개선이 리더보드에서 사라졌습니다.
자세한 분석은 [`docs/DEVELOPMENT_LOG.md`](../docs/DEVELOPMENT_LOG.md) §10 참고.
