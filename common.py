"""공통 설정 · 프롬프트 · 데이터 유틸.

모든 스크립트가 이 파일을 먼저 import 합니다. 경로는 환경변수로 바꿀 수 있습니다.

    DATA_DIR    대회 데이터 폴더 (기본: ./data)
    OUTPUT_DIR  확률·제출 파일 저장 폴더 (기본: ./outputs)
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", ROOT / "outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- 모델 ----------
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"   # 실험에 쓴 스냅샷
IMAGE_PX = 640                                                 # 어댑터 학습 해상도

# ---------- LoRA (학습·추론 동일해야 함) ----------
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# ---------- 홀드아웃 ----------
SEED = 42
VAL_RATIO = 0.12

# ---------- 블렌딩 ----------
ALPHA = 0.75            # P = P_vlm^α · P_text^(1-α)

CHOICES = list("abcd")
CH = np.array(CHOICES)

CNT_PAT = r"몇 개|몇개|개입니까|개인가요|몇 가지"


def is_counting(questions: pd.Series) -> pd.Series:
    """카운팅 문항 여부 (전체의 약 34%)."""
    return questions.astype(str).str.contains(CNT_PAT, regex=True)


# ---------- 프롬프트 (학습·추론이 한 글자도 달라서는 안 됨) ----------
SYSTEM_INSTRUCT = (
    "You are a helpful visual question answering assistant. "
    "Answer using exactly one letter among a, b, c, or d. No explanation."
)


def build_mc_prompt(question, a, b, c, d):
    return (
        f"{question}\n"
        f"(a) {a}\n(b) {b}\n(c) {c}\n(d) {d}\n\n"
        "정답을 반드시 a, b, c, d 중 하나의 소문자 한 글자로만 출력하세요."
    )


def build_messages(row, img):
    o = [str(row[k]) for k in "abcd"]
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_INSTRUCT}]},
        {"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": build_mc_prompt(str(row["question"]), *o)},
        ]},
    ]


# ---------- 데이터 ----------
def load_csv(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 가 없습니다. 대회 데이터를 {DATA_DIR} 에 두세요 (README '데이터 준비' 참고).")
    return pd.read_csv(path)


def get_holdout(train_df: pd.DataFrame) -> pd.DataFrame:
    """make_holdout.py 가 만든 id 목록 순서대로 홀드아웃 609건을 돌려준다."""
    ids_path = DATA_DIR / "holdout_ids.csv"
    if not ids_path.exists():
        raise FileNotFoundError(f"{ids_path} 가 없습니다. 먼저 `python make_holdout.py` 를 실행하세요.")
    ids = pd.read_csv(ids_path)["id"].tolist()
    return train_df.set_index("id").loc[ids].reset_index()


def norm(p):
    return p / p.sum(1, keepdims=True)


def blend(p_vlm, p_text, alpha=ALPHA):
    """로그공간 가중 기하평균."""
    return norm(np.clip(p_vlm, 1e-9, 1) ** alpha * np.clip(p_text, 1e-9, 1) ** (1 - alpha))
