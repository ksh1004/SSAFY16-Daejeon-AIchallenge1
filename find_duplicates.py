"""train 과 test 에 똑같은 이미지·질문이 있으면 train 정답을 test 로 옮긴다.

    python find_duplicates.py      ->  outputs/dup_answers.csv  (id, answer)

- 이미지는 파일 MD5 로 대조합니다.
- **질문까지 같아야** 옮깁니다. 같은 사진에 다른 질문이 붙은 경우가 훨씬 많습니다.
- 선지 순서가 바뀐 경우가 많아(실측 24건 중 18건) 정답 '글자'가 아니라
  정답 '텍스트'를 test 선지에서 찾아 글자를 다시 매깁니다.
- 같은 텍스트가 test 선지에 정확히 하나 있을 때만 옮깁니다.
"""
import hashlib
from collections import defaultdict

import pandas as pd

from common import CHOICES, DATA_DIR, OUTPUT_DIR, load_csv


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    train = load_csv("train.csv")
    test = load_csv("test.csv")
    train["_h"] = [md5(DATA_DIR / p) for p in train["path"]]
    test["_h"] = [md5(DATA_DIR / p) for p in test["path"]]

    by_hash = defaultdict(list)
    for i, h in enumerate(train["_h"]):
        by_hash[h].append(i)

    shared = int(test["_h"].isin(by_hash.keys()).sum())    # train 에 같은 이미지가 있는 test 파일 수
    rows, reordered = [], 0
    for _, t in test.iterrows():
        for i in by_hash.get(t["_h"], []):
            s = train.iloc[i]
            if str(s["question"]).strip() != str(t["question"]).strip():
                continue
            gold_letter = str(s["answer"]).strip()
            gold_text = str(s[gold_letter]).strip()
            hits = [c for c in CHOICES if str(t[c]).strip() == gold_text]
            if len(hits) == 1:
                rows.append((t["id"], hits[0]))
                reordered += hits[0] != gold_letter
            break

    out = OUTPUT_DIR / "dup_answers.csv"
    pd.DataFrame(rows, columns=["id", "answer"]).to_csv(out, index=False)
    print(f"train 에 같은 이미지가 있는 test 파일 {shared}장")
    print(f"질문까지 같아 정답을 옮긴 문항 {len(rows)}건 (선지 순서가 바뀐 것 {reordered}건)")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
