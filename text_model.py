"""이미지 없이 (질문 + 선지 4개)만으로 정답 확률을 낸다 — VLM 블렌딩 파트너.

    python text_model.py   ->  outputs/text_probs_holdout.npy / outputs/text_probs_test.npy

왜 필요한가: 이 데이터는 선지 구조 자체에 편향이 있습니다(카운팅 정답이 '2개'에 쏠림 등).
이미지를 전혀 안 보는 모델도 랜덤(0.25)보다 훨씬 높고, VLM 과 다른 신호를 쓰므로
로그공간 블렌딩(α=0.75)에서 +1.5%p 를 냈습니다.

설계: 4-클래스(위치) 분류가 아니라 **선지별 이진 분류**. "이 선지가 정답인가?"를 묻기
때문에 선지 텍스트 · 숫자 크기 · 숫자 순위 · 길이 순위를 전부 피처로 쓸 수 있습니다.
4개 점수를 softmax 해 확률로 만듭니다. 홀드아웃 609건은 학습에서 제외합니다.

⚠️ 최종 제출에 쓴 텍스트 모델(홀드아웃 0.6355)의 학습 코드는 보존되지 않았습니다.
   이 파일은 같은 목적의 재구현이며 홀드아웃 0.6141 입니다. README '재현성' 절 참고.
"""
import re

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from common import CH, OUTPUT_DIR, get_holdout, is_counting, load_csv

_NUM = re.compile(r"(\d+)")


def num_of(s):
    m = _NUM.search(str(s))
    return float(m.group(1)) if m else np.nan


def build_rows(df):
    """(N, 4) 문항을 (N*4,) 선지 행으로 편다."""
    q = df["question"].astype(str).values
    opts = df[["a", "b", "c", "d"]].astype(str).values
    nums = np.array([[num_of(o) for o in row] for row in opts])

    feats, texts_q, texts_o, texts_qo = [], [], [], []
    for i in range(len(df)):
        row_nums = nums[i]
        has = ~np.isnan(row_nums)
        order = np.full(4, np.nan)                     # 숫자 선지 사이 순위 (0~1)
        if has.sum() >= 2:
            vals = row_nums[has]
            order[has] = vals.argsort().argsort() / max(len(vals) - 1, 1)
        lens = np.array([len(o) for o in opts[i]], float)
        len_rank = lens.argsort().argsort() / 3.0
        for j in range(4):
            o = opts[i, j]
            texts_q.append(q[i])
            texts_o.append(o)
            texts_qo.append(q[i] + " ||| " + o)
            same_tok = sum(1 for k in range(4) if k != j and set(o.split()) & set(opts[i, k].split()))
            feats.append([
                j == 0, j == 1, j == 2, j == 3,                  # 위치
                has[j],                                          # 숫자 선지인가
                row_nums[j] if has[j] else 0.0,                  # 숫자 값
                order[j] if not np.isnan(order[j]) else 0.5,     # 숫자 순위
                float(has.sum()),                                # 숫자 선지 개수
                has[j] and row_nums[j] == np.nanmin(row_nums),   # 최솟값인가
                has[j] and row_nums[j] == np.nanmax(row_nums),   # 최댓값인가
                lens[j], len_rank[j],                            # 길이 / 길이 순위
                same_tok,                                        # 다른 선지와 토큰 공유
                float(o in q[i]),                                # 선지가 질문에 등장
                len(set(opts[i])) < 4,                           # 중복 선지 존재
            ])
    return np.array(feats, dtype=np.float64), texts_q, texts_o, texts_qo


def fit_predict(train_df, targets):
    Xf, tq, to, tqo = build_rows(train_df)
    gold = train_df["answer"].astype(str).str.strip().values
    y = np.concatenate([[CH[j] == g for j in range(4)] for g in gold]).astype(int)

    def tfidf(**kw):
        return TfidfVectorizer(min_df=3, max_features=60000, sublinear_tf=True, **kw)

    vec_o = tfidf(analyzer="char_wb", ngram_range=(2, 4))
    vec_q = tfidf(analyzer="char_wb", ngram_range=(2, 4))
    vec_qo = tfidf(analyzer="word", ngram_range=(1, 2))
    scaler = StandardScaler().fit(Xf)
    X = sparse.hstack([vec_o.fit_transform(to), vec_q.fit_transform(tq),
                       vec_qo.fit_transform(tqo), sparse.csr_matrix(scaler.transform(Xf))]).tocsr()
    clf = LogisticRegression(C=1.0, max_iter=3000, solver="liblinear").fit(X, y)

    out = []
    for df in targets:
        f, q, o, qo = build_rows(df)
        Xt = sparse.hstack([vec_o.transform(o), vec_q.transform(q), vec_qo.transform(qo),
                            sparse.csr_matrix(scaler.transform(f))]).tocsr()
        s = clf.decision_function(Xt).reshape(len(df), 4)
        e = np.exp(s - s.max(1, keepdims=True))
        out.append(e / e.sum(1, keepdims=True))
    return out


def main():
    train_df = load_csv("train.csv")
    test_df = load_csv("test.csv")
    va = get_holdout(train_df)
    tr = train_df[~train_df["id"].isin(va["id"])].reset_index(drop=True)
    print(f"학습 {len(tr)} / 홀드아웃 {len(va)} / test {len(test_df)}")

    p_va, p_te = fit_predict(tr, [va, test_df])

    gold = va["answer"].astype(str).str.strip().values
    m = is_counting(va["question"]).values
    pred = CH[p_va.argmax(1)]
    print(f"홀드아웃  전체 {(pred == gold).mean():.4f}"
          f"  (카운팅 {(pred[m] == gold[m]).mean():.4f} / 기타 {(pred[~m] == gold[~m]).mean():.4f})")

    np.save(OUTPUT_DIR / "text_probs_holdout.npy", p_va)
    np.save(OUTPUT_DIR / "text_probs_test.npy", p_te)
    print(f"저장: {OUTPUT_DIR / 'text_probs_holdout.npy'}, {OUTPUT_DIR / 'text_probs_test.npy'}")


if __name__ == "__main__":
    main()
