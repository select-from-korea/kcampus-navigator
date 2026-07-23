"""
run_eval.py — 실제 코퍼스(46문서) + 평가셋으로 검색 재보정 (SSOT §13, 발표소스 §7)

기존 `src/retriever.py`·`src/vector_store.py` 의 __main__ 은 더미 코퍼스로
측정합니다. 이 스크립트는 **실제 인덱스(vectors.npz)** 와 `eval/questions.csv`
로 아래를 재측정해 발표 슬라이드 5번 수치를 확정합니다.

  1. 다국어 브릿지  — 번역 전/후 평균 Top-1 Dense 유사도
  2. 검색 전략      — Dense / BM25 / Hybrid(score) / Hybrid(rrf) Top-1 정확도
  3. Abstention    — 임계값 스윕 → 총 정확도 최대점(권장 임계값)
  4. 가중치        — W_BM25 = 1.0 / 1.5 / 2.0 비교

정답 근거(ground truth)는 questions.csv 의 `expected_doc`(hikorea 번호, `|` 로 복수 허용).
답변가능 = expected_outcome==answer & route∈(rag,hybrid). 답변불가 = 아래 out-of-domain 프로브 + refused(rag).

실행
  python eval/run_eval.py
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

BASE = Path(__file__).resolve().parent.parent
load_dotenv(BASE / ".env")
import sys
sys.path.insert(0, str(BASE / "src"))
from vector_store import VectorStore, embed, translate_to_korean   # noqa: E402
from retriever import tokenize_ko, _minmax                          # noqa: E402

# 코퍼스에 근거가 없는 out-of-domain 프로브 (Abstention 이 거부해야 정답)
_OOD_PROBES = [
    "What is the weather like in Seoul in winter?",
    "How do I open a bank account in Korea?",
    "Can I bring my pet dog when I move to Korea?",
    "What is the best Korean food to try in Busan?",
]


def _num(meta) -> str:
    m = re.search(r"hikorea_(\d+)_", meta.get("doc_id", ""))
    return m.group(1) if m else ""


def load_rows():
    with open(BASE / "eval" / "questions.csv", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    store = VectorStore.load()
    bm25 = BM25Okapi([tokenize_ko(t) for t in store.texts])
    metas = store.metas
    print(f"인덱스: {store} | BM25 코퍼스 {len(store.texts)}\n")

    rows = load_rows()
    answerable = [r for r in rows
                  if r["expected_outcome"] == "answer" and r["expected_route"] in ("rag", "hybrid")]
    refused_rag = [r for r in rows
                   if r["expected_outcome"] == "refused" and r["expected_route"] == "rag"]

    # ---- 각 답변가능 질문: 전략별 Top-1 doc + confidence ----
    bridge_raw, bridge_ko = [], []
    strat_hit = {"dense": 0, "bm25": 0, "hybrid_score": 0, "hybrid_rrf": 0}
    recall3 = 0                      # 융합 top-3 안에 정답 문서가 있는가 (LLM 은 top-3 사용)
    w_hit = {1.0: 0, 1.5: 0, 2.0: 0}
    ans_conf = []
    fixed, broke = [], []

    def top_by_score(scores):
        return int(np.argmax(scores))

    def top3(scores):
        return [int(i) for i in np.argsort(-scores)[:3]]

    for r in answerable:
        q = r["question"]
        accept = set(r["expected_doc"].split("|")) if r["expected_doc"] else set()
        ko = translate_to_korean(q)

        qv_ko = embed([ko])[0]
        qv_raw = embed([q])[0]
        dense_ko = store.vectors @ qv_ko
        dense_raw = store.vectors @ qv_raw
        bm = np.asarray(bm25.get_scores(tokenize_ko(ko)), dtype=np.float32)

        bridge_ko.append(float(dense_ko.max()))
        bridge_raw.append(float(dense_raw.max()))
        ans_conf.append(float(dense_ko.max()))

        dense_top = _num(metas[top_by_score(dense_ko)])
        bm_top = _num(metas[top_by_score(bm)])
        dn, bn = _minmax(dense_ko), _minmax(bm)
        score_top = _num(metas[top_by_score(dn + bn)])
        # rrf (k=10)
        dr = {int(i): rk for rk, i in enumerate(np.argsort(-dense_ko))}
        br = {int(i): rk for rk, i in enumerate(np.argsort(-bm))}
        rrf = np.array([1/(10+dr[i]+1) + (1/(10+br[i]+1) if bm[i] > 0 else 0)
                        for i in range(len(metas))])
        rrf_top = _num(metas[top_by_score(rrf)])

        d_ok = dense_top in accept
        s_ok = score_top in accept
        strat_hit["dense"] += d_ok
        strat_hit["bm25"] += bm_top in accept
        strat_hit["hybrid_score"] += s_ok
        strat_hit["hybrid_rrf"] += rrf_top in accept
        recall3 += any(_num(metas[i]) in accept for i in top3(dn + bn))
        if s_ok and not d_ok:
            fixed.append(q)
        elif d_ok and not s_ok:
            broke.append(q)

        for w in w_hit:
            w_top = _num(metas[top_by_score(dn + w*bn)])
            w_hit[w] += w_top in accept

    # ---- 답변불가 confidence (OOD 프로브 + refused rag) ----
    unans_conf = []
    for q in _OOD_PROBES + [r["question"] for r in refused_rag]:
        ko = translate_to_korean(q)
        unans_conf.append(float((store.vectors @ embed([ko])[0]).max()))

    n_ans = len(answerable)
    # ---- 리포트 ----
    print("=" * 78)
    print("1. 다국어 브릿지 (답변가능 %d문항, 평균 Top-1 Dense 유사도)" % n_ans)
    print("   영어 그대로 : %.3f" % np.mean(bridge_raw))
    print("   한국어 번역 : %.3f   (%+.1f%%)"
          % (np.mean(bridge_ko), (np.mean(bridge_ko)/np.mean(bridge_raw)-1)*100))
    print()
    print("2. 검색 전략 Top-1 정확도 (n=%d)" % n_ans)
    for k in ("dense", "bm25", "hybrid_score", "hybrid_rrf"):
        print("   %-14s Top-1 %d/%d" % (k, strat_hit[k], n_ans))
    print("   hybrid_score   Recall@3 %d/%d  ← 실제 답변 생성이 보는 범위" % (recall3, n_ans))
    if fixed:
        print("   하이브리드(score)가 교정: %d건" % len(fixed))
    if broke:
        print("   하이브리드(score)가 악화: %d건 %s" % (len(broke), broke))
    print()
    print("3. BM25 가중치 스윕 (Hybrid score Top-1 정확도)")
    for w in (1.0, 1.5, 2.0):
        print("   W_BM25=%.1f : %d/%d" % (w, w_hit[w], n_ans))
    print()
    print("4. Abstention 임계값 스윕  (답변가능 %d · 답변불가 %d)"
          % (len(ans_conf), len(unans_conf)))
    print("   답변가능 conf: min %.3f / mean %.3f / max %.3f"
          % (min(ans_conf), np.mean(ans_conf), max(ans_conf)))
    print("   답변불가 conf: min %.3f / mean %.3f / max %.3f"
          % (min(unans_conf), np.mean(unans_conf), max(unans_conf)))
    cands = sorted(set(round(c, 3) for c in ans_conf + unans_conf))
    best_t, best_acc = None, -1
    total = len(ans_conf) + len(unans_conf)
    print("   thr    answer_ok  refuse_ok  total_acc")
    for t in cands:
        a_ok = sum(c >= t for c in ans_conf)
        r_ok = sum(c < t for c in unans_conf)
        acc = (a_ok + r_ok) / total
        # 동점이면 더 높은(보수적=더 잘 거부하는) 임계값 선택 — 오답이 위험한 도메인
        if acc > best_acc or (acc == best_acc and t > best_t):
            best_acc, best_t = acc, t
        print("   %.3f   %2d/%-2d      %2d/%-2d      %.0f%%"
              % (t, a_ok, len(ans_conf), r_ok, len(unans_conf), acc*100))
    print("   → 권장 임계값 %.3f (총 정확도 %.0f%%)" % (best_t, best_acc*100))
    print("=" * 78)


if __name__ == "__main__":
    main()
