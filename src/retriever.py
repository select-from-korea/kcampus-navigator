"""
retriever.py — Hybrid Retrieval (BM25 + Dense) + Abstention

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
왜 하이브리드인가 — 7/22 실측 근거
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Dense 단독 검색에서 아래 실패가 관측됐습니다.

    Q: "How do I extend my student visa?"
    → Top-1: 출입국관리법 시행령 제23조 (0.535)   ❌
       정답: 체류기간 연장 안내

  원인: 시행령·건강보험 문서에 "체류기간 연장"이 부수적으로 언급되어
        정답 문서를 밀어냄 (distractor 문제).
  결정적 사실: 오답 점수 0.535 가 정답군(0.498~0.648) 한가운데 있어
        어떤 임계값으로도 걸러낼 수 없음. → 검색 자체를 고쳐야 함.

  BM25 는 "연장허가", "만료일" 같은 정확한 어휘 매칭에 강합니다.
  실제로 BM25 는 이 질문에서 정답 문서를 1위로 찾아냈습니다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
핵심 설계 1: 두 신호의 역할 분리
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "답할 수 있는가?"   → Dense 코사인 최대값 (정규화하지 않은 절대 스케일)
    "어느 문서를 인용?" → 융합 점수 (Dense + BM25)

  Abstention 은 코퍼스 전체에 의미적으로 가까운 문서가 하나라도 있는지를
  묻는 질의 수준(query-level) 판단이고,
  Ranking 은 그중 어느 것을 인용할지를 묻는 문서 수준 판단입니다.
  질문이 다르므로 신호도 달라야 합니다.

  이 분리 덕분에, 순위용 점수는 자유롭게 정규화해도 됩니다.
  거부 판단은 정규화되지 않은 dense 최대값을 따로 쓰기 때문입니다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
핵심 설계 2: 왜 RRF 를 버리고 점수 융합을 쓰는가  ★ v3
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1차 구현에서 표준 RRF (k=60) 를 썼습니다. 실패했습니다.
  2차로 가중 RRF (k=10, w_bm25=1.5) 를 썼습니다. 더 나빠졌습니다.

    Q3 결과
      Dense Top-1 : 출입국관리법 시행령      (오답)
      BM25  Top-1 : 체류기간 연장 안내       (정답)
      RRF   Top-1 : 유학생 건강보험 안내     (제3의 문서!)

  원인 — RRF 의 "합의 편향(consensus bias)":
    건강보험 문서가 Dense 에서도 2위, BM25 에서도 2위였습니다.
    1위 두 개가 서로 다른 문서라 표를 나눠 가지는 동안,
    양쪽에서 꾸준히 2위인 문서가 합계에서 이겼습니다.

  근본 원인 — RRF 는 점수의 "크기"를 버립니다:
    BM25 에서 정답이 1.133 점으로 1위인데, 2위와 얼마나 벌어졌는지는
    반영되지 않습니다. 1위/2위/3위 라는 서열만 남습니다.
    코퍼스가 작을수록 이 정보 손실이 치명적입니다.

  RRF 는 원래 대규모 코퍼스에서 스케일이 전혀 다른 검색 시스템을
  합칠 때의 도구입니다. 문서 수십 개 규모에는 맞지 않습니다.

  → 점수 기반 융합으로 전환합니다.

      dense_norm = (dense - min) / (max - min)    # 쿼리 내 0~1
      bm25_norm  = (bm25  - min) / (max - min)
      fused = w_dense * dense_norm + w_bm25 * bm25_norm

    1위와 2위의 격차가 보존되므로 합의 편향이 사라집니다.

  ⚠️ 두 방식을 FUSION 환경변수로 전환할 수 있게 남겨뒀습니다.
     문서 5건으로 융합 전략을 고르는 것은 통계적 근거가 약합니다.
     7/24 실제 코퍼스 + 평가셋 30문항으로 재측정하세요.
     그 비교표가 발표 슬라이드 5번의 재료가 됩니다.

사용법
  from src.retriever import Retriever
  r = Retriever.from_store(vs)        # VectorStore 인스턴스
  res = r.retrieve("Can I work part-time on a D-2 visa?")

  res.refused        # True 면 답변 거부
  res.confidence     # Dense 최대 코사인
  res.hits           # 융합 순위 상위 k개
  res.disagreement   # Dense/BM25 Top-1 불일치 여부 (불확실성 신호)

단독 실행
  python src/retriever.py            # 점수 융합 (기본)
  FUSION=rrf python src/retriever.py # RRF 비교 (Windows: set FUSION=rrf)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

# 같은 폴더 실행(python src/retriever.py)과 패키지 임포트 모두 지원
try:
    from .vector_store import VectorStore, Hit, embed, translate_to_korean
except ImportError:
    from vector_store import VectorStore, Hit, embed, translate_to_korean

BASE = Path(__file__).resolve().parent.parent
load_dotenv(BASE / ".env")

THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.45"))

# ---- 융합 설정 ----
#   FUSION="score" : 쿼리 내 min-max 정규화 후 가중합 (기본값, 위 설명 참조)
#   FUSION="rrf"   : 순위 기반 RRF (비교 실험용)
FUSION = os.getenv("FUSION", "score")
RRF_K = int(os.getenv("RRF_K", "10"))
W_DENSE = float(os.getenv("W_DENSE", "1.0"))
W_BM25 = float(os.getenv("W_BM25", "1.0"))


# =================================================================
#  한국어 토크나이저 (BM25 용)
# =================================================================
#  공백 분리로는 한국어 BM25 가 거의 작동하지 않습니다.
#  "체류기간 연장허가는" 과 "체류기간 연장" 이 다른 토큰이 되기 때문입니다.
#  형태소 분석으로 어미·조사를 떼어내야 어휘 매칭이 성립합니다.

_CONTENT_TAGS = (
    "NNG", "NNP", "NNB", "NR", "NP",     # 명사류
    "VV", "VA", "XR",                    # 용언·어근
    "SL", "SH", "SN",                    # 외국어·한자·숫자 (D-2, TOPIK)
)

_kiwi = None


def _get_kiwi():
    global _kiwi
    if _kiwi is None:
        from kiwipiepy import Kiwi
        _kiwi = Kiwi()
    return _kiwi


def tokenize_ko(text: str) -> list[str]:
    """한국어 텍스트를 내용어 형태소 리스트로 변환합니다."""
    if not text or not text.strip():
        return []
    try:
        toks = _get_kiwi().tokenize(text)
        out = [t.form for t in toks if t.tag in _CONTENT_TAGS and len(t.form) > 1]
        # 숫자·영문 코드는 1글자여도 유지 (D-2 등)
        out += [t.form for t in toks
                if t.tag in ("SL", "SN", "SH") and len(t.form) == 1]
        return out
    except Exception:
        # 형태소 분석 실패 시 공백 분리로 폴백 (검색이 멈추면 안 됨)
        return re.findall(r"[가-힣A-Za-z0-9\-]+", text.lower())


def _minmax(x: np.ndarray) -> np.ndarray:
    """쿼리 내 min-max 정규화. 전부 같은 값이면 0 벡터를 반환합니다."""
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


# =================================================================
#  결과 구조체
# =================================================================

@dataclass
class RetrievalResult:
    query: str                       # 원문 질의 (영어)
    ko_query: str                    # 검색에 쓰인 한국어 명사구
    hits: list[Hit]                  # 융합 순위 상위 k개
    confidence: float                # Dense 최대 코사인 (Abstention 기준)
    refused: bool
    refused_reason: str = ""
    disagreement: bool = False       # Dense/BM25 Top-1 불일치
    debug: dict[str, Any] = field(default_factory=dict)


# =================================================================
#  하이브리드 검색기
# =================================================================

class Retriever:
    def __init__(self, store: VectorStore, threshold: float = THRESHOLD,
                 fusion: str = FUSION, rrf_k: int = RRF_K,
                 w_dense: float = W_DENSE, w_bm25: float = W_BM25):
        self.store = store
        self.threshold = threshold
        self.fusion = fusion
        self.rrf_k = rrf_k
        self.w_dense = w_dense
        self.w_bm25 = w_bm25
        self._bm25: BM25Okapi | None = None
        self._build_bm25()

    @classmethod
    def from_store(cls, store: VectorStore, **kw) -> "Retriever":
        return cls(store, **kw)

    def _build_bm25(self) -> None:
        if not self.store.texts:
            self._bm25 = None
            return
        corpus = [tokenize_ko(t) for t in self.store.texts]
        if not any(corpus):          # 전부 빈 토큰이면 BM25 구성 불가
            self._bm25 = None
            return
        self._bm25 = BM25Okapi(corpus)

    # ---------- 검색 ----------

    def retrieve(self, query: str, k: int = 5,
                 translate: bool = True) -> RetrievalResult:
        n = len(self.store.texts)
        if n == 0:
            return RetrievalResult(query, query, [], 0.0, True,
                                   "인덱스가 비어 있습니다")

        # --- 1) 질의를 한국어 명사구로 변환 (API 1회) ---
        ko_query = translate_to_korean(query) if translate else query

        # --- 2) Dense: 코퍼스 전체에 대한 코사인 점수 ---
        qv = embed([ko_query], is_query=True)[0]
        dense_scores = self.store.vectors @ qv          # (n,)
        dense_order = np.argsort(-dense_scores)
        dense_rank = {int(idx): r for r, idx in enumerate(dense_order)}

        # --- 3) BM25: 어휘 매칭 점수 ---
        if self._bm25 is not None:
            bm_scores = np.asarray(
                self._bm25.get_scores(tokenize_ko(ko_query)), dtype=np.float32
            )
        else:
            bm_scores = np.zeros(n, dtype=np.float32)
        bm_order = np.argsort(-bm_scores)
        bm_rank = {int(idx): r for r, idx in enumerate(bm_order)}

        # --- 4) 융합 ---
        if self.fusion == "rrf":
            # 순위 기반. 점수 크기를 버리므로 합의 편향이 생깁니다 (docstring 참조)
            fused = np.zeros(n, dtype=np.float32)
            for i in range(n):
                fused[i] = self.w_dense / (self.rrf_k + dense_rank[i] + 1)
                if bm_scores[i] > 0:
                    fused[i] += self.w_bm25 / (self.rrf_k + bm_rank[i] + 1)
        else:
            # 점수 기반: 쿼리 내 정규화 후 가중합. 1위-2위 격차가 보존됩니다.
            fused = (self.w_dense * _minmax(dense_scores)
                     + self.w_bm25 * _minmax(bm_scores))

        fused_order = np.argsort(-fused)[:min(k, n)]

        hits = [
            Hit(float(dense_scores[i]), self.store.texts[i], self.store.metas[i])
            for i in fused_order
        ]

        # --- 5) Abstention: 질의 수준 판단 (정규화하지 않은 Dense 최대값) ---
        confidence = float(dense_scores.max())
        refused = confidence < self.threshold
        reason = ""
        if refused:
            reason = (
                f"No supporting document was found above our confidence "
                f"threshold ({self.threshold:.2f}; best match scored "
                f"{confidence:.3f}). We do not generate answers about "
                f"immigration or academic rules without a verifiable source."
            )

        # --- 6) 불일치 신호 (발표·디버깅용) ---
        dense_top = int(dense_order[0])
        bm_top = int(bm_order[0]) if bm_scores.max() > 0 else dense_top
        disagreement = dense_top != bm_top

        return RetrievalResult(
            query=query,
            ko_query=ko_query,
            hits=hits,
            confidence=confidence,
            refused=refused,
            refused_reason=reason,
            disagreement=disagreement,
            debug={
                "fusion": self.fusion,
                "dense_top": self.store.metas[dense_top].get("title", ""),
                "dense_top_score": float(dense_scores[dense_top]),
                "bm25_top": self.store.metas[bm_top].get("title", ""),
                "bm25_top_score": float(bm_scores[bm_top]),
                "fused_top": self.store.metas[int(fused_order[0])].get("title", ""),
                "fused_top_score": float(fused[int(fused_order[0])]),
            },
        )

    # ---------- Dense 단독 (발표 비교용) ----------

    def retrieve_dense_only(self, query: str, k: int = 5,
                            translate: bool = True) -> RetrievalResult:
        """발표 슬라이드 3번(순수 RAG 실패 시연)에서 사용합니다."""
        ko_query = translate_to_korean(query) if translate else query
        hits = self.store.search(ko_query, k=k, translate=False)
        conf = hits[0].score if hits else 0.0
        return RetrievalResult(
            query=query, ko_query=ko_query, hits=hits,
            confidence=conf, refused=conf < self.threshold,
        )


# =================================================================
#  단독 실행 — Dense vs Hybrid 비교
# =================================================================

if __name__ == "__main__":
    try:
        from .vector_store import _DUMMY_DOCS, _TEST_QUERIES
    except ImportError:
        from vector_store import _DUMMY_DOCS, _TEST_QUERIES

    print(f"\n임계값: {THRESHOLD}  |  융합: {FUSION}"
          f"{f' (k={RRF_K})' if FUSION == 'rrf' else ''}  |  "
          f"가중치 dense={W_DENSE} bm25={W_BM25}\n")

    vs = VectorStore()
    vs.add([t for t, _ in _DUMMY_DOCS], [m for _, m in _DUMMY_DOCS])
    r = Retriever.from_store(vs)
    print(f"인덱스: {vs}  |  BM25: {'구성됨' if r._bm25 else '없음'}\n")

    # 토크나이저 동작 확인 — 여기가 깨지면 BM25 전체가 무의미합니다
    sample = "체류기간 연장허가는 만료일 4개월 전부터 신청할 수 있다"
    print("토크나이저 확인")
    print(f"  입력: {sample}")
    print(f"  토큰: {tokenize_ko(sample)}\n")

    dense_hit = bm25_hit = hybrid_hit = answerable = 0
    fixed, broke = [], []

    print("=" * 98)
    for q, expect in _TEST_QUERIES:
        res = r.retrieve(q, k=3)
        d = res.debug

        print(f"\nQ  {q}")
        print(f"   번역   → {res.ko_query}")
        print(f"   Dense  : {d['dense_top_score']:.3f}  {d['dense_top']}")
        print(f"   BM25   : {d['bm25_top_score']:.3f}  {d['bm25_top']}")
        print(f"   Fused  : {d['fused_top_score']:.3f}  {d['fused_top']}"
              f"{'   ⚠️ 두 신호 불일치' if res.disagreement else ''}")

        if expect is None:
            ok = res.refused
            print(f"   판정   : {'REFUSE (정답)' if ok else '⚠️ 답변함 (오답)'}"
                  f"   conf={res.confidence:.3f}")
        else:
            answerable += 1
            dense_ok = (d["dense_top"] == expect)
            bm25_ok = (d["bm25_top"] == expect)
            hybrid_ok = (d["fused_top"] == expect)
            dense_hit += dense_ok
            bm25_hit += bm25_ok
            hybrid_hit += hybrid_ok
            changed = ""
            if hybrid_ok and not dense_ok:
                changed = "   ✅ 하이브리드가 교정함"
                fixed.append(q)
            elif dense_ok and not hybrid_ok:
                changed = "   ❌ 하이브리드가 악화시킴"
                broke.append(q)
            print(f"   판정   : [{'OK' if not res.refused else 'REFUSE'}] "
                  f"Dense {'O' if dense_ok else 'X'} / "
                  f"BM25 {'O' if bm25_ok else 'X'} → "
                  f"Hybrid {'O' if hybrid_ok else 'X'}{changed}")

    print("\n" + "=" * 98)
    print("  ※ 발표 슬라이드 재료 — 기록해두세요")
    print("=" * 98)
    print(f"{'Dense 단독            Top-1 정확도':38s} {dense_hit}/{answerable}")
    print(f"{'BM25 단독             Top-1 정확도':38s} {bm25_hit}/{answerable}")
    print(f"{f'Hybrid ({FUSION})            Top-1 정확도':38s} "
          f"{hybrid_hit}/{answerable}")
    if fixed:
        print(f"\n  하이브리드가 교정한 질문 ({len(fixed)}건):")
        for q in fixed:
            print(f"    ✅ {q}")
    if broke:
        print(f"\n  하이브리드가 악화시킨 질문 ({len(broke)}건) — 가중치 재검토:")
        for q in broke:
            print(f"    ❌ {q}")
    print("=" * 98 + "\n")