"""
local.py — K-Campus 선배 라운지 (Sunbae Lounge): 큐레이션 로컬 팁 레이어

정부 문서로는 절대 안 나오지만 유학생이 진짜 궁금해하는 캠퍼스 생활·문화·
행정 꿀팁을, '선배(seonbae)' 페르소나로 답합니다. LLM 생성이 아니라 손수
큐레이션한 docs/local_tips.json 에서 매칭하므로 **할루시네이션이 0** 입니다.

2단계 매칭 (정밀도와 리콜을 동시에)
  1) 키워드(부분 문자열) — 라우터보다 **먼저**, 무API·결정적.
     트리거 문구가 그대로 들어있으면 즉시 그 팁을 반환합니다. 회색지대
     (위장결혼·초과근무·오버스테이) 안전 리다이렉트를 확실히 잡는 빠른 길.
  2) 의미(임베딩) — RAG 가 근거 부족으로 **거부하려는 순간에만** 구제.
     "should I drink soju in Korean society?" 처럼 트리거에 없던 표현도
     의미가 가까운 큐레이션 팁으로 연결합니다. 답변은 여전히 손으로 쓴
     팁이라 지어내지 않습니다.

왜 의미 매칭을 라우터 前에 전면 적용하지 않는가 (안전)
  진짜 규정 질문("Can I work part-time on D-2, how many hours?")이 회색지대
  팁("work more hours than allowed...")과 의미가 겹쳐 **정답(공식 문서)을
  가로챌** 수 있습니다. 그래서 의미 매칭은 RAG 가 문서를 못 찾아 거부할
  때만 개입합니다. 규정 질문은 그전에 문서로 정상 답변되어 안전합니다.

출처 표기
  로컬 팁의 source 는 '🎓 K-Campus 선배 라운지 (campus-life tip, not an
  official regulation)' 로 명시해 규정 인용과 명확히 구분합니다.

단독 실행
  python src/local.py             # 키워드 매칭 스모크 (무API)
  python src/local.py --semantic  # 의미 매칭 유사도까지 (API 필요)
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

BASE = Path(__file__).resolve().parent.parent
TIPS_PATH = BASE / "docs" / "local_tips.json"
EMB_CACHE = BASE / "data" / "processed" / "local_tips.emb.npz"

# contract·vector_store 임포트를 위해 루트와 src 를 경로에 올립니다.
for _p in (str(BASE), str(BASE / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from contract import Answer, EMPTY_CHART, Source

# 의미 매칭 임계값 — 이 코사인 유사도 이상이어야 선배 팁으로 구제합니다.
# 너무 낮으면 무관한 질문까지 팁이 붙고, 너무 높으면 표현이 조금만 달라도
# 놓칩니다. .env 의 LOCAL_SEM_THRESHOLD 로 조정 가능.
SEM_THRESHOLD = float(os.getenv("LOCAL_SEM_THRESHOLD", "0.50"))

# semantic 구제는 '명백히 안전한 생활' 카테고리로만 제한합니다. 비자·이민에
# 인접한 카테고리(grayzone·bureaucracy·money·safety)는 어휘가 겹쳐, 퍼지
# 매칭을 허용하면 "귀화 방법?" 같은 심각한 이민 질문을 엉뚱한 생활 팁으로
# 낚아채 거부 신뢰를 훼손합니다. 그 카테고리는 정확한 키워드로만 발동합니다.
SEMANTIC_CATEGORIES = {"campus-life", "culture", "food"}

# 로컬 팁 소스 라벨 — 규정 문서와 명확히 구분됩니다.
SOURCE_TITLE = "🎓 K-Campus 선배 라운지 · campus-life tip (not an official regulation)"


@dataclass
class LocalHit:
    tip: dict
    score: float          # 키워드: 트리거 단어 수 합 / 의미: 코사인 유사도
    via: str = "keyword"  # "keyword" | "semantic"

    @property
    def is_grayzone(self) -> bool:
        return self.tip.get("kind") == "grayzone"


class LocalTips:
    """docs/local_tips.json 을 1회 로드해 질문을 큐레이션 팁에 매칭합니다."""

    def __init__(self, tips: list[dict]):
        self.tips = tips
        self._emb: Optional[np.ndarray] = None   # 지연 계산되는 팁 임베딩 행렬

    # ---- 로딩 ------------------------------------------------------
    @classmethod
    def load(cls, path: Path = TIPS_PATH) -> "LocalTips":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(data.get("tips", []))

    # ---- 1) 키워드(부분 문자열) 매칭 -------------------------------
    @staticmethod
    def _norm(text: str) -> str:
        # 소문자 + 곡선 따옴표(’)를 곧은 따옴표(')로 통일해 "don't/don’t" 를
        # 모두 잡습니다. 앞뒤 공백은 단어 경계 오매칭을 줄여줍니다.
        return " " + text.lower().replace("’", "'").strip() + " "

    def _kw_score(self, q_norm: str, tip: dict) -> float:
        """등장한 트리거 문구의 '단어 수 합'. 긴 문구일수록 강하게 봅니다."""
        score = 0.0
        for trig in tip.get("triggers", []):
            t = trig.lower().replace("’", "'").strip()
            if t and t in q_norm:
                score += max(1, len(t.split()))
        return score

    def match(self, question: str) -> Optional[LocalHit]:
        """키워드 매칭. 라우터보다 먼저 쓰는 빠른 길. 없으면 None."""
        if not question or not question.strip():
            return None
        q = self._norm(question)
        best: Optional[LocalHit] = None
        for tip in self.tips:
            s = self._kw_score(q, tip)
            if s <= 0:
                continue
            if best is None or s > best.score:
                best = LocalHit(tip=tip, score=s, via="keyword")
        return best

    # ---- 2) 의미(임베딩) 매칭 --------------------------------------
    @staticmethod
    def _emb_text(tip: dict) -> str:
        # 대표 질문 + 트리거 어휘 + 답변 본문을 합쳐 임베딩합니다. 답변 본문이
        # 주제 어휘(소주·술·신발·보증금 등)를 풍부하게 담아 표현이 달라도
        # 의미로 잡히게 해줍니다(리콜↑).
        trg = " ".join(tip.get("triggers", [])[:8])
        return f"{tip.get('ask', '')} {trg} {tip.get('answer', '')}".strip()

    def _ensure_embeddings(self) -> Optional[np.ndarray]:
        """팁 임베딩 행렬을 1회 계산(또는 디스크 캐시 로드)합니다.

        캐시는 tip id 목록이 일치할 때만 재사용합니다. json 을 고치면
        자동으로 다시 계산합니다. 임베딩/캐시 실패는 조용히 무시하고
        의미 매칭을 비활성화합니다(→ 정상 거부로 폴백).
        """
        if self._emb is not None:
            return self._emb
        import hashlib
        texts = [self._emb_text(t) for t in self.tips]
        # 팁 내용(질문/트리거/답변)이 바뀌면 sig 가 달라져 캐시가 자동 무효화됩니다.
        sig = hashlib.sha1("\x00".join(texts).encode("utf-8")).hexdigest()

        if EMB_CACHE.exists():
            try:
                d = np.load(EMB_CACHE, allow_pickle=True)
                if str(d["sig"]) == sig:
                    self._emb = d["emb"].astype(np.float32)
                    return self._emb
            except Exception:
                pass

        try:
            from vector_store import embed          # 프로젝트 표준 임베딩 재사용
            mat = embed(texts)                       # L2 정규화됨
        except Exception:
            return None
        self._emb = mat
        try:
            EMB_CACHE.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(EMB_CACHE, sig=np.array(sig), emb=mat)
        except Exception:
            pass
        return self._emb

    def semantic_match(self, question: str,
                       threshold: float = SEM_THRESHOLD) -> Optional[LocalHit]:
        """임베딩 코사인으로 가장 가까운 팁. 임계값 미만이거나 API 실패면 None.

        ⚠️ SEMANTIC_CATEGORIES(생활) 팁만 대상입니다. 비자·이민 인접 카테고리
        (grayzone·bureaucracy·money·safety)는 어휘가 겹쳐, 퍼지 매칭을 허용하면
        '귀화 방법?' 같은 진짜 심각한 이민 질문을 엉뚱한 생활 팁으로 낚아채
        거부 신뢰를 훼손합니다. 그 카테고리는 정확한 키워드(match)로만 발동.
        """
        if not question or not question.strip():
            return None
        try:
            emb = self._ensure_embeddings()
            if emb is None or len(emb) == 0:
                return None
            from vector_store import embed
            qv = embed([question], is_query=True)[0]     # L2 정규화됨
            sims = emb @ qv                              # 정규화 → 내적 = 코사인
            best_i, best = -1, -1.0
            for i, tip in enumerate(self.tips):
                if tip.get("category") not in SEMANTIC_CATEGORIES:
                    continue                             # 안전한 생활 카테고리만
                s = float(sims[i])
                if s > best:
                    best_i, best = i, s
            if best_i >= 0 and best >= threshold:
                return LocalHit(tip=self.tips[best_i], score=best, via="semantic")
        except Exception:
            return None
        return None

    # ---- 계약 변환 --------------------------------------------------
    @staticmethod
    def to_answer(hit: LocalHit) -> Answer:
        """LocalHit → contract.Answer (route="local").

        app.py 는 route=="refused" 만 특별 처리하고 나머지는 answer_text +
        sources 로 렌더하므로 "local" 은 기존 프론트에서도 안전하게 표시됩니다.
        """
        tip = hit.tip
        source: Source = {
            "title": SOURCE_TITLE,
            "snippet": tip.get("ask", ""),
            "url": None,
            "score": round(float(hit.score), 3) if hit.via == "semantic" else 1.0,
        }
        # 키워드 정확 매칭은 신뢰도 1.0, 의미 매칭은 실제 유사도를 노출해 정직하게.
        confidence = 1.0 if hit.via == "keyword" else round(float(hit.score), 3)
        return {
            "route": "local",
            "answer_text": tip.get("answer", ""),
            "table_markdown": "",
            "chart": EMPTY_CHART,
            "sources": [source],
            "confidence": confidence,
            "refused_reason": "",
        }


# =================================================================
#  모듈 싱글턴 — 프로세스당 한 번만 로드
# =================================================================

_tips: Optional[LocalTips] = None


def _ensure_loaded() -> LocalTips:
    global _tips
    if _tips is None:
        _tips = LocalTips.load()
    return _tips


def match_local(question: str) -> Optional[Answer]:
    """[라우터 前] 키워드 매칭. 맞으면 Answer, 아니면 None."""
    hit = _ensure_loaded().match(question)
    return LocalTips.to_answer(hit) if hit else None


def rescue_local(question: str,
                 threshold: float = SEM_THRESHOLD) -> Optional[Answer]:
    """[거부 직전 구제] 키워드 → 의미 순으로 시도. 맞으면 Answer, 아니면 None.

    RAG 가 근거를 못 찾아 refused 로 갈 때 파이프라인이 호출합니다. 여기서도
    안 걸리면 원래대로 정직하게 거부합니다.
    """
    lt = _ensure_loaded()
    hit = lt.match(question) or lt.semantic_match(question, threshold)
    return LocalTips.to_answer(hit) if hit else None


# =================================================================
#  단독 실행 — 스모크 테스트
# =================================================================

_KW_SMOKE = [
    ("Should I marry a Korean citizen to get a visa?", True),
    ("Can I just work more hours if immigration doesn't find out?", True),
    ("Do I really need KakaoTalk in Korea?", True),
    ("Do I have to drink at MT?", True),
    ("Do I tip in Korea?", True),
    ("What is jeonse and wolse?", True),
    ("Can I survive without speaking Korean?", True),
    # --- 아래는 절대 로컬로 새면 안 되는 진짜 규정/통계 질문 ---
    ("Can I work part-time on a D-2 visa? How many hours?", False),
    ("Which universities in Seoul have the most international students?", False),
    ("What TOPIK level do I need for a bachelor's program?", False),
    ("Do I need health insurance as an international student?", False),
    ("Will I definitely get a scholarship if I apply now?", False),
]

# 키워드 트리거엔 없지만 '의미'로는 잡혀야 하는 표현들 (True=매칭 기대).
# (soju·refuse alcohol 등 흔한 표현은 이미 키워드로 잡히므로 여기선 제외)
_SEM_SMOKE = [
    ("How do I get around the city cheaply?", True),          # -> tmoney-transit
    ("Why do people take their shoes off at home?", True),    # -> shoes-off
    ("Where do broke students eat?", True),                   # -> student-cafeteria
    ("Do people expect me to drink at work gatherings?", True),  # -> mt-hwesik-drinking
    # --- 아래는 진짜 거부여야 함(생활 팁 아님) → 임계값 미만 기대 ---
    ("Will I definitely get a scholarship if I apply now?", False),
    ("What is the GDP growth rate of Korea?", False),
    ("Is it hard to make friends as an exchange student?", False),  # 해당 팁 없음
]


def _run_keyword(lt: LocalTips):
    print("\n[1] 키워드 매칭 (무API)")
    print("=" * 92)
    ok = 0
    for q, expect in _KW_SMOKE:
        hit = lt.match(q)
        got = hit is not None
        good = got == expect
        ok += good
        tag = (f"local:{hit.tip['id']} (score={hit.score:g}"
               f"{', GRAYZONE' if hit.is_grayzone else ''})") if hit else "-> passes through (router)"
        print(f"{'O' if good else 'X'}  {q[:56]:56s} {tag}")
    print("-" * 92)
    print(f"키워드 정확도: {ok}/{len(_KW_SMOKE)}\n")


def _run_semantic(lt: LocalTips):
    print("\n[2] 의미 매칭 (API 필요, threshold=%.2f)" % SEM_THRESHOLD)
    print("=" * 92)
    ok = 0
    for q, expect in _SEM_SMOKE:
        hit = lt.semantic_match(q)
        got = hit is not None
        good = got == expect
        ok += good
        near = lt.semantic_match(q, threshold=0.0)  # 최근접(디버그용)
        ninfo = f"{near.tip['id']} cos={near.score:.3f}" if near else "no embeddings"
        verdict = f"-> local:{ninfo}" if got else f"-> below threshold (nearest {ninfo})"
        print(f"{'O' if good else 'X'}  {q[:50]:50s} {verdict}")
    print("-" * 92)
    print(f"의미 매칭 정확도: {ok}/{len(_SEM_SMOKE)}\n")


if __name__ == "__main__":
    lt = _ensure_loaded()
    print(f"\nLoaded {len(lt.tips)} curated tips from {TIPS_PATH.name}")
    _run_keyword(lt)
    if "--semantic" in sys.argv:
        _run_semantic(lt)
    else:
        print("(의미 매칭까지 보려면:  python src/local.py --semantic)\n")
