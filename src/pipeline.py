"""
pipeline.py — 전체 조립: 질문 → 라우팅 → 검색/생성 → Answer 계약

당일 프론트는 mock.answer_question 을 이 파일의 answer_question 으로
한 줄만 바꿉니다 (contract.py 시그니처 그대로).

    from src.pipeline import answer_question
    ans = answer_question("Can I work part-time on a D-2 visa?")

흐름 (SSOT §4 아키텍처)
    질문
     → route_question          라우터: sql / rag / hybrid
     → rag  : Retriever(하이브리드+Abstention) → 근거 있으면 LLM 이 영어로 답변 인용,
              근거 없으면 refused
     → sql  : Text-to-SQL (데이터 적재 후 연결). 현재는 데이터 대기 placeholder
     → hybrid: rag 답변 + (SQL 부분은 데이터 대기 안내)
     → Answer(dict) 반환

책임 분리 (SSOT §4-2)
    "답할 수 있는가?" 는 Retriever 가 Dense 최대 코사인으로 판단합니다.
    파이프라인은 그 결정을 신뢰만 하고, 답변 문장 생성은 근거가 있을 때만 합니다.
    → 환각(hallucination) 방지의 핵심. 근거 없으면 문장 자체를 만들지 않습니다.

단독 실행
    python src/pipeline.py        # 대표 질문 스모크 테스트
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

BASE = Path(__file__).resolve().parent.parent
load_dotenv(BASE / ".env")

# contract.py 는 루트, 나머지 모듈은 src/ 에 있습니다. 스크립트 실행
# (python src/pipeline.py)과 패키지 임포트(from src.pipeline import ...)를
# 모두 지원하도록 루트와 src 를 모두 경로에 올리고 절대 임포트를 씁니다.
import sys
for _p in (str(BASE), str(BASE / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from contract import Answer, EMPTY_CHART, Source
from router import route_question
from retriever import Retriever, RetrievalResult
from vector_store import VectorStore

_client = OpenAI()
_LLM = os.getenv("LLM_MODEL", "gpt-4o-mini")

# 답변에 인용으로 노출할 근거 문서 수. 너무 많으면 프론트가 지저분해집니다.
N_SOURCES = 3
SNIPPET_CHARS = 200        # contract.Source.snippet: 200자 이내


# =================================================================
#  지연 로딩 — 인덱스는 프로세스당 한 번만 읽습니다
# =================================================================

_retriever: Retriever | None = None


def _ensure_loaded() -> Retriever:
    """VectorStore + Retriever 를 최초 1회 로드합니다.

    인덱스가 없으면 먼저 `python src/loader.py` 로 만들라고 안내합니다.
    """
    global _retriever
    if _retriever is None:
        try:
            store = VectorStore.load()
        except FileNotFoundError as e:
            raise RuntimeError(
                "벡터 인덱스가 없습니다. 먼저 다음을 실행하세요:\n"
                "    python src/loader.py\n"
                f"(원인: {e})"
            )
        _retriever = Retriever.from_store(store)
    return _retriever


# =================================================================
#  답변 생성 (RAG) — 근거가 있을 때만 문장을 만듭니다
# =================================================================

_ANSWER_SYSTEM = """You help international students who are considering studying in \
Korea. Answer the user's question using ONLY the provided source excerpts, which are \
official Korean government documents.

Rules:
- Answer in {language}.
- Be specific: preserve every number, weekly hour, TOPIK level, grade, visa code, \
fee, and deadline EXACTLY as written in the sources. Do not round or generalize.
- Use ONLY facts contained in the excerpts. Never invent rules that are not there.
- If the excerpts do not actually answer the question, say so plainly instead of guessing.
- Keep it concise (2-5 sentences), then note which document title(s) you relied on.
- These are general rules, not personalized legal advice; when relevant, remind the \
user to confirm with the relevant office."""


def _sources_from(result: RetrievalResult, n: int = N_SOURCES) -> list[Source]:
    srcs: list[Source] = []
    for hit in result.hits[:n]:
        meta = hit.meta
        srcs.append({
            "title": meta.get("title", ""),
            "snippet": hit.text[:SNIPPET_CHARS],
            "url": meta.get("url"),
            "score": round(float(hit.score), 3),
        })
    return srcs


def _context_block(result: RetrievalResult, n: int = N_SOURCES) -> str:
    lines = []
    for i, hit in enumerate(result.hits[:n], 1):
        title = hit.meta.get("title", "")
        lines.append(f"[{i}] {title}\n{hit.text}")
    return "\n\n".join(lines)


def _generate_answer(question: str, result: RetrievalResult,
                     lang: str) -> str:
    """근거 문서를 바탕으로 영어(또는 lang) 답변을 생성합니다.

    LLM 호출이 실패해도 파이프라인이 멈추면 안 되므로, 실패 시 최상위 근거
    문서를 가리키는 안전한 문장으로 폴백합니다.
    """
    language = {"en": "English", "ko": "Korean", "zh": "Chinese"}.get(lang, "English")
    system = _ANSWER_SYSTEM.format(language=language)
    context = _context_block(result)
    user = (
        f"Question: {question}\n\n"
        f"Source excerpts (Korean government documents):\n{context}\n\n"
        f"Answer in {language}, citing the document title(s) you used."
    )
    try:
        resp = _client.chat.completions.create(
            model=_LLM,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            max_tokens=400,
            timeout=20,
        )
        out = (resp.choices[0].message.content or "").strip()
        if out:
            return out
    except Exception:
        pass
    # 폴백: 문장 생성 실패 → 최상위 근거의 제목만 안내 (환각 없이)
    top = result.hits[0].meta.get("title", "the source document") if result.hits else ""
    return (
        "I found a relevant official document but could not generate a full "
        f"summary just now. Please see the cited source: “{top}”."
    )


def _rag_answer(question: str, route: str, lang: str) -> Answer:
    r = _ensure_loaded()
    result = r.retrieve(question)

    if result.refused:
        return {
            "route": "refused",
            "answer_text": "",
            "table_markdown": "",
            "chart": EMPTY_CHART,
            "sources": [],
            "confidence": round(result.confidence, 3),
            "refused_reason": result.refused_reason,
        }

    answer_text = _generate_answer(question, result, lang)
    return {
        "route": route,               # "rag" 또는 "hybrid"
        "answer_text": answer_text,
        "table_markdown": "",
        "chart": EMPTY_CHART,
        "sources": _sources_from(result),
        "confidence": round(result.confidence, 3),
        "refused_reason": "",
    }


# =================================================================
#  SQL 경로 — 공공데이터 적재 후 연결 (현재는 데이터 대기)
# =================================================================

_SQL_PENDING_MSG = (
    "This question needs university statistics (counts, rankings, comparisons) "
    "from the structured dataset, which is not loaded in this build yet. "
    "The statistical data pipeline is being finalized separately."
)


def _sql_answer(question: str, lang: str) -> Answer:
    """Text-to-SQL 경로. data/processed/kcampus.db 를 조회합니다.

    DB 가 아직 없거나(빌드 전) sql_chain 로드에 실패하면 계약을 지키는
    placeholder 를 반환해 파이프라인 전체가 멈추지 않도록 합니다.
    """
    try:
        try:
            from .sql_chain import run_sql_question
        except ImportError:
            from sql_chain import run_sql_question
        return run_sql_question(question, lang)
    except Exception:
        return {
            "route": "sql",
            "answer_text": _SQL_PENDING_MSG,
            "table_markdown": "",
            "chart": EMPTY_CHART,
            "sources": [],
            "confidence": 0.0,
            "refused_reason": "",
        }


# =================================================================
#  공개 함수 — 프론트가 호출하는 유일한 진입점
# =================================================================

def baseline_answer(question: str, lang: str = "en") -> Answer:
    """발표 슬라이드 4 장면 A — "일반 RAG 챗봇" 실패 시연용.

    라우팅도 SQL도 Abstention도 없이, 순수 벡터(dense) 검색만 수행해 최상위
    문단을 그대로 돌려줍니다. 정량형 질문(예: "서울에서 유학생이 가장 많은
    대학은?")을 넣으면, 규정 문서 문단 하나를 반환할 뿐 **집계·순위를 못 합니다.**
    같은 질문을 answer_question() 으로 다시 던지면 SQL 표+차트가 나오는 대비를
    라이브로 보여줄 수 있습니다.
    """
    r = _ensure_loaded()
    res = r.retrieve_dense_only(question, k=3)
    if res.hits:
        answer_text = (
            "⚠️ Baseline: pure vector search — no routing, no SQL. "
            "It retrieves a passage; it cannot count or rank.\n\n"
            f"Top passage returned:\n\n{res.hits[0].text}"
        )
    else:
        answer_text = "⚠️ Baseline: pure vector search returned no passage."
    return {
        "route": "rag",
        "answer_text": answer_text,
        "table_markdown": "",
        "chart": EMPTY_CHART,
        "sources": _sources_from(res),
        "confidence": round(res.confidence, 3),
        "refused_reason": "",
    }


def answer_question(question: str, lang: str = "en") -> Answer:
    if not question or not question.strip():
        return {
            "route": "refused", "answer_text": "", "table_markdown": "",
            "chart": EMPTY_CHART, "sources": [], "confidence": 0.0,
            "refused_reason": "Empty question.",
        }

    decision = route_question(question)

    if decision.route == "sql":
        return _sql_answer(question, lang)

    if decision.route == "hybrid":
        # 규정(RAG) 부분은 지금 완전히 답합니다. 통계(SQL) 부분은 데이터 대기.
        ans = _rag_answer(question, "hybrid", lang)
        if ans["route"] == "hybrid" and ans["answer_text"]:
            ans["answer_text"] += (
                "\n\nNote: the statistical part of this question (university "
                "numbers/rankings) will be answered once the dataset is loaded."
            )
        return ans

    # 기본: rag
    return _rag_answer(question, "rag", lang)


# =================================================================
#  단독 실행 — 대표 질문 스모크 테스트
# =================================================================

_SMOKE = [
    "Can I work part-time on a D-2 visa? How many hours?",
    "What TOPIK level do I need to work more hours?",
    "Why would my visa extension be rejected?",
    "Can I stay in Korea after graduation to find a job?",
    "Do I need health insurance as an international student?",
    "Which university has the most Vietnamese students?",   # sql (데이터 대기)
    "Will I definitely get a scholarship if I apply now?",   # 근거 부족 → refused 기대
]

if __name__ == "__main__":
    print(f"\nLLM: {_LLM}\n" + "=" * 92)
    for q in _SMOKE:
        a = answer_question(q)
        print(f"\nQ  {q}")
        print(f"   route={a['route']}  confidence={a['confidence']}")
        if a["route"] == "refused":
            print(f"   REFUSED: {a['refused_reason'][:100]}...")
        else:
            print(f"   A: {a['answer_text'][:240]}")
            for s in a["sources"][:2]:
                print(f"      └ [{s['score']}] {s['title']}  {s['url'] or ''}")
    print("\n" + "=" * 92 + "\n")
