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
from local import match_local, rescue_local

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


def _profile_block(profile: dict | None) -> str:
    """학생 프로필을 프롬프트용 블록으로. 비어 있으면 빈 문자열."""
    if not profile:
        return ""
    order = [
        ("visa", "Visa / stay status"),
        ("program", "Degree program"),
        ("topik", "TOPIK level"),
        ("nationality", "Nationality"),
        ("grad_date", "Expected graduation"),
        ("region", "Region in Korea"),
    ]
    lines = [f"- {label}: {profile[k]}" for k, label in order if profile.get(k)]
    return "Student profile:\n" + "\n".join(lines) if lines else ""


def _generate_answer(question: str, result: RetrievalResult,
                     lang: str, profile: dict | None = None) -> str:
    """근거 문서를 바탕으로 영어(또는 lang) 답변을 생성합니다.

    profile 이 있으면 근거 안의 '조건별 규정'(예: 학부 25h / 석·박사 30h,
    TOPIK 3급/4급 기준) 중 그 학생에게 해당하는 가지를 골라 맞춤 답변합니다.
    개인화는 근거 문서 안의 값을 '선택'할 뿐, 지어내지 않습니다(환각 방지 유지).

    LLM 호출이 실패해도 파이프라인이 멈추면 안 되므로, 실패 시 최상위 근거
    문서를 가리키는 안전한 문장으로 폴백합니다.
    """
    language = {"en": "English", "ko": "Korean", "zh": "Chinese"}.get(lang, "English")
    system = _ANSWER_SYSTEM.format(language=language)
    context = _context_block(result)
    pblock = _profile_block(profile)
    personalize = ""
    if pblock:
        personalize = (
            f"\n\n{pblock}\n\n"
            "Personalize: from the excerpts, apply the specific branch that matches "
            "this student (e.g. the correct weekly work-hour limit for their degree "
            "level and TOPIK level, the deadline for their graduation date). If the "
            "answer depends on a detail they did not provide, state which missing "
            "detail changes it. Use ONLY values found in the excerpts."
        )
    user = (
        f"Question: {question}\n\n"
        f"Source excerpts (Korean government documents):\n{context}"
        f"{personalize}\n\n"
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


# =================================================================
#  최신화(freshness) — 비자처럼 예민한 도메인은 "언제 기준인지" 가 중요합니다.
#  Claude 같은 일반 모델은 학습 컷오프에 지식이 얼어 있어 "이게 최신인지" 를
#  스스로 알 수 없습니다. 우리는 근거 문서의 수집 기준일을 함께 노출하고,
#  규정은 바뀔 수 있으니 공식 창구 확인을 권합니다.
# =================================================================

_HIKOREA = "HiKorea (☎ 1345, hikorea.go.kr)"


def _freshness_footer(result: RetrievalResult) -> str:
    dates = [h.meta.get("retrieved", "") for h in result.hits if h.meta.get("retrieved")]
    asof = max(dates) if dates else ""
    head = f"🗓 Grounded in official sources as of {asof}. " if asof else "🗓 "
    return (
        f"\n\n{head}Immigration rules can change — confirm the current rule at "
        f"{_HIKOREA} or your university's international office before acting on it."
    )


def _smart_refusal(result: RetrievalResult, threshold: float) -> Answer:
    """막다른 '답 없음' 이 아니라, 다음 행동을 알려주는 거부.

    가장 가까웠던(임계값 미달) 문서의 주제를 짚어 "우리가 실제로 다루는 가장
    가까운 공식 주제" 를 안내하고, 개인 사안은 공식 창구로 넘깁니다. 그래도
    '검증된 출처가 없으면 답을 만들지 않는다' 는 원칙은 그대로입니다.
    """
    # 유사도가 이 값 미만이면 '가까운 주제' 제안은 무의미(완전 무관한 질문)하므로
    # 원칙적 거부 + 공식 창구 안내만 합니다. 그 이상이면 가장 가까운 주제를 제안.
    RELATED_FLOOR = 0.25
    nearest_title = nearest_topic = ""
    if result.hits and result.confidence >= RELATED_FLOOR:
        m = result.hits[0].meta
        nearest_title = m.get("title", "")
        nearest_topic = m.get("topic", "")

    reason = (
        f"We don't have a verified official source for this "
        f"(best match {result.confidence:.3f} < threshold {threshold:.2f}), so we won't "
        f"guess — a wrong visa or immigration answer can put you at real risk."
    )
    if nearest_title:
        reason += (
            f" The closest official topic we do cover is “{nearest_title}”"
            f"{f' ({nearest_topic})' if nearest_topic else ''} — try rephrasing toward that."
        )
    reason += f" For your specific situation, contact {_HIKOREA} or your international office."

    return {
        "route": "refused",
        "answer_text": "",
        "table_markdown": "",
        "chart": EMPTY_CHART,
        "sources": [],
        "confidence": round(result.confidence, 3),
        "refused_reason": reason,
    }


def _rag_answer(question: str, route: str, lang: str,
                profile: dict | None = None) -> Answer:
    r = _ensure_loaded()
    result = r.retrieve(question)

    if result.refused:
        # 거부 직전 구제: 규정 문서로는 못 답하지만 캠퍼스 생활/문화 질문일 수
        # 있습니다("Should I drink soju in Korean society?"). 의미가 가까운
        # 선배 팁이 있으면 차가운 거부 대신 그 팁을 돌려줍니다. 없으면 스마트 거부.
        rescue = rescue_local(question)
        if rescue is not None:
            return rescue
        return _smart_refusal(result, r.threshold)

    answer_text = _generate_answer(question, result, lang, profile) + _freshness_footer(result)
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
#  대조군 — "근거 없는 일반 AI" (Claude/ChatGPT 식). 발표에서 우리와 나란히
#  놓아 grounding 의 가치를 제품이 스스로 증명하게 합니다. 문서 컨텍스트를
#  전혀 주지 않고 LLM 에 그대로 물어봐, '그럴듯하지만 출처 없는' 답을 만듭니다.
# =================================================================

_UNGROUNDED_SYSTEM = """You are a generic general-purpose AI chatbot (like a typical \
assistant) with NO access to any official document, database, or the internet. Answer \
the user's question about studying, visas, or living in Korea from your own trained \
memory, the way a normal chatbot would. Be direct and give specific numbers, hours, \
or rules if you recall them. Do NOT cite sources (you have none), and do NOT add \
disclaimers telling the user to verify — just answer confidently in {language}."""


def ungrounded_answer(question: str, lang: str = "en",
                      profile: dict | None = None) -> dict:
    """근거 0 LLM 답변(대조군). contract.Answer 가 아니라 {"text": ...} 를 반환합니다.

    일부러 헤지·인용을 시키지 않아 '자신 있지만 검증 안 된' 답을 재현합니다.
    profile 을 주면 일반 AI 도 개인화는 하지만 여전히 출처가 없다는 점을
    대비로 보여줄 수 있습니다. 화면에서는 반드시 '⚠️ 출처 없음' 라벨과 함께.
    """
    language = {"en": "English", "ko": "Korean", "zh": "Chinese"}.get(lang, "English")
    system = _UNGROUNDED_SYSTEM.format(language=language)
    pblock = _profile_block(profile)
    user_msg = f"{question}\n\n{pblock}" if pblock else question
    try:
        resp = _client.chat.completions.create(
            model=_LLM,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.5,
            max_tokens=350,
            timeout=20,
        )
        text = (resp.choices[0].message.content or "").strip()
    except Exception:
        text = ("(A generic AI would answer here from memory — with no source, and no way "
                "to know whether the rule is still current.)")
    return {"text": text or "(no answer)"}


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


def answer_question(question: str, lang: str = "en",
                    profile: dict | None = None) -> Answer:
    """프론트가 호출하는 유일한 진입점.

    profile(선택): {visa, program, topik, nationality, grad_date, region} 중
    채워진 것만. 규정(rag/hybrid) 답변을 그 학생 기준으로 맞춤화합니다.
    기존 호출부는 profile 없이 그대로 동작합니다(하위호환).
    """
    if not question or not question.strip():
        return {
            "route": "refused", "answer_text": "", "table_markdown": "",
            "chart": EMPTY_CHART, "sources": [], "confidence": 0.0,
            "refused_reason": "Empty question.",
        }

    # 선배 라운지 — 라우터보다 먼저. 규정/통계로는 답할 수 없지만 유학생이
    # 실제로 궁금해하는 캠퍼스 생활·문화·행정 질문(예: "결혼으로 비자?",
    # "회식에서 술 꼭 마셔야 해?")을 큐레이션 팁으로 가로챕니다. 매칭이
    # 없으면 None → 평소대로 sql/rag/hybrid/refused 로 흐릅니다.
    local = match_local(question)
    if local is not None:
        return local

    decision = route_question(question)

    if decision.route == "sql":
        return _sql_answer(question, lang)

    if decision.route == "hybrid":
        # 규정(RAG) 부분은 지금 완전히 답합니다. 통계(SQL) 부분은 데이터 대기.
        ans = _rag_answer(question, "hybrid", lang, profile)
        if ans["route"] == "hybrid" and ans["answer_text"]:
            ans["answer_text"] += (
                "\n\nNote: the statistical part of this question (university "
                "numbers/rankings) will be answered once the dataset is loaded."
            )
        return ans

    # 기본: rag
    return _rag_answer(question, "rag", lang, profile)


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
