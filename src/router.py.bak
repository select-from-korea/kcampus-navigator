"""
router.py — 질문을 SQL / RAG / Hybrid 경로로 분류

설계 원칙
  1. LLM 분류 실패 시 키워드 규칙으로 폴백 → 무조건 결과가 나옵니다.
     (API 타임아웃, JSON 파싱 실패, 쿼터 초과 모두 커버)
  2. "refused" 는 여기서 결정하지 않습니다.
     거부는 검색 점수가 임계값 미만일 때 retriever 단계에서 발생합니다.
     라우터가 판단하려 들면 오분류가 급증합니다. 책임 분리가 핵심.
  3. 분류 근거(reason)를 함께 반환합니다.
     평가셋 30문항을 돌렸을 때 "왜 틀렸는지"가 바로 보입니다.

사용법
  from src.router import route_question
  d = route_question("Which universities have the most Chinese students?")
  d.route    # "sql"
  d.reason   # "Requires ranking by student count."
  d.method   # "llm" | "keyword_fallback"

단독 실행
  python src/router.py              # 내장 테스트 10문항
  python src/router.py eval/questions.csv   # CSV 배치 평가
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI

BASE = Path(__file__).resolve().parent.parent
load_dotenv(BASE / ".env")

_client = OpenAI()
_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

Route = Literal["sql", "rag", "hybrid"]
VALID_ROUTES = ("sql", "rag", "hybrid")


@dataclass
class RouteDecision:
    route: Route
    reason: str
    method: str          # "llm" | "keyword_fallback"

    def __str__(self) -> str:
        return f"{self.route} ({self.method}): {self.reason}"


# =================================================================
#  키워드 폴백 규칙
#  LLM 이 죽어도 데모가 멈추지 않게 하는 안전망입니다.
# =================================================================

_SQL_KEYWORDS = [
    # 집계 · 개수
    "how many", "how much", "number of", "count", "total",
    # 순위 · 극값
    "most", "least", "top", "highest", "lowest", "largest", "smallest",
    "biggest", "rank", "ranking", "best",
    # 비교 · 통계
    "compare", "comparison", "average", "percentage", "percent",
    "rate", "trend", "distribution", "statistics",
    # 목록형 질문
    "which university", "which universities", "which region",
    "which city", "list of", "show me the",
]

_RAG_KEYWORDS = [
    # 가능 여부 · 자격
    "can i", "am i allowed", "do i need", "is it possible", "may i",
    "eligible", "eligibility", "qualify", "requirement", "required",
    # 절차 · 방법
    "how do i", "how can i", "procedure", "process", "steps to",
    "apply for", "application", "submit",
    # 규정 도메인
    "visa", "d-2", "d2", "topik", "immigration", "permission",
    "permit", "extend", "renew", "rule", "regulation", "law",
    "policy", "insurance", "alien registration",
]


def _keyword_route(question: str) -> RouteDecision:
    q = question.lower()
    sql_hits = [k for k in _SQL_KEYWORDS if k in q]
    rag_hits = [k for k in _RAG_KEYWORDS if k in q]

    if sql_hits and rag_hits:
        return RouteDecision(
            "hybrid",
            f"keyword: sql={sql_hits[:2]}, rag={rag_hits[:2]}",
            "keyword_fallback",
        )
    if sql_hits:
        return RouteDecision(
            "sql", f"keyword: {sql_hits[:2]}", "keyword_fallback"
        )
    if rag_hits:
        return RouteDecision(
            "rag", f"keyword: {rag_hits[:2]}", "keyword_fallback"
        )

    # 아무것도 안 걸리면 rag 가 안전합니다.
    # 검색 결과가 부실하면 자연스럽게 refused 로 흘러갑니다.
    return RouteDecision("rag", "keyword: no match, default to rag",
                         "keyword_fallback")


# =================================================================
#  LLM 분류
# =================================================================

_SYSTEM = """You are a query router for a Korean higher-education information system \
that helps international students considering study in Korea.

The system has TWO data sources:

[A] STRUCTURED TABLES (SQLite) — route "sql"
    Statistics about Korean universities:
      - international student counts by university / nationality / degree program
      - university location (city, province), size, public vs private
      - dropout rates, dormitory capacity, scholarship counts
    Use this when answering requires COUNTING, RANKING, COMPARING, FILTERING,
    or AGGREGATING numbers.

[B] REGULATION DOCUMENTS (vector + keyword search) — route "rag"
    Text documents about rules and procedures:
      - D-2 student visa rules, immigration procedures, part-time work permission
      - admission requirements, TOPIK level requirements
      - academic policies, alien registration, health insurance
    Use this when answering requires quoting a RULE, PROCEDURE, REQUIREMENT,
    or ELIGIBILITY condition.

Choose "hybrid" ONLY when the question explicitly asks for BOTH a statistic
AND a rule/procedure/requirement. A useful test: if you removed all
regulation documents, would the question still be fully answerable from
tables alone? If yes, choose "sql", not "hybrid".

Personal circumstances (nationality, budget, preferred region) are FILTERS
on the tables — they do not by themselves make a question "hybrid".

Important: you are NOT deciding whether the question can be answered.
Unanswerable or speculative questions still get routed to the best-fitting source.
Refusal is handled downstream by the retriever.

Reply with JSON only, no markdown:
{"route": "sql" | "rag" | "hybrid", "reason": "<one short sentence>"}"""

_FEWSHOT = [
    ("Which universities in Seoul have the most international students?",
     '{"route": "sql", "reason": "Requires ranking universities by student count."}'),
    ("Can I work part-time on a D-2 visa?",
     '{"route": "rag", "reason": "Asks about visa work permission rules."}'),
    ("I have a tight budget. Which universities offer scholarships, and what are the requirements?",
     '{"route": "hybrid", "reason": "Needs scholarship statistics plus eligibility rules."}'),
    ("How many Vietnamese students study in Busan?",
     '{"route": "sql", "reason": "Simple count filtered by nationality and region."}'),
    ("What TOPIK level do I need for a bachelor's program?",
     '{"route": "rag", "reason": "Admission requirement documented in regulations."}'),
    ("Will I definitely get a scholarship if I apply now?",
     '{"route": "rag", "reason": "Speculative, but scholarship conditions live in documents."}'),
]


def _llm_route(question: str) -> RouteDecision:
    messages = [{"role": "system", "content": _SYSTEM}]
    for q, a in _FEWSHOT:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": question})

    resp = _client.chat.completions.create(
        model=_MODEL,
        messages=messages,
        temperature=0,
        max_tokens=120,
        response_format={"type": "json_object"},
        timeout=10,
    )
    data = json.loads(resp.choices[0].message.content)
    route = data.get("route")
    if route not in VALID_ROUTES:
        raise ValueError(f"unexpected route from LLM: {route!r}")
    return RouteDecision(route, data.get("reason", "").strip(), "llm")


# =================================================================
#  공개 함수
# =================================================================

def route_question(question: str) -> RouteDecision:
    """질문을 분류합니다. 어떤 경우에도 예외를 던지지 않습니다."""
    if not question or not question.strip():
        return RouteDecision("rag", "empty question", "keyword_fallback")
    try:
        return _llm_route(question)
    except Exception as e:
        fb = _keyword_route(question)
        fb.reason = f"{fb.reason} | llm failed: {type(e).__name__}"
        return fb


# =================================================================
#  단독 실행 — 테스트
# =================================================================

_BUILTIN_TESTS = [
    # (질문, 기대 경로)  기대값은 검수 기준일 뿐 강제가 아닙니다.
    ("Which universities in Seoul have the most international students?", "sql"),
    ("How many Chinese students are studying in Daejeon?", "sql"),
    ("Compare Seoul and Busan for international students.", "sql"),
    ("What is the average dropout rate for international students?", "sql"),
    ("Can I work part-time on a D-2 visa?", "rag"),
    ("Do I need a TOPIK score to apply for a master's program?", "rag"),
    ("How do I extend my student visa?", "rag"),
    ("Will I get a scholarship if I apply now?", "rag"),
    ("I'm from Vietnam with a tight budget. Where should I apply?", "sql"),
    ("Which region has the cheapest dormitories and what are the rules for applying?", "hybrid"),
]


def _run(tests):
    hit = 0
    llm_used = 0
    print(f"\n{'':2s} {'route':7s} {'expect':7s} {'method':17s} question")
    print("-" * 100)
    for i, (q, expect) in enumerate(tests, 1):
        d = route_question(q)
        ok = (d.route == expect)
        hit += ok
        llm_used += (d.method == "llm")
        mark = "O" if ok else "X"
        print(f"{mark:2s} {d.route:7s} {expect:7s} {d.method:17s} {q[:60]}")
        print(f"{'':2s} {'':7s} {'':7s} {'':17s} └ {d.reason}")
    n = len(tests)
    print("-" * 100)
    print(f"일치: {hit}/{n} ({hit/n:.0%})   LLM 사용: {llm_used}/{n}\n")


def _load_csv(path: Path):
    """eval/questions.csv 형식: question,expected_route[,...]"""
    import csv
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            q = (r.get("question") or r.get("질문") or "").strip()
            e = (r.get("expected_route") or r.get("기대경로") or "rag").strip()
            if q:
                rows.append((q, e))
    return rows


if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])
        if not csv_path.is_absolute():
            csv_path = BASE / csv_path
        _run(_load_csv(csv_path))
    else:
        _run(_BUILTIN_TESTS)