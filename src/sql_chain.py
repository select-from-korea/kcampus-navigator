"""
sql_chain.py — Text-to-SQL: 영어 질문 → SQLite 조회 → 표 + 차트 + 영어 답변

라우터가 "sql" 로 분류한 정량형 질문(개수·순위·비교·필터)을 처리합니다.
SSOT §4-1 의 정량형 경로이며, 벡터 검색(RAG)이 구조적으로 못 하는
집계 연산을 담당합니다 (발표 최고 득점 포인트).

핵심 난점 — 값이 한국어입니다
  질문은 영어("Vietnamese students in Seoul")인데 DB 값은 한국어
  (nationality='베트남', region='서울', visa_status='학사과정')입니다.
  스키마 프롬프트에 값 용어집을 넣어 LLM 이 한국어 리터럴로 필터하게 합니다.
  이 번역이 빠지면 WHERE 절이 전부 빈 결과를 냅니다.

안전장치
  1. SELECT 문 하나만 허용 (INSERT/UPDATE/DELETE/DROP/PRAGMA/ATTACH 차단)
  2. 세미콜론 다중 구문 차단
  3. 읽기 전용 커넥션(mode=ro) + LIMIT 자동 부착
  → Public 레포 데모에서 임의 쿼리로 DB 가 손상될 여지를 없앱니다.

반환
  contract.Answer (route="sql"). 결과가 없거나 SQL 생성 실패 시에도
  계약을 지키는 Answer 를 돌려줍니다 (파이프라인이 그대로 반환).

단독 실행
  python src/sql_chain.py           # 대표 정량 질문 스모크 테스트
"""

from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

BASE = Path(__file__).resolve().parent.parent
load_dotenv(BASE / ".env")

try:
    from .contract import Answer, EMPTY_CHART, ChartData
except ImportError:
    import sys
    sys.path.insert(0, str(BASE))
    from contract import Answer, EMPTY_CHART, ChartData

_client = OpenAI()
_LLM = os.getenv("LLM_MODEL", "gpt-4o-mini")
DB_PATH = BASE / "data" / "processed" / "kcampus.db"

MAX_ROWS = 100        # 결과·차트 상한
CHART_ROWS = 12       # 막대차트에 그릴 최대 항목 수


# =================================================================
#  스키마 프롬프트 — 값이 한국어라는 점을 LLM 에 명확히 알립니다
# =================================================================

SCHEMA_PROMPT = """You write ONE SQLite SELECT query for a database of international \
students at Korean universities (2025). Return ONLY the SQL, no explanation, no code fence.

TABLES

universities (one row per university, join key = univ_key)
  univ_key TEXT PK, univ_name TEXT(Korean), univ_name_en TEXT(English),
  region TEXT(Korean province/city), establishment TEXT(사립=private/국립=national/공립=public),
  school_type TEXT, year INTEGER,
  total INTEGER            -- total international students
  degree_total INTEGER, humanities, natural_science, engineering, arts_sports, medicine  -- degree students by field
  training_total, language_training, exchange   -- non-degree students
  lang_qualified_total INTEGER, topik4_plus INTEGER  -- # students meeting Korean req / with TOPIK 4+
  lang_qualified_ratio REAL   -- % of degree students meeting language requirement
  dorm_capacity REAL, dorm_capacity_rate REAL,  -- dormitory beds / coverage %
  dorm_fee_min REAL           -- minimum monthly dorm fee in KRW (NULL = not disclosed)
  dorm_available TEXT         -- '예'(yes)/'아니오'(no)/NULL

students (one row per univ × nationality × visa_status × gender)
  univ_key TEXT (join to universities), nationality TEXT(Korean), visa_status TEXT(Korean),
  gender TEXT('남'=male/'여'=female), headcount INTEGER
  → student counts come from SUM(headcount), never COUNT(*).

VALUES ARE KOREAN. Translate English filters to the exact Korean literal:
  regions: 서울=Seoul, 부산=Busan, 대구=Daegu, 인천=Incheon, 광주=Gwangju, 대전=Daejeon,
    울산=Ulsan, 세종=Sejong, 경기=Gyeonggi, 강원=Gangwon, 충북/충남=North/South Chungcheong,
    전북/전남=North/South Jeolla, 경북/경남=North/South Gyeongsang, 제주=Jeju
  nationalities: 중국=China, 베트남=Vietnam, 몽골=Mongolia, 우즈베키스탄=Uzbekistan,
    네팔=Nepal, 미얀마=Myanmar, 일본=Japan, 미국=USA, 인도=India, 러시아(연방)=Russia
    (for others use the standard Korean country name)
  visa_status literals: '학사과정'(bachelor), '석사과정'(master), '박사과정'(doctoral),
    '전문학사과정'(associate), '교환학생'(exchange), '대학부설 어학원 연수'(language training),
    '외국어연수생', '학술연구기관 특정연구자'

RULES
  - Output a single SELECT only. No INSERT/UPDATE/DELETE/DDL/PRAGMA/ATTACH, no semicolons.
  - Join students→universities on univ_key when you need university attributes.
  - For "how many students" use SUM(headcount). For rankings, GROUP BY + ORDER BY ... DESC.
  - Select univ_name (and univ_name_en when helpful) rather than univ_key for readability.
  - Always add LIMIT (<= 100). Prefer LIMIT 10 for "top" questions.
  - If the question cannot be answered from these tables, return exactly: SELECT NULL LIMIT 0

EXAMPLE — ranking universities by a nationality (note the JOIN; univ_name lives in universities):
  SELECT u.univ_name, u.univ_name_en, SUM(s.headcount) AS students
  FROM students s JOIN universities u ON s.univ_key = u.univ_key
  WHERE s.nationality = '중국'
  GROUP BY u.univ_key ORDER BY students DESC LIMIT 10
"""

_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|pragma|attach|"
    r"detach|vacuum|reindex)\b", re.IGNORECASE)


# =================================================================
#  SQL 생성 · 검증 · 실행
# =================================================================

def generate_sql(question: str) -> str:
    resp = _client.chat.completions.create(
        model=_LLM,
        messages=[
            {"role": "system", "content": SCHEMA_PROMPT},
            {"role": "user", "content": question},
        ],
        temperature=0,
        max_tokens=300,
        timeout=15,
    )
    sql = (resp.choices[0].message.content or "").strip()
    # 코드펜스 제거
    sql = re.sub(r"^```(?:sql)?|```$", "", sql, flags=re.IGNORECASE).strip()
    return sql


def repair_sql(question: str, bad_sql: str, error: str) -> str:
    """실행에 실패한 SQL 을 에러 메시지와 함께 되먹여 1회 교정합니다.

    가장 흔한 실패는 JOIN 누락(예: students 에서 univ_name 조회)입니다.
    에러를 그대로 보여주면 모델이 대부분 한 번에 고칩니다.
    """
    resp = _client.chat.completions.create(
        model=_LLM,
        messages=[
            {"role": "system", "content": SCHEMA_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": bad_sql},
            {"role": "user", "content": (
                f"That query failed with error: {error}\n"
                f"Return a corrected single SELECT query only.")},
        ],
        temperature=0,
        max_tokens=300,
        timeout=15,
    )
    sql = (resp.choices[0].message.content or "").strip()
    return re.sub(r"^```(?:sql)?|```$", "", sql, flags=re.IGNORECASE).strip()


def _sanitize(sql: str) -> str:
    """단일 SELECT 만 통과시키고 LIMIT 을 강제합니다. 위반 시 예외."""
    s = sql.strip().rstrip(";").strip()
    if ";" in s:
        raise ValueError("multiple statements are not allowed")
    if not re.match(r"(?is)^\s*(select|with)\b", s):
        raise ValueError("only SELECT queries are allowed")
    if _FORBIDDEN.search(s):
        raise ValueError("forbidden keyword in query")
    if not re.search(r"(?is)\blimit\b", s):
        s += f"\nLIMIT {MAX_ROWS}"
    return s


def run_query(sql: str) -> pd.DataFrame:
    """읽기 전용 커넥션으로 실행합니다."""
    uri = f"file:{DB_PATH.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        return pd.read_sql_query(sql, conn)
    finally:
        conn.close()


# =================================================================
#  결과 → 표 · 차트 · 영어 답변
# =================================================================

def _df_to_markdown(df: pd.DataFrame) -> str:
    """의존성(tabulate) 없이 DataFrame 을 마크다운 표로 변환합니다."""
    cols = [str(c) for c in df.columns]
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, r in df.iterrows():
        cells = []
        for v in r.tolist():
            if isinstance(v, float) and v.is_integer():
                v = int(v)
            cells.append("" if pd.isna(v) else str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep, *rows])


def _to_chart(df: pd.DataFrame) -> ChartData:
    """텍스트 1열 + 숫자 1열이고 2행 이상이면 막대차트를 만듭니다."""
    if len(df) < 2:
        return EMPTY_CHART
    text_cols = [c for c in df.columns if df[c].dtype == object]
    num_cols = [c for c in df.columns
                if pd.api.types.is_numeric_dtype(df[c])]
    if len(text_cols) != 1 or not num_cols:
        return EMPTY_CHART
    label_col, value_col = text_cols[0], num_cols[0]
    d = df.head(CHART_ROWS)
    return {
        "kind": "bar",
        "x_label": str(label_col),
        "y_label": str(value_col),
        "labels": [str(x) for x in d[label_col].tolist()],
        "values": [float(v) for v in d[value_col].fillna(0).tolist()],
    }


def _summarize(question: str, df: pd.DataFrame, lang: str) -> str:
    """결과 표를 근거로 영어 요약을 생성합니다. 실패 시 템플릿 폴백."""
    language = {"en": "English", "ko": "Korean", "zh": "Chinese"}.get(lang, "English")
    table = _df_to_markdown(df.head(15))
    try:
        resp = _client.chat.completions.create(
            model=_LLM,
            messages=[
                {"role": "system", "content": (
                    f"Summarize the SQL result for the user in {language}. "
                    "State the concrete numbers from the table. Be concise "
                    "(1-3 sentences). Do not invent values not in the table. "
                    "Korean names may be kept as-is.")},
                {"role": "user", "content":
                    f"Question: {question}\n\nResult table:\n{table}"},
            ],
            temperature=0.2,
            max_tokens=250,
            timeout=15,
        )
        out = (resp.choices[0].message.content or "").strip()
        if out:
            return out
    except Exception:
        pass
    return f"The query returned {len(df)} row(s). See the table below."


def _empty_answer(msg: str) -> Answer:
    return {
        "route": "sql", "answer_text": msg, "table_markdown": "",
        "chart": EMPTY_CHART, "sources": [], "confidence": 0.0,
        "refused_reason": "",
    }


# =================================================================
#  공개 함수 — 파이프라인이 호출
# =================================================================

def run_sql_question(question: str, lang: str = "en") -> Answer:
    if not DB_PATH.exists():
        return _empty_answer(
            "The statistics database is not built yet. Run "
            "`python src/build_db.py` first.")
    try:
        raw_sql = generate_sql(question)
        sql = _sanitize(raw_sql)
    except Exception as e:
        return _empty_answer(
            f"I could not turn this into a safe database query ({type(e).__name__}).")

    try:
        df = run_query(sql)
    except Exception as e:
        # 1회 자가수정: 실패한 SQL + 에러를 되먹여 교정 시도
        try:
            fixed = _sanitize(repair_sql(question, sql, str(e)))
            df = run_query(fixed)
            sql = fixed
        except Exception:
            return _empty_answer(
                f"The database query failed ({type(e).__name__}).")

    if df.empty:
        return {
            "route": "sql",
            "answer_text": ("No matching records were found in the university "
                            "statistics for this question."),
            "table_markdown": "", "chart": EMPTY_CHART, "sources": [],
            "confidence": 0.0, "refused_reason": "",
            "debug_sql": sql,   # type: ignore  (디버깅용 여분 키, 프론트는 무시)
        }  # type: ignore

    df = df.head(MAX_ROWS)
    answer_text = _summarize(question, df, lang)
    source = {
        "title": "대학별 외국인 유학생 현황 · 대학 기본정보 (2025)",
        "snippet": f"SQL: {sql[:180]}",
        "url": "https://www.data.go.kr/",
        "score": 1.0,
    }
    return {
        "route": "sql",
        "answer_text": answer_text,
        "table_markdown": _df_to_markdown(df),
        "chart": _to_chart(df),
        "sources": [source],
        "confidence": 1.0,
        "refused_reason": "",
    }


# =================================================================
#  단독 실행 — 스모크 테스트
# =================================================================

_SMOKE = [
    "Which universities in Seoul have the most international students?",
    "How many Vietnamese students study in Korea?",
    "Which universities have the most Chinese students?",
    "Compare the number of international students in Seoul vs Busan.",
    "Which universities offer the cheapest dormitories?",
    "How many students are in master's programs nationwide?",
]

if __name__ == "__main__":
    print(f"\nDB: {DB_PATH}\nLLM: {_LLM}\n" + "=" * 92)
    for q in _SMOKE:
        a = run_sql_question(q)
        print(f"\nQ  {q}")
        print(f"   A: {a['answer_text'][:220]}")
        if a["table_markdown"]:
            first = a["table_markdown"].splitlines()[:4]
            for line in first:
                print("     " + line)
        if a["chart"]["kind"] != "none":
            print(f"   chart: {a['chart']['kind']} "
                  f"({len(a['chart']['labels'])} bars)")
    print("\n" + "=" * 92 + "\n")
