"""
build_db.py — 공공데이터 CSV → SQLite (SQL 경로의 데이터 소스)

무엇을 만드는가
  data/raw/ 의 두 CSV 를 정제해 data/processed/kcampus.db 를 만듭니다.
  Text-to-SQL(src/sql_chain.py)이 이 DB 를 조회합니다.

    universities  대학 차원 테이블 (소재지·규모·전공·어학·기숙사)   univ_key PK
    students      대학 × 국적 × 체류자격 × 성별 → 인원 (팩트 테이블)

원본 (data.go.kr / 대학알리미 계열, 2025 기준)
  universities.csv            226행 · univ_key 로 조인
  students_by_nationality.csv 14,550행 · 197,163명

전처리에서 실제로 한 일 (발표/EDA 재료)
  1. 다중 캠퍼스 univ_key 중복 2건(경동대_, 을지대_) → 카운트 합산 병합
  2. 인코딩: 두 파일 모두 UTF-8-SIG (BOM). SSOT §8 인코딩 지뢰 회피
  3. 조인 정합성 검증: students 의 univ_key 173개 전부 universities 에 존재(고아 0)
  4. 결측 그대로 보존 — dorm_fee_min 등은 NULL 로 남깁니다.
     "왜 비었는가"(소규모 대학일수록 결측)가 EDA Step 4 의 핵심이라
     여기서 채우면 그 신호가 사라집니다.

사용법 / 단독 실행
  python src/build_db.py            # data/raw → data/processed/kcampus.db
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "data" / "raw"
DB_PATH = BASE / "data" / "processed" / "kcampus.db"

# DB 에 실을 대학 컬럼 (원본 36개 중 질의에 유용한 것만 선별 → 스키마 프롬프트 축소)
_SUM_COLS = [
    "total", "degree_total", "humanities", "natural_science", "engineering",
    "arts_sports", "medicine", "training_total", "language_training", "exchange",
    "lang_qualified_total", "topik4_plus", "dorm_capacity",
]
_FIRST_COLS = [   # 큰 캠퍼스 기준으로 대표값을 취함
    "year", "school_type", "establishment", "region",
    "univ_name", "univ_name_en", "dorm_capacity_rate", "dorm_fee_min",
    "dorm_available",
]


def _clean_name(name: str) -> str:
    return re.sub(r"_?제\d+캠퍼스$", "", str(name)).rstrip("_").strip()


def load_universities() -> pd.DataFrame:
    u = pd.read_csv(RAW / "universities.csv", encoding="utf-8-sig")

    # --- 다중 캠퍼스 중복 병합: 카운트는 합, 속성은 큰 캠퍼스 값 ---
    u = u.sort_values("total", ascending=False)
    agg = {c: "sum" for c in _SUM_COLS}
    agg.update({c: "first" for c in _FIRST_COLS})
    u = u.groupby("univ_key", as_index=False).agg(agg)

    # 이름에서 캠퍼스 접미사 정리
    u["univ_name"] = u["univ_name"].map(_clean_name)
    # 어학요건 충족 비율은 합산 후 재계산 (퍼센트를 더하면 안 되므로)
    u["lang_qualified_ratio"] = (
        u["lang_qualified_total"] / u["degree_total"].replace(0, pd.NA) * 100
    ).round(1)

    cols = ["univ_key"] + _SUM_COLS + ["lang_qualified_ratio"] + \
           [c for c in _FIRST_COLS if c != "dorm_capacity_rate"] + ["dorm_capacity_rate"]
    return u[[c for c in dict.fromkeys(cols)]]


def load_students() -> pd.DataFrame:
    return pd.read_csv(RAW / "students_by_nationality.csv", encoding="utf-8-sig")


_SCHEMA = """
DROP TABLE IF EXISTS students;
DROP TABLE IF EXISTS universities;

CREATE TABLE universities (
    univ_key             TEXT PRIMARY KEY,   -- 조인 키 (예: '가천대')
    univ_name            TEXT,               -- 한글 대학명
    univ_name_en         TEXT,               -- 영문 대학명
    region               TEXT,               -- 소재지 시·도 (서울, 경기 ...)
    establishment        TEXT,               -- 설립구분 (사립/국립/공립 ...)
    school_type          TEXT,               -- 학교종류 (대학교 ...)
    year                 INTEGER,            -- 기준연도 (2025)
    total                INTEGER,            -- 외국인 유학생 총원
    degree_total         INTEGER,            -- 학위과정 유학생 수
    humanities           INTEGER,            -- 인문사회 계열 학위 유학생
    natural_science      INTEGER,            -- 자연과학 계열
    engineering          INTEGER,            -- 공학 계열
    arts_sports          INTEGER,            -- 예체능 계열
    medicine             INTEGER,            -- 의학 계열
    training_total       INTEGER,            -- 비학위(연수) 유학생 수
    language_training    INTEGER,            -- 어학연수생 수
    exchange             INTEGER,            -- 교환학생 수
    lang_qualified_total INTEGER,            -- 한국어능력 요건 충족 유학생 수
    topik4_plus          INTEGER,            -- TOPIK 4급 이상 보유 유학생 수
    lang_qualified_ratio REAL,               -- 어학요건 충족 비율(%)
    dorm_capacity        REAL,               -- 기숙사 수용 인원
    dorm_capacity_rate   REAL,               -- 기숙사 수용률(%)
    dorm_fee_min         REAL,               -- 월 최소 기숙사비(원). NULL=미공시
    dorm_available       TEXT                -- 기숙사 제공 여부(예/아니오). NULL=미상
);

CREATE TABLE students (
    univ_key    TEXT,        -- universities.univ_key 로 조인
    nationality TEXT,        -- 국적 (베트남, 중국 ...)
    visa_status TEXT,        -- 체류자격/과정 (학사과정, 석사과정, 대학부설 어학원 연수 ...)
    gender      TEXT,        -- 성별 (남/여)
    headcount   INTEGER,     -- 인원수
    FOREIGN KEY (univ_key) REFERENCES universities(univ_key)
);

CREATE INDEX idx_students_key  ON students(univ_key);
CREATE INDEX idx_students_nat  ON students(nationality);
CREATE INDEX idx_students_visa ON students(visa_status);
CREATE INDEX idx_univ_region   ON universities(region);
"""


def build() -> Path:
    uni = load_universities()
    stu = load_students()

    # 조인 정합성 검증
    orphans = set(stu.univ_key) - set(uni.univ_key)
    if orphans:
        print(f"⚠️  students 에 있으나 universities 에 없는 키 {len(orphans)}개: "
              f"{sorted(orphans)[:5]}")

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.executescript(_SCHEMA)
        uni.to_sql("universities", conn, if_exists="append", index=False)
        stu.to_sql("students", conn, if_exists="append", index=False)
        conn.commit()

        nu = conn.execute("SELECT COUNT(*) FROM universities").fetchone()[0]
        ns = conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]
        tot = conn.execute("SELECT SUM(headcount) FROM students").fetchone()[0]
    finally:
        conn.close()

    print(f"✅ DB 생성: {DB_PATH}")
    print(f"   universities {nu}개 · students {ns}행 · 유학생 합계 {tot:,}명")
    return DB_PATH


if __name__ == "__main__":
    build()
