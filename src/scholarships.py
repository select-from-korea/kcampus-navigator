"""
scholarships.py — 학교별 장학금 개인화 (My School layer)

왜 별도 레이어인가
  장학금은 우리 두 데이터 소스 어디에도 없습니다. 정부 규정 코퍼스(RAG)에는
  GKS 같은 '국가 장학금' 만 있고, 공공데이터 통계(SQL)에는 장학금 컬럼 자체가
  없습니다. 각 대학이 자기 홈페이지 공지로만 내기 때문입니다.
  → 검색으로도 집계로도 답할 수 없는 질문. 그래서 우리가 학교별 구조화 데이터를
    직접 만들었습니다 (docs/scholarships.json).

무엇이 개인화인가 (스크립트가 아니라는 증거)
  같은 질문 "What scholarships can I get at my school?" 이
    · 서울시립대 석사 · TOPIK 4  →  TOPIK 장학금 '지금 해당'
    · 숙명여대 학부 · TOPIK 2   →  같은 장학금 '아직 미달 — TOPIK 4 필요'
  로 갈립니다. 프로필의 학교·과정·TOPIK 이 실제로 결과를 바꿉니다.

정직성 (프로젝트 원칙 유지)
  - 문장을 LLM 이 생성하지 않습니다. 큐레이션 항목을 그대로 렌더 → 할루시네이션 0.
  - 우리 데이터에 없는 학교면 **아무 말도 지어내지 않고** None 을 돌려
    평소 경로(GKS 등 정부 문서 RAG)로 흘려보냅니다.
  - 모든 답변에 학교 공지 링크 · 기준일(as-of) · '국제교류처 확인' 안내를 붙입니다.

확장
  docs/scholarships.json 의 schools 에 항목 하나 추가 = 그 학교 지원 완료.
  코드 수정 없음. (지금은 발표 데모 범위로 서울시립대·숙명여대 2교)

단독 실행
  python src/scholarships.py        # 무API 스모크 (프로필별 분기 확인)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

BASE = Path(__file__).resolve().parent.parent
DATA_PATH = BASE / "docs" / "scholarships.json"

for _p in (str(BASE), str(BASE / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from contract import Answer, EMPTY_CHART, Source

# 장학금 질문으로 볼 트리거. 좁게 잡습니다 — 넓히면 등록금·생활비 질문까지
# 가로채 정부 문서 경로(정답)를 빼앗습니다.
_TRIGGERS = (
    "scholarship", "scholarships", "장학금", "장학",
    "tuition waiver", "tuition reduction", "tuition support",
    "financial aid", "financial support", "funding for my studies",
)

# 프로필 program 값 → 과정 단계. app.py 의 선택지와 같은 문자열입니다.
_GRAD_LEVELS = ("Master's", "Ph.D.")


def _norm(text: str) -> str:
    return " " + (text or "").lower().replace("’", "'").strip() + " "


def is_scholarship_question(question: str) -> bool:
    q = _norm(question)
    return any(t in q for t in _TRIGGERS)


class Scholarships:
    """docs/scholarships.json 을 1회 로드해 프로필에 맞는 항목을 고릅니다."""

    def __init__(self, data: dict):
        self.as_of: str = data.get("as_of", "")
        self.schools: list[dict] = data.get("schools", [])
        self.nationwide: list[dict] = data.get("nationwide", [])

    @classmethod
    def load(cls, path: Path = DATA_PATH) -> "Scholarships":
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    # ---- 학교 식별 -------------------------------------------------
    def find_school(self, school_text: str) -> Optional[dict]:
        """프로필의 학교 문자열을 별칭 표로 해석합니다. 없으면 None."""
        if not school_text:
            return None
        s = _norm(school_text)
        for sc in self.schools:
            for alias in sc.get("aliases", []):
                if alias.lower() in s:
                    return sc
        return None

    # ---- 자격 판정 -------------------------------------------------
    @staticmethod
    def _topik_num(topik: str) -> Optional[int]:
        try:
            return int(str(topik).strip())
        except (TypeError, ValueError):
            return None      # "None" / "" / 미입력

    def _eligibility(self, item: dict, program: str, topik: str) -> tuple[str, str]:
        """(상태, 사유) — 'match' | 'gap' | 'unknown' | 'level_mismatch'.

        지어내지 않습니다: json 에 적힌 조건(levels·topik_min·gpa_min)과
        프로필을 비교할 뿐입니다. 프로필에 없는 정보(성적 등)로 판단해야 하는
        항목은 'unknown' 으로 두고 '무엇이 결과를 바꾸는지' 를 밝힙니다.
        """
        levels = item.get("levels") or []
        if program and levels and program not in levels:
            return "level_mismatch", f"for {', '.join(levels)}"

        need = item.get("topik_min")
        have = self._topik_num(topik)
        if need is not None:
            if have is None:
                return "gap", f"needs TOPIK {need} — your profile has no TOPIK level yet"
            if have < need:
                return "gap", f"needs TOPIK {need} — you have TOPIK {have}"
            return "match", f"you hold TOPIK {have} ≥ {need}"

        gpa = item.get("gpa_min")
        if gpa is not None:
            return "unknown", (f"depends on your GPA — needs {gpa}+ last semester, "
                               f"which your profile doesn't include")
        return "match", ""

    # ---- 답변 조립 -------------------------------------------------
    def answer_for(self, school: dict, profile: dict) -> Answer:
        program = (profile.get("program") or "").strip()
        topik = (profile.get("topik") or "").strip()

        matched: list[str] = []
        gaps: list[str] = []
        unknown: list[str] = []
        skipped = 0
        needs_verify = False

        for item in school.get("scholarships", []):
            status, why = self._eligibility(item, program, topik)
            if status == "level_mismatch":
                skipped += 1
                continue
            needs_verify = needs_verify or bool(item.get("verify"))
            amount = item.get("amount_hint", "")
            head = f"**{item['name_en']}** ({item.get('name_ko','')})"
            body = item.get("benefit", "")
            money = f" — {amount}*" if amount else ""
            line = f"- {head}{money}\n  {body}"
            if status == "match":
                matched.append(line + (f"\n  ✅ {why}" if why else ""))
            elif status == "unknown":
                unknown.append(line + f"\n  ℹ️ {why}")
            else:
                gaps.append(line + f"\n  ⚠️ {why}")

        who = " · ".join(x for x in [program or None, f"TOPIK {topik}" if topik else None] if x)
        lines = [
            f"### Scholarships at {school['name_en']} ({school['name_ko']})",
            f"Matched to your profile{f' — {who}' if who else ''}.",
        ]
        if school.get("context_note"):
            lines.append(f"_{school['context_note']}_")

        if matched:
            lines.append("\n**You are eligible to apply for:**")
            lines.extend(matched)
        if gaps:
            lines.append("\n**Not yet — one condition short:**")
            lines.extend(gaps)
        if unknown:
            lines.append("\n**Depends on something we don't know about you:**")
            lines.extend(unknown)
        if not matched and not gaps and not unknown:
            lines.append("\nWe have no scholarship entry for your degree level at this "
                         "school yet — ask the international office directly.")
        if skipped:
            lines.append(f"\n_({skipped} more scholarship(s) at this school are for other "
                         f"degree levels, so they are not listed for you.)_")

        # 학교 장학금과 별개로, 국가 장학금은 누구에게나 열려 있습니다.
        for nat in self.nationwide:
            lines.append(f"\n**Nationwide:** {nat['name_en']} — {nat['benefit']}")

        if needs_verify:
            lines.append("\n\\* Amounts and tiers are set by the university each semester "
                         "and change — treat them as the published structure, not a promise.")
        lines.append(
            f"\n🗓 Curated from the university's own scholarship notice, as of {self.as_of}. "
            f"Scholarship rules are set by the school, not by immigration — confirm the "
            f"current terms and deadline with the {school['name_en']} international office "
            f"before you apply."
        )

        sources: list[Source] = [{
            "title": school.get("source_title", school["name_en"]),
            "snippet": (f"Curated scholarship data for {school['name_en']} "
                        f"({len(school.get('scholarships', []))} programs), as of {self.as_of}."),
            "url": school.get("source_url"),
            "score": 1.0,
        }]
        for nat in self.nationwide:
            sources.append({
                "title": nat.get("source_title", nat["name_en"]),
                "snippet": nat.get("benefit", "")[:200],
                "url": nat.get("source_url"),
                "score": 1.0,
            })

        return {
            "route": "rag",          # 계약 유지: 인용 있는 규정성 답변과 같은 취급
            "answer_text": "\n".join(lines),
            "table_markdown": "",
            "chart": EMPTY_CHART,
            "sources": sources,
            "confidence": 1.0,       # 큐레이션 데이터 — 검색 불확실성이 없습니다
            "refused_reason": "",
        }


# =================================================================
#  모듈 싱글턴 · 공개 함수
# =================================================================

_data: Optional[Scholarships] = None


def _ensure_loaded() -> Scholarships:
    global _data
    if _data is None:
        _data = Scholarships.load()
    return _data


def match_scholarship(question: str, profile: dict | None) -> Optional[Answer]:
    """[라우터 前] 장학금 질문 + 프로필에 아는 학교 → 학교별 맞춤 답변.

    다음 경우에는 None 을 돌려 평소 경로(정부 문서 RAG)로 보냅니다.
      · 장학금 질문이 아님
      · 프로필에 학교가 없음      → GKS 등 일반 장학금은 문서가 답합니다
      · 우리가 모르는 학교         → 지어내지 않습니다
    """
    if not profile or not is_scholarship_question(question):
        return None
    try:
        s = _ensure_loaded()
        school = s.find_school(profile.get("school", ""))
        if school is None:
            return None
        return s.answer_for(school, profile)
    except Exception:
        return None        # 데이터 파일 문제로 데모가 멈추지 않게


def known_schools() -> list[str]:
    """UI 셀렉트박스용 — 개인화가 실제로 동작하는 학교 목록."""
    try:
        return [f"{sc['name_en']} ({sc['name_ko']})" for sc in _ensure_loaded().schools]
    except Exception:
        return []


# =================================================================
#  단독 실행 — 무API 스모크
# =================================================================

_SMOKE = [
    # (질문, 프로필, 기대 매칭 여부)
    ("What scholarships can I get at my school?",
     {"school": "University of Seoul (서울시립대학교)", "program": "Master's", "topik": "4"}, True),
    ("What scholarships can I get at my school?",
     {"school": "Sookmyung Women's University (숙명여자대학교)",
      "program": "Undergraduate (3-4yr)", "topik": "2"}, True),
    ("장학금 뭐 받을 수 있어?",
     {"school": "시립대", "program": "Master's", "topik": "4"}, True),
    # --- 아래는 None 이어야 정상 ---
    ("What scholarships can I get?", {"program": "Master's"}, False),        # 학교 없음
    ("What scholarships can I get at my school?",
     {"school": "Seoul National University", "program": "Master's"}, False),  # 모르는 학교
    ("How many hours can I work on a D-2 visa?",
     {"school": "University of Seoul", "program": "Master's"}, False),        # 규정 질문
]

if __name__ == "__main__":
    # Windows 콘솔 기본 코드페이지(cp949)에서 '—' 같은 문자가 깨지지 않도록.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    s = _ensure_loaded()
    print(f"\nLoaded {len(s.schools)} schools (as of {s.as_of}): "
          f"{', '.join(sc['name_en'] for sc in s.schools)}\n" + "=" * 92)
    ok = 0
    for q, prof, expect in _SMOKE:
        a = match_scholarship(q, prof)
        got = a is not None
        ok += (got == expect)
        mark = "O" if got == expect else "X"
        print(f"{mark}  {q[:44]:44s} {str(prof.get('school',''))[:26]:26s} -> "
              f"{'personalized' if got else 'passes through'}")
    print("-" * 92)
    print(f"분기 정확도: {ok}/{len(_SMOKE)}\n")

    print("=" * 92)
    for q, prof, expect in _SMOKE[:2]:
        a = match_scholarship(q, prof)
        if a:
            print(f"\n[{prof['school']} · {prof['program']} · TOPIK {prof['topik']}]")
            print(a["answer_text"])
            print("-" * 92)
