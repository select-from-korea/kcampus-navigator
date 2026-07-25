"""
demo_check.py — 발표 직전 프리플라이트: 데모 6장면이 지금도 그대로 나오는가

왜 필요한가
  발표 데모는 즉흥 질문이 아니라 **사전 검증된 6개 질문**입니다. 그런데 API 키
  만료·DB 미빌드·인덱스 누락·라우터 프롬프트 수정 중 하나만 어긋나도 장면
  하나가 조용히 다른 경로로 샙니다. 무대 위에서 발견하면 늦습니다.
  → 발표 20분 전에 이걸 한 번 돌려서 7/7 이 나오는지만 봅니다.

무엇을 검사하는가
  각 장면의 **route 가 대본이 약속한 값과 같은지**. 답변 문장까지 채점하지는
  않습니다(LLM 이라 문장은 매번 다릅니다). 대신 route 와 핵심 수치(30/10시간)
  존재 여부만 봅니다 — 대본이 그 두 가지에 의존하기 때문입니다.

사용법
  python eval/demo_check.py          # 전체 (API 필요, 약 40초)
  python eval/demo_check.py --dry    # 무API — 장학금 분기·선배 키워드·파일 존재만
"""

from __future__ import annotations

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
for _p in (str(BASE), str(BASE / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 발표 대본과 동일한 페르소나 (app.py 의 PERSONAS 와 같은 값)
LINH = {
    "visa": "D-2 (student)", "program": "Master's",
    "school": "University of Seoul (서울시립대학교)", "topik": "4",
    "nationality": "Vietnam", "grad_date": "2027-02", "region": "Seoul",
}
MAI = {
    "visa": "D-2 (student)", "program": "Undergraduate (3-4yr)",
    "school": "Sookmyung Women's University (숙명여자대학교)", "topik": "2",
    "nationality": "Vietnam", "grad_date": "2028-02", "region": "Seoul",
}

# (장면, 프로필, 질문, 기대 route, 답변에 있어야 하는 문자열 중 하나 — 표기 흔들림 허용)
SCENES = [
    ("A  ", None, "Which universities in Seoul have the most international students?",
     "sql", ("한양", "Hanyang")),
    ("B-1", LINH, "How many hours can I work part-time on a D-2 visa?",
     "rag", ("30",)),
    ("B-2", MAI, "How many hours can I work part-time on a D-2 visa?",
     "rag", ("10",)),
    ("B-3", MAI, "What scholarships can I get at my school?",
     "rag", ("TOPIK 4",)),          # 미달 경고가 떠야 함
    ("C-1", MAI, "How do I get Korean citizenship?",
     "refused", ("threshold",)),
    ("C-2", MAI, "Should I marry a Korean to get a visa?",
     "local", ()),
    ("opt", LINH, "Which universities in Seoul have the most Vietnamese students?",
     "sql", ()),
]


def run_dry() -> int:
    """무API 검사 — 데이터 파일과 결정적 레이어만."""
    from scholarships import match_scholarship, known_schools
    from local import match_local

    fails = 0
    print("\n[파일]")
    for rel in ("data/processed/kcampus.db", "data/processed/vectors.npz",
                "docs/scholarships.json", "docs/local_tips.json"):
        exists = (BASE / rel).exists()
        fails += (not exists)
        print(f"  {'OK' if exists else 'XX'}  {rel}")

    print("\n[장학금 레이어]")
    print(f"  커버 학교: {', '.join(known_schools()) or '(없음!)'}")
    for tag, prof in (("Linh", LINH), ("Mai", MAI)):
        a = match_scholarship("What scholarships can I get at my school?", prof)
        ok = a is not None and a["answer_text"]
        fails += (not ok)
        print(f"  {'OK' if ok else 'XX'}  {tag} → {'개인화 답변' if ok else '매칭 실패'}")
    # 규정 질문이 장학금 레이어로 새면 안 됩니다
    leak = match_scholarship("How many hours can I work on a D-2 visa?", LINH)
    fails += (leak is not None)
    print(f"  {'OK' if leak is None else 'XX'}  규정 질문은 통과(가로채면 안 됨)")

    print("\n[선배 라운지 키워드]")
    hit = match_local("Should I marry a Korean to get a visa?")
    fails += (hit is None)
    print(f"  {'OK' if hit else 'XX'}  C-2 위장결혼 → 선배 라운지")

    print("\n" + ("무API 프리플라이트 통과" if not fails else f"⚠️ {fails}건 실패"))
    return 1 if fails else 0


def run_full() -> int:
    from pipeline import answer_question

    print(f"\n{'':4s} {'route':9s} {'expect':9s} {'conf':6s} question")
    print("-" * 96)
    ok = 0
    for tag, prof, q, expect, must in SCENES:
        a = answer_question(q, profile=prof)
        body = a["answer_text"] or a["refused_reason"]
        good = (a["route"] == expect) and (any(m in body for m in must) if must else True)
        ok += good
        print(f"{'OK' if good else 'XX':4s} {a['route']:9s} {expect:9s} "
              f"{a['confidence']:.3f}  {q[:52]}")
        print(f"     └ {' '.join(body.split())[:110]}")
        if not good and a["route"] == expect:
            print(f"     ⚠️ route 는 맞지만 기대 문자열 {list(must)} 중 아무것도 답변에 없습니다")
    print("-" * 96)
    print(f"데모 시나리오 일치: {ok}/{len(SCENES)}\n")
    return 0 if ok == len(SCENES) else 1


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")   # Windows cp949 대비
    except Exception:
        pass
    code = run_dry() if "--dry" in sys.argv else run_full()
    sys.exit(code)
