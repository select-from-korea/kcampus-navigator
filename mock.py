# ============================================================
#  mock.py - 프론트 개발용 목업
#  백엔드 완성 전까지 이걸로 UI를 만드세요.
#  app.py 에서:  from mock import answer_question
#  당일 교체:    from src.pipeline import answer_question
# ============================================================

import time
from contract import Answer, EMPTY_CHART


def answer_question(question: str, lang: str = "en") -> Answer:
    time.sleep(1.2)   # 실제 API 지연 시뮬레이션 (로딩 UI 테스트용)
    q = question.lower()

    # ---------- 케이스 1: 정량형 (SQL 경로) ----------
    if any(k in q for k in ["how many", "most", "top", "compare", "rank"]):
        return {
            "route": "sql",
            "answer_text": (
                "Seoul National University hosts the most international "
                "students in Seoul, with 3,142 enrolled as of 2024, "
                "followed by Korea University and Yonsei University."
            ),
            "table_markdown": (
                "| University | Students | Region |\n"
                "|---|---|---|\n"
                "| Seoul National Univ. | 3,142 | Seoul |\n"
                "| Korea Univ. | 2,880 | Seoul |\n"
                "| Yonsei Univ. | 2,455 | Seoul |"
            ),
            "chart": {
                "kind": "bar",
                "x_label": "University", "y_label": "Students",
                "labels": ["SNU", "Korea", "Yonsei"],
                "values": [3142, 2880, 2455],
            },
            "sources": [{
                "title": "대학별 외국인 유학생 현황 (2024)",
                "snippet": "서울대학교 3,142명, 고려대학교 2,880명 ...",
                "url": "https://www.data.go.kr/",
                "score": 0.91,
            }],
            "confidence": 0.88,
            "refused_reason": "",
        }

    # ---------- 케이스 2: 거부 (발표의 핵심 장면) ----------
    if any(k in q for k in ["marry", "citizenship", "will i get", "guarantee"]):
        return {
            "route": "refused",
            "answer_text": "",
            "table_markdown": "",
            "chart": EMPTY_CHART,
            "sources": [],
            "confidence": 0.21,
            "refused_reason": (
                "No supporting document was found above our confidence "
                "threshold (0.60). We do not generate answers about "
                "immigration status without a verifiable source, because "
                "an incorrect answer here can have serious consequences."
            ),
        }

    # ---------- 케이스 3: 정성형 (RAG 경로) ----------
    return {
        "route": "rag",
        "answer_text": (
            "Yes. D-2 visa holders may work part-time after obtaining "
            "prior permission from the immigration office. Weekly hour "
            "limits depend on your TOPIK level and academic standing."
        ),
        "table_markdown": "",
        "chart": EMPTY_CHART,
        "sources": [{
            "title": "출입국관리법 시행령 제23조",
            "snippet": (
                "유학(D-2) 자격을 가진 사람이 체류자격 외 활동허가를 받아 "
                "시간제 취업을 하는 경우 ..."
            ),
            "url": "https://www.hikorea.go.kr/",
            "score": 0.83,
        }],
        "confidence": 0.83,
        "refused_reason": "",
    }