# ============================================================
#  contract.py - 백엔드 <-> 프론트 인터페이스 계약 (v1, 7/22 확정)
#
#  ⚠️ 이 파일은 확정본입니다. 임의 변경 금지.
#     변경이 필요하면 반드시 팀 채팅방에 먼저 알릴 것.
#
#  프론트는 이 시그니처만 믿고 개발하세요.
#  당일에는 mock.answer_question 을 src.pipeline.answer_question 으로
#  교체하기만 하면 됩니다.
# ============================================================

from typing import TypedDict, Literal, List, Optional


class Source(TypedDict):
    title: str            # 문서 제목 (예: "출입국관리법 시행령 제23조")
    snippet: str          # 인용 원문 일부 (200자 이내)
    url: Optional[str]    # 출처 링크 (없으면 None)
    score: float          # 검색 유사도 0~1


class ChartData(TypedDict):
    kind: Literal["bar", "line", "scatter", "none"]
    x_label: str
    y_label: str
    labels: List[str]     # x축 항목 (예: 대학명)
    values: List[float]   # y축 값


class Answer(TypedDict):
    route: Literal["sql", "rag", "hybrid", "refused"]
    answer_text: str      # 사용자에게 보여줄 최종 답변 (영어)
    table_markdown: str   # SQL 결과 표. 없으면 ""
    chart: ChartData      # 없으면 kind="none"
    sources: List[Source] # 없으면 []
    confidence: float     # 0~1
    refused_reason: str   # route=="refused" 일 때만 채움. 아니면 ""


EMPTY_CHART: ChartData = {
    "kind": "none", "x_label": "", "y_label": "",
    "labels": [], "values": [],
}


def answer_question(question: str, lang: str = "en") -> Answer:
    """프론트는 이 함수 하나만 호출합니다."""
    raise NotImplementedError