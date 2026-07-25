"""
반영본 ② — sql / rag / hybrid 답변 표면 폴리싱 (가장 많이 시연되는 경로)

문제: 거부·선배 카드는 이미 스타일링돼 있지만, 정작 데모의 주 경로인
      sql/rag/hybrid 답변은 st.write + 기본 st.bar_chart + 밋밋한 expander라
      "근거 기반"이라는 제품의 핵심 가치가 시각적으로 안 보입니다.

반영: app.py 의 render_answer() 안
      "# ---------- sql / rag / hybrid ----------" 아래 블록을
      통째로 아래 버전으로 교체하세요. (백엔드/계약 변경 없음, 순수 표현층)

핵심 추가:
  - route별 grounded 배지(📊 통계 / 📄 정부 규정 / 🔗 통계+규정)
  - 답변 텍스트를 은은한 카드로 감싸 "근거로 뒷받침됨"을 시각화
  - 출처를 점수칩(score chip)으로 요약 → 클릭 시 스니펫/링크
  - as-of(기준일)를 answer_text에서 감지해 상단 배지에 노출
    (freshness_footer가 이미 answer_text 끝에 붙으므로 별도 백엔드 불필요)
"""

import re
import pandas as pd
import streamlit as st

# ── 상단에 한 번만 두면 되는 헬퍼 ─────────────────────────────

_ROUTE_BADGE = {
    "sql":    ("📊", "From official statistics", "#1D4ED8", "#EFF4FF"),
    "rag":    ("📄", "Grounded in government sources", "#047857", "#ECFDF5"),
    "hybrid": ("🔗", "Statistics + rules, both grounded", "#7C3AED", "#F5F3FF"),
}

# answer_text 꼬리의 freshness footer에서 기준일 추출 (예: "as of 2026-07-24")
_ASOF = re.compile(r"as[- ]of\s*[:\-]?\s*(\d{4}[.\-/]\d{1,2}[.\-/]?\d{0,2})", re.I)


def _grounded_header(route: str, answer_text: str) -> None:
    icon, label, fg, bg = _ROUTE_BADGE.get(route, ("", "", "#374151", "#F3F4F6"))
    m = _ASOF.search(answer_text or "")
    asof = f'&nbsp;·&nbsp;<span style="opacity:.75">current as of {m.group(1)}</span>' if m else ""
    st.markdown(
        f"""
        <div style="
            display:inline-flex; align-items:center; gap:8px;
            background:{bg}; color:{fg};
            font-size:13px; font-weight:700;
            padding:6px 12px; border-radius:999px; margin:2px 0 10px 0;">
            <span style="font-size:14px">{icon}</span>{label}{asof}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _answer_card(text: str) -> None:
    st.markdown(
        f"""
        <div style="
            background:#FFFFFF; border:1px solid #E5E9F0;
            border-left:4px solid #1D4ED8; border-radius:10px;
            padding:16px 20px; margin-bottom:8px;
            font-size:15px; line-height:1.65; color:#1F2933;">
            {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _source_chips(sources: list) -> None:
    if not sources:
        return
    st.markdown(
        '<div style="font-size:13px;font-weight:700;color:#4B5563;margin:6px 0 4px">'
        'Sources cited</div>',
        unsafe_allow_html=True,
    )
    for src in sources:
        score = src.get("score", 0.0)
        # 점수를 신뢰도 칩 색으로 (높을수록 진한 초록)
        chip = "#047857" if score >= 0.55 else "#B45309" if score >= 0.45 else "#6B7280"
        with st.expander(f"{src['title']}  ·  match {score:.2f}"):
            st.markdown(
                f'<span style="display:inline-block;background:{chip};color:#fff;'
                f'font-size:11px;font-weight:700;padding:2px 8px;border-radius:6px;'
                f'margin-bottom:6px">official source</span>',
                unsafe_allow_html=True,
            )
            if src.get("snippet"):
                st.write(src["snippet"])
            if src.get("url"):
                st.markdown(f"[Open source ↗]({src['url']})")


# ── app.py render_answer() 의 sql/rag/hybrid 블록을 이걸로 교체 ──

def render_grounded(result) -> None:
    """route in (sql, rag, hybrid) 전용. render_answer 안에서 호출."""
    route = result["route"]
    _grounded_header(route, result.get("answer_text", ""))

    if result["answer_text"]:
        _answer_card(result["answer_text"])

    if result["table_markdown"]:
        st.markdown(result["table_markdown"])

    if result["chart"]["kind"] != "none":
        chart = result["chart"]
        df = pd.DataFrame(
            {chart["x_label"]: chart["labels"], chart["y_label"]: chart["values"]}
        ).set_index(chart["x_label"])
        {"bar": st.bar_chart, "line": st.line_chart, "scatter": st.scatter_chart}\
            .get(chart["kind"], st.bar_chart)(df)

    _source_chips(result.get("sources", []))


# ── 적용 방법 요약 ────────────────────────────────────────────
#  1) 위 헬퍼 4개(_grounded_header/_answer_card/_source_chips/render_grounded)를
#     app.py 상단(import 아래)에 붙여넣기.
#  2) render_answer() 안의 "# ---------- sql / rag / hybrid ----------" 이하
#     (st.write ~ 출처 expander 루프) 전체를  →  `render_grounded(result)` 한 줄로.
#  3) local/refused 분기는 기존 그대로 유지(이미 스타일링됨).
