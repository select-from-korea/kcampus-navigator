import streamlit as st
import pandas as pd
from src.pipeline import answer_question, baseline_answer  # 실제 파이프라인
# 목업으로 UI만 볼 때는 위 줄 대신: from mock import answer_question

# ============================================================
# 테스트용 질문 4종 (route별 확인용, mock.py 기준)
# ------------------------------------------------------------
# sql     : "Which universities in Seoul have the most international students?"
# rag     : "Can I work part-time on a D-2 visa?"
# hybrid  : "Which region has the cheapest dormitories and what are the rules?" -> X
# refused : "Will I get a scholarship if I apply now?"
#
# ⚠ 라이브 데모 시나리오(발표_슬라이드_구성 문서)와는 다름 — 데모는 A(실패시연)/B(sql)/C(rag)/D(refused) 4장면만 사용, hybrid 미포함
# 🚧 mock.py에 hybrid 케이스가 아직 없음. 현재 hybrid 예시 질문을 넣으면 조건 미매칭으로 기본값(rag)이 리턴됨.
# ============================================================

st.set_page_config(page_title="K-Campus Navigator", layout="centered")
st.title("K-Campus Navigator")

# 결과를 rerun 사이에도 유지하기 위한 세션 상태
if "result" not in st.session_state:
    st.session_state.result = None

with st.form(key="ask_form"):
    question = st.text_input("Ask your question (e.g. visa, university stats...)")
    # 데모 슬라이드 4 장면 A용: 순수 벡터검색(라우팅·SQL 없음) 실패 시연 토글
    baseline = st.checkbox("🔬 Baseline: pure vector search (no routing)")
    submitted = st.form_submit_button("Ask")

if submitted and question.strip():
    with st.spinner("Thinking..."):
        if baseline:
            st.session_state.result = baseline_answer(question)
        else:
            st.session_state.result = answer_question(question)

result = st.session_state.result

if result is not None:
    route = result["route"]

    if route == "refused":
        st.markdown(
            f"""
            <div style="
                background-color:#F5F6F8;
                border-left: 4px solid #6B7280;
                border-radius: 8px;
                padding: 20px 24px;
                margin-top: 16px;
            ">
                <div style="font-size:15px; font-weight:600; color:#374151; margin-bottom:8px;">
                    🔍 No verified answer available
                </div>
                <div style="font-size:14px; color:#4B5563; line-height:1.6;">
                    {result['refused_reason']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(f"Confidence: {result['confidence']:.0%} (below answer threshold)")

    # ---------- sql / rag / hybrid ----------
    else:
        if result["answer_text"]:
            st.write(result["answer_text"])

        if result["table_markdown"]:
            st.markdown(result["table_markdown"])

        if result["chart"]["kind"] != "none":
            chart = result["chart"]
            df = pd.DataFrame({
                chart["x_label"]: chart["labels"],
                chart["y_label"]: chart["values"],
            }).set_index(chart["x_label"])

            if chart["kind"] == "bar":
                st.bar_chart(df)
            elif chart["kind"] == "line":
                st.line_chart(df)
            elif chart["kind"] == "scatter":
                st.scatter_chart(df)

        if result["sources"]:
            st.subheader("Sources")
            for src in result["sources"]:
                with st.expander(f"{src['title']} (score: {src['score']:.2f})"):
                    st.write(src["snippet"])
                    if src["url"]:
                        st.markdown(f"[Link]({src['url']})")