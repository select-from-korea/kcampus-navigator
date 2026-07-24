import streamlit as st
import pandas as pd
from src.pipeline import answer_question, baseline_answer, ungrounded_answer  # 실제 파이프라인
# 목업으로 UI만 볼 때는 위 줄 대신: from mock import answer_question

# ============================================================
# 데모 시나리오 (발표_슬라이드_구성 기준)
# ------------------------------------------------------------
# A(대조)   : "🆚 Compare" 체크 → 근거 없는 일반 AI vs 우리(인용/거부) 나란히
# B(sql)    : "Which universities in Seoul have the most international students?"
# C(rag)    : "Can I work part-time on a D-2 visa?"  → 인용 + 기준일(as-of)
# D(refused): "Should I marry a Korean to get a visa?" → 선배 라운지 / 스마트 거부
# ============================================================

st.set_page_config(page_title="K-Campus Navigator", layout="centered")
st.title("K-Campus Navigator")
st.caption(
    "Grounded visa & campus guidance for international students in Korea — "
    "cited from official sources, or it refuses. It won't guess."
)

# 결과를 rerun 사이에도 유지하기 위한 세션 상태
if "result" not in st.session_state:
    st.session_state.result = None
    st.session_state.ungrounded = None
    st.session_state.mode = "single"
    st.session_state.profile = None

with st.form(key="ask_form"):
    question = st.text_input("Ask your question (e.g. visa, university stats...)")

    # 개인화 프로필 — 채운 항목만 규정 답변을 그 학생 기준으로 맞춤화합니다.
    # (일반 AI 는 당신의 비자·성적·졸업일을 모릅니다. 이게 차별점.)
    with st.expander("🧑‍🎓 My profile — personalize the answer (optional)"):
        p1, p2, p3 = st.columns(3)
        with p1:
            visa = st.selectbox("Visa / stay status",
                                ["", "D-2 (student)", "D-4 (trainee)", "D-10 (job-seeking)",
                                 "E-7 (work)", "Other"])
            nationality = st.text_input("Nationality", placeholder="e.g. Vietnam")
        with p2:
            program = st.selectbox("Degree program",
                                   ["", "Undergraduate (1-2yr)", "Undergraduate (3-4yr)",
                                    "Master's", "Ph.D.", "Language course"])
            grad_date = st.text_input("Expected graduation", placeholder="e.g. 2027-02")
        with p3:
            topik = st.selectbox("TOPIK level", ["", "None", "1", "2", "3", "4", "5", "6"])
            region = st.text_input("Region in Korea", placeholder="e.g. Seoul")

    col_a, col_b = st.columns(2)
    with col_a:
        # 발표 장면 A: 근거 없는 일반 AI(=Claude/ChatGPT 식)와 우리를 나란히 비교
        compare = st.checkbox("🆚 Compare with a generic AI (no sources)")
    with col_b:
        # 순수 벡터검색(라우팅·SQL·거부 없음) 실패 시연 토글
        baseline = st.checkbox("🔬 Baseline: pure vector search")
    submitted = st.form_submit_button("Ask")

if submitted and question.strip():
    # 채워진 항목만 프로필로 (빈 문자열 제외)
    profile = {k: v for k, v in {
        "visa": visa, "program": program, "topik": topik,
        "nationality": nationality, "grad_date": grad_date, "region": region,
    }.items() if v and str(v).strip()}
    profile = profile or None
    st.session_state.profile = profile

    with st.spinner("Thinking..."):
        if compare:
            st.session_state.mode = "compare"
            st.session_state.ungrounded = ungrounded_answer(question, profile=profile)["text"]
            st.session_state.result = answer_question(question, profile=profile)
        elif baseline:
            st.session_state.mode = "single"
            st.session_state.ungrounded = None
            st.session_state.result = baseline_answer(question)
        else:
            st.session_state.mode = "single"
            st.session_state.ungrounded = None
            st.session_state.result = answer_question(question, profile=profile)


# ============================================================
#  렌더 — contract.Answer 하나를 그리는 재사용 함수
# ============================================================

def render_answer(result, compact: bool = False):
    route = result["route"]

    # ---------- local : 선배 라운지 (campus-life tip) ----------
    if route == "local":
        st.markdown(
            """
            <div style="
                background: linear-gradient(135deg, #FFF7ED 0%, #FEF3F2 100%);
                border-left: 4px solid #F97316;
                border-radius: 8px;
                padding: 16px 20px 8px 20px;
                margin-top: 8px;
            ">
                <div style="font-size:14px; font-weight:700; color:#9A3412;">
                    🎓 Sunbae Lounge · 선배 라운지
                </div>
                <div style="font-size:12px; color:#B45309; margin-top:2px;">
                    A campus-life tip from your senior — not an official regulation.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if result["answer_text"]:
            st.write(result["answer_text"])
        st.caption(
            "💡 For visa & official rules, ask a regulation question — "
            "those are answered only from government sources."
        )
        return

    # ---------- refused : 스마트 거부 ----------
    if route == "refused":
        st.markdown(
            f"""
            <div style="
                background-color:#F5F6F8;
                border-left: 4px solid #6B7280;
                border-radius: 8px;
                padding: 18px 22px;
                margin-top: 8px;
            ">
                <div style="font-size:15px; font-weight:600; color:#374151; margin-bottom:8px;">
                    🔒 No verified answer — we won't guess
                </div>
                <div style="font-size:14px; color:#4B5563; line-height:1.6;">
                    {result['refused_reason']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(f"Confidence: {result['confidence']:.0%} (below answer threshold)")
        return

    # ---------- sql / rag / hybrid ----------
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


# ============================================================
#  출력 — 단일 뷰 또는 대조 뷰
# ============================================================

result = st.session_state.result

if result is not None:
    prof = st.session_state.get("profile")
    if prof:
        chips = " · ".join(str(v) for v in prof.values())
        st.info(f"🧑‍🎓 Personalized for: {chips}")

    if st.session_state.mode == "compare" and st.session_state.ungrounded is not None:
        left, right = st.columns(2)
        with left:
            st.markdown("#### 🤖 Generic AI · no sources")
            st.caption(
                "Answers from memory frozen at a training cutoff. It can't cite a "
                "source, and can't tell you if the rule has since changed."
            )
            st.warning(st.session_state.ungrounded)
        with right:
            st.markdown("#### 🎓 K-Campus Navigator · grounded")
            st.caption("Cited from dated official sources — or it refuses.")
            render_answer(result, compact=True)
    else:
        render_answer(result)
