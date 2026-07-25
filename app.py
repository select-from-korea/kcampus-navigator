import streamlit as st
import pandas as pd
from src.pipeline import answer_question, ungrounded_answer  # 실제 파이프라인
from src.scholarships import known_schools

# ============================================================
# 데모 시나리오 (발표 덱 슬라이드 5 'Live Demo' · 대본 v7 기준)
# 검증: python eval/demo_check.py  → 7장면 route 확인 (현재 7/7)
# ------------------------------------------------------------
# A(대조)     : "🆚 Compare" 체크 → 근거 없는 일반 AI vs 우리(인용/거부) 나란히
#               "Which universities in Seoul have the most international students?"
# B(개인화)   : 프로필 없음 → 규정의 모든 가지(25/30/35/10/15h) 나열
#               페르소나 Roger 적용 → "30 hours" 한 줄
#               ② "What scholarships can I get at my school?" → 서울시립대 맞춤
# C(거부/구제): "How do I get Korean citizenship?" → 거부
#               "Should I marry a Korean to get a visa?" → 선배 라운지
# ============================================================

# 표기는 발표 덱과 통일: "Kcampus Navigator"
st.set_page_config(page_title="Kcampus Navigator", layout="centered")
st.title("Kcampus Navigator")
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

# ============================================================
#  데모 페르소나 — 발표용 원클릭 프리셋. 같은 질문을 프로필 없이 한 번,
#  Roger 로 한 번 던지면 답이 갈립니다:
#    프로필 없음 → 규정의 모든 가지 나열 (25/30/35/10/15시간 — 맞지만 쓸모없음)
#    Roger      → "주 30시간" 한 줄 (그에게 해당하는 가지)
#  발표 중 프로필 6칸을 손으로 채우면 20초가 날아가므로 드롭다운으로 끝냅니다.
# ============================================================

PERSONAS = {
    "— no profile (generic student) —": {},
    "🎓 Roger · 24 · Software Master's @ University of Seoul": {
        "visa": "D-2 (student)", "program": "Master's",
        "school": "University of Seoul (서울시립대학교)", "major": "Software",
        "topik": "4", "nationality": "Switzerland", "grad_date": "2028-02",
        "region": "Seoul",
    },
}

_FIELD_KEYS = ("visa", "program", "school", "major", "topik",
               "nationality", "grad_date", "region")

persona = st.selectbox("🧑‍🎓 Demo persona", list(PERSONAS), key="persona_name")
# 선택이 바뀐 순간에만 위젯 기본값을 덮어씁니다. 이후 사용자가 직접 고친 값은
# 유지됩니다(위젯 생성 전에 세팅해야 하므로 폼보다 위에 있습니다).
if st.session_state.get("_persona_applied") != persona:
    st.session_state["_persona_applied"] = persona
    for k in _FIELD_KEYS:
        st.session_state[f"p_{k}"] = PERSONAS[persona].get(k, "")

_SCHOOL_OPTIONS = [""] + known_schools() + ["Other / not listed"]

with st.form(key="ask_form"):
    question = st.text_input("Ask your question (e.g. visa, university stats...)")

    # 개인화 프로필 — 채운 항목만 규정 답변을 그 학생 기준으로 맞춤화합니다.
    # (일반 AI 는 당신의 비자·성적·졸업일을 모릅니다. 이게 차별점.)
    with st.expander("🧑‍🎓 My profile — personalize the answer (optional)", expanded=True):
        p1, p2, p3 = st.columns(3)
        with p1:
            visa = st.selectbox("Visa / stay status",
                                ["", "D-2 (student)", "D-4 (trainee)", "D-10 (job-seeking)",
                                 "E-7 (work)", "Other"], key="p_visa")
            nationality = st.text_input("Nationality", placeholder="e.g. Switzerland",
                                        key="p_nationality")
        with p2:
            program = st.selectbox("Degree program",
                                   ["", "Undergraduate (1-2yr)", "Undergraduate (3-4yr)",
                                    "Master's", "Ph.D.", "Language course"], key="p_program")
            major = st.text_input("Field of study", placeholder="e.g. Software",
                                  key="p_major")
        with p3:
            topik = st.selectbox("TOPIK level", ["", "None", "1", "2", "3", "4", "5", "6"],
                                 key="p_topik")
            region = st.text_input("Region in Korea", placeholder="e.g. Seoul",
                                   key="p_region")
        # 학교 — 장학금처럼 '학교만 아는' 정보를 개인화하는 키입니다.
        p4, p5 = st.columns(2)
        with p4:
            school = st.selectbox(
                "University", _SCHOOL_OPTIONS, key="p_school",
                help="Scholarship answers are personalized for the schools we have "
                     "curated data for. Other schools fall back to national scholarships.",
            )
        with p5:
            grad_date = st.text_input("Expected graduation", placeholder="e.g. 2028-02",
                                      key="p_grad_date")

    # 발표 장면 A: 근거 없는 일반 AI(=Claude/ChatGPT 식)와 우리를 나란히 비교
    compare = st.checkbox("🆚 Compare with a generic AI (no sources)")
    submitted = st.form_submit_button("Ask")

if submitted and question.strip():
    # 채워진 항목만 프로필로 (빈 문자열 제외)
    profile = {k: v for k, v in {
        "visa": visa, "program": program, "school": school, "major": major,
        "topik": topik, "nationality": nationality, "grad_date": grad_date,
        "region": region,
    }.items() if v and str(v).strip() and v != "Other / not listed"}
    profile = profile or None
    st.session_state.profile = profile

    with st.spinner("Thinking..."):
        if compare:
            st.session_state.mode = "compare"
            st.session_state.ungrounded = ungrounded_answer(question, profile=profile)["text"]
            st.session_state.result = answer_question(question, profile=profile)
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
            st.markdown("#### 🎓 Kcampus Navigator · grounded")
            st.caption("Cited from dated official sources — or it refuses.")
            render_answer(result, compact=True)
    else:
        render_answer(result)
