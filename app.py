import streamlit as st
from mock import answer_question  # 당일에 src.pipeline으로 교체

st.set_page_config(page_title="K-Campus Navigator", layout="centered")
st.title("K-Campus Navigator")

question = st.text_input("Ask your question (e.g. visa, university stats...)")

if st.button("Ask") and question:
    with st.spinner("Thinking..."):
        result = answer_question(question)

    route = result["route"]

    if route == "refused":
        st.error("⚠️ I can't answer this with confidence.")
        st.markdown(f"**Reason:** {result['refused_reason']}")
        st.caption(f"Confidence: {result['confidence']:.0%}")

    else:
        st.write(result["answer_text"])

        if result["table_markdown"]:
            st.markdown(result["table_markdown"])

        if result["chart"]["kind"] != "none":
            chart = result["chart"]
            import pandas as pd
            df = pd.DataFrame({chart["x_label"]: chart["labels"],
                                chart["y_label"]: chart["values"]})
            df = df.set_index(chart["x_label"])
            if chart["kind"] == "bar":
                st.bar_chart(df)
            elif chart["kind"] == "line":
                st.line_chart(df)

        if result["sources"]:
            st.subheader("Sources")
            for src in result["sources"]:
                with st.expander(f"{src['title']} (score: {src['score']:.2f})"):
                    st.write(src["snippet"])
                    if src["url"]:
                        st.markdown(f"[Link]({src['url']})")