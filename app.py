"""A small English financial-news sentiment demo."""

from pathlib import Path
import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parent
EXAMPLES = {
    "Growth": "Operating profit rose 18% as demand improved.",
    "Pressure": "Sales fell sharply and the company reported an operating loss.",
    "Announcement": "The company will publish its annual report on Thursday.",
    "Mixed signals": "Revenue increased, but higher costs reduced operating profit.",
}
COLORS = {"positive": "#087E73", "neutral": "#71809A", "negative": "#D96C55"}

st.set_page_config(page_title="FinSignal", page_icon="↗", layout="centered")
st.markdown("""
<style>
.block-container {max-width:1100px;padding-top:2.5rem;}
.stApp {background:#F6F7F4;}
[data-testid="stVerticalBlockBorderWrapper"] {background:white;border-radius:14px;}
.sentiment-label {font-size:32px;font-weight:700;margin:8px 0 18px;}
.prob-row {display:flex;justify-content:space-between;font-size:14px;margin:15px 0 6px;}
.prob-track {height:8px;background:#EEF1EC;border-radius:6px;overflow:hidden;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_modified):
    return joblib.load(ROOT / "outputs/model.joblib")


def choose_example(name):
    st.session_state["sentence"] = EXAMPLES[name]
    st.session_state.pop("result", None)


st.title("FinSignal")
st.caption("Classify the sentiment of an English financial sentence.")
try:
    model = load_model((ROOT / "outputs/model.joblib").stat().st_mtime_ns)
except (OSError, ValueError) as error:
    st.error(f"Cannot load the model: {error}")
    st.stop()

left, right = st.columns([1.35, 1], gap="large")
with left:
    with st.container(border=True):
        st.subheader("Enter a sentence")
        for column, name in zip(st.columns(4), EXAMPLES):
            column.button(name, on_click=choose_example, args=(name,))
        text = st.text_area(
            "English financial sentence", key="sentence",
            height=180, max_chars=3000,
            placeholder="Example: Operating profit increased this quarter.",
        )
        if st.button("Analyze sentiment", type="primary", key="analyze"):
            sentence = " ".join(text.split())
            if not sentence:
                st.session_state.pop("result", None)
                st.error("Please enter a sentence first.")
            else:
                probabilities = model.predict_proba([sentence])[0]
                scores = dict(zip(model.classes_, probabilities.tolist()))
                st.session_state["result"] = {
                    "text": sentence, "scores": scores,
                    "label": max(scores, key=scores.get),
                }
with right:
    with st.container(border=True):
        st.subheader("Sentence sentiment")
        result = st.session_state.get("result")
        if result:
            label = result["label"]
            st.markdown(
                f'<div class="sentiment-label" style="color:{COLORS[label]}">{label.capitalize()}</div>',
                unsafe_allow_html=True,
            )
            for name in ["positive", "neutral", "negative"]:
                percent = result["scores"][name] * 100
                st.markdown(
                    f'<div class="prob-row"><span>{name.capitalize()}</span><strong>{percent:.1f}%</strong></div>'
                    f'<div class="prob-track"><div style="height:8px;width:{percent:.2f}%;background:{COLORS[name]}"></div></div>',
                    unsafe_allow_html=True,
                )
            if result["text"] != " ".join(text.split()):
                st.caption("Showing the previous result. Click Analyze to update.")
        else:
            st.info("Enter a sentence and click Analyze sentiment.")
        st.caption("Model-estimated probabilities; not calibrated confidence or stock-price predictions.")
