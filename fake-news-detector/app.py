import streamlit as st
import pickle
import re
import os

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# ---------------- Custom Styling (Inshorts Look) ----------------
st.markdown("""
<style>
body {
    background-color: #f4f6f8;
}

/* App Title */
.app-title {
    font-size: 34px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 4px;
}

.app-subtitle {
    text-align: center;
    color: #6c757d;
    margin-bottom: 30px;
}

/* News Card */
.news-card {
    background: white;
    padding: 22px;
    border-radius: 14px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.06);
    margin-bottom: 25px;
}

/* Result Cards */
.result {
    padding: 20px;
    border-radius: 12px;
    font-size: 18px;
    margin-top: 20px;
}

.fake {
    background-color: #fff1f1;
    border-left: 6px solid #ff4b4b;
}

.real {
    background-color: #f1fff7;
    border-left: 6px solid #00c853;
}

/* Button */
.stButton>button {
    width: 100%;
    background: linear-gradient(135deg, #ff512f, #dd2476);
    color: white;
    border-radius: 10px;
    height: 48px;
    font-size: 18px;
    font-weight: 600;
    border: none;
}

.stButton>button:hover {
    opacity: 0.9;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Load Model Safely ----------------
@st.cache_resource
def load_model():
    if not os.path.exists("model.pkl") or not os.path.exists("vectorizer.pkl"):
        st.error("❌ Model files missing. Please train the model first.")
        st.stop()
    return (
        pickle.load(open("model.pkl", "rb")),
        pickle.load(open("vectorizer.pkl", "rb"))
    )

model, vectorizer = load_model()

# ---------------- Text Cleaning ----------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+|#", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text

# ---------------- UI ----------------
st.markdown("<div class='app-title'>📰 Fake News Detector</div>", unsafe_allow_html=True)
st.markdown("<div class='app-subtitle'>Inshorts-style AI news credibility checker</div>", unsafe_allow_html=True)

st.markdown("<div class='news-card'>", unsafe_allow_html=True)

news = st.text_area(
    "Paste a news article",
    height=200,
    placeholder="Just like Inshorts — short, crisp, and clear news content..."
)

st.markdown("</div>", unsafe_allow_html=True)

if st.button("Check Credibility"):
    if news.strip() == "":
        st.warning("⚠️ Please paste a news article.")
    else:
        cleaned = clean_text(news)
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]
        confidence = max(model.predict_proba(vec)[0]) * 100

        if prediction == "FAKE":
            st.markdown(
                f"<div class='result fake'>❌ <b>FAKE NEWS</b><br>Confidence: {confidence:.2f}%</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='result real'>✅ <b>REAL NEWS</b><br>Confidence: {confidence:.2f}%</div>",
                unsafe_allow_html=True
            )

st.markdown("<br>", unsafe_allow_html=True)
st.caption("⚠️ Educational prototype. Results may vary.")
