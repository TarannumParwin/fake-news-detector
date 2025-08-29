import streamlit as st
import pickle
import re

# Load saved model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Text cleaning function (same as used in training)
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Streamlit UI
st.title("📰 Fake News Detector (Improved)")

st.write("Paste a news article below to check if it's **FAKE** or **REAL**:")

user_input = st.text_area("News Article Text", height=250)

if st.button("Check Now"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        probabilities = model.predict_proba(vectorized)[0]
        confidence = max(probabilities) * 100

        if prediction == "FAKE":
            st.error(f"❌ This news is **FAKE**. (Confidence: {confidence:.2f}%)")
        else:
            st.success(f"✅ This news is **REAL**. (Confidence: {confidence:.2f}%)")

        st.caption("⚠️ Note: This model is a prototype trained on a small sample dataset. It may not always be accurate.")
