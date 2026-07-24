import streamlit as st
import pickle
import numpy as np

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

@st.cache_resource
def load_model():
    model = pickle.load(open('model.pkl', 'rb'))
    vectorizer = pickle.load(open('tfidf.pkl', 'rb'))
    return model, vectorizer

model, tfidf = load_model()

with st.sidebar:
    st.title("📰 About")
    st.info(
        "This app uses Machine Learning to classify news as REAL or FAKE.\n\n"
        "**Model Used:** Random Forest\n"
        "**Feature:** TF-IDF Vectorizer\n"
        "**Dataset Accuracy:** 99.80%"
    )

    st.title("⚙️ How it Works")
    st.write("1. Enter news article in the text box")
    st.write("2. Click Predict button")
    st.write("3. Get REAL/FAKE prediction with confidence score")

    st.title("📊 Features")
    st.write("- TF-IDF Feature Extraction")
    st.write("- Confidence Percentage")
    st.write("- Progress Bar Visualization")

    st.markdown("---")
    st.caption("Made with Streamlit & Scikit-learn")

st.title("📰 Fake News Detector")
st.markdown("### Check if a news article is REAL or FAKE using AI")
st.write("Paste the full news article below and click Predict to check its authenticity.")

user_input = st.text_area("Enter News Content Here", height=250, placeholder="Paste news article here...")

col1, col2 = st.columns([1,4])
with col1:
    predict_btn = st.button("🔍 Predict", use_container_width=True)

if predict_btn:
    if user_input.strip():
        with st.spinner("Analyzing..."):
            vector_input = tfidf.transform([user_input])
            prediction = model.predict(vector_input)[0]
            probabilities = model.predict_proba(vector_input)[0]
            fake_prob = probabilities[0] * 100
            real_prob = probabilities[1] * 100

        st.write("---")

        col1, col2 = st.columns(2)
        with col1:
            if prediction == 1:
                st.success(f"✅ **PREDICTION: REAL NEWS**")
                st.metric(label="Real Confidence", value=f"{real_prob:.2f}%")
                st.progress(int(real_prob))
            else:
                st.error(f"❌ **PREDICTION: FAKE NEWS**")
                st.metric(label="Fake Confidence", value=f"{fake_prob:.2f}%")
                st.progress(int(fake_prob))

        with col2:
            st.write("**Confidence Breakdown**")
            st.write(f"Real: {real_prob:.2f}%")
            st.write(f"Fake: {fake_prob:.2f}%")
            st.bar_chart({"Real": real_prob, "Fake": fake_prob})

    else:
        st.warning("⚠️ Please enter some news content to predict")

st.markdown("---")
st.warning(
    "**Disclaimer:** This tool is for educational purposes only. "
    "The prediction is based on a trained ML model and may not be 100% accurate. "
    "Always verify news from trusted sources before sharing."
)

st.caption("© 2026 Fake News Detector Project")
