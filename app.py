import streamlit as st
import pickle
import numpy as np

# 1. Load Models
model=pickle.load(open('model.pkl', 'rb')) # Random Forest
tfidf=pickle.load(open('tfidf.pkl', 'rb'))

st.title("📰 Fake News Detector")
st.write("Model: Random Forest | Accuracy: 99.80%")

user_input=st.text_area("News Content")

if st.button("Predict"):
    if user_input:
        # 2. Vectorize
        vector_input=tfidf.transform([user_input])

        # 3. Predict + Confidence %
        prediction=model.predict(vector_input)[0]
        probabilities=model.predict_proba(vector_input)[0] 

        fake_prob=probabilities[0]*100 # Class 0=Fake
        real_prob=probabilities[1]*100 # Class 1=Real

        st.write("---") # Line

        if prediction==1:
            st.success(f"✅ REAL News")
            st.info(f"Confidence: {real_prob:.2f}% Real | {fake_prob:.2f}% Fake")
            st.progress(int(real_prob))
        else:
            st.error(f"❌ FAKE News")
            st.info(f"Confidence: {fake_prob:.2f}% Fake | {real_prob:.2f}% Real")
            st.progress(int(fake_prob))

    else:
        st.warning("Please enter some text")
