import streamlit as st
from textblob import TextBlob
import pandas as pd

st.set_page_config(page_title="Stress AI (Lite)", layout="centered")

st.title("🧠 Stress Detection System (Lite Version)")
st.caption("AI-based sentiment analysis for stress detection")

# -------- TEXT INPUT --------
text = st.text_area("💬 Enter your thoughts")

# -------- ANALYSIS --------
def analyze_text(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    if polarity < 0:
        sentiment = "Negative"
    elif polarity > 0:
        sentiment = "Positive"
    else:
        sentiment = "Neutral"

    return sentiment, polarity

# -------- STRESS --------
def compute_stress(sentiment):
    if sentiment == "Negative":
        return 80
    elif sentiment == "Neutral":
        return 40
    else:
        return 10

# -------- BUTTON --------
if st.button("Analyze Stress"):

    if text:
        sentiment, polarity = analyze_text(text)
        stress_score = compute_stress(sentiment)

        st.success("✅ Analysis Complete")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Sentiment", sentiment)

        with col2:
            st.metric("Stress Score", f"{stress_score}/100")

        st.subheader("🧠 Stress Level")
        st.progress(stress_score)

        if stress_score > 70:
            st.error("High Stress ⚠️")
        elif stress_score > 40:
            st.warning("Moderate Stress")
        else:
            st.success("Low Stress 😊")

    else:
        st.warning("⚠️ Enter some text")