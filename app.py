"""
Stress Detection AI - Streamlit Cloud Compatible Version
"""

import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import os
import cv2
import tempfile
import numpy as np
from textblob import TextBlob


# Page configuration
st.set_page_config(
    page_title="Stress Detection AI",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title("🧠 Stress Detection System")
st.caption("Face + Text based Stress Analysis")


# Face cascade loading
@st.cache_resource
def get_cascade():
    try:
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        return cascade if not cascade.empty() else None
    except:
        return None


# Face detection
def has_face(image_path, cascade):
    if cascade is None:
        return False
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.2, 6, minSize=(40, 40))
        return len(faces) > 0
    except:
        return False


# UI elements
st.subheader("📸 Step 1: Upload or Capture Photo")
mode = st.radio("Choose input method:", ["Upload Photo", "Take Photo"], horizontal=True)

emotion = "Not detected"

if mode == "Upload Photo":
    uploaded = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded:
        st.image(uploaded, use_column_width=True)
        
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            
            cascade = get_cascade()
            if has_face(tmp_path, cascade):
                emotion = "Face detected"
            else:
                st.info("No face detected in image")
                emotion = "No face"
            
            try:
                os.unlink(tmp_path)
            except:
                pass
        except Exception as e:
            st.error(f"Error: {e}")
else:
    photo = st.camera_input("Take a photo")
    if photo:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(photo.read())
                tmp_path = tmp.name
            
            cascade = get_cascade()
            if has_face(tmp_path, cascade):
                emotion = "Face detected"
            else:
                st.info("No face detected")
                emotion = "No face"
            
            try:
                os.unlink(tmp_path)
            except:
                pass
        except Exception as e:
            st.error(f"Error: {e}")


# Text input
st.subheader("💭 Step 2: Share Your Thoughts")
user_text = st.text_area(
    "How are you feeling?",
    height=100,
    placeholder="Describe what's on your mind..."
)

# Analysis
st.subheader("📊 Step 3: Get Analysis")

if st.button("Analyze", use_container_width=True):
    if not user_text.strip():
        st.warning("Please share your thoughts first")
    else:
        try:
            blob = TextBlob(user_text)
            polarity = blob.sentiment.polarity
            
            sentiment = "Positive" if polarity > 0.1 else ("Negative" if polarity < -0.1 else "Neutral")
            
            stress_score = 0
            if emotion != "Not detected":
                stress_score += 30
            
            if sentiment == "Negative":
                stress_score += 50
            elif sentiment == "Neutral":
                stress_score += 20
            
            stress_score = min(stress_score, 100)
            
            st.success("✅ Analysis Complete!")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Emotion", emotion)
            col2.metric("Sentiment", sentiment)
            col3.metric("Stress %", f"{stress_score}%")
            
            st.progress(stress_score / 100)
            
            if stress_score >= 70:
                st.error("🔴 High Stress - Take a break!")
            elif stress_score >= 40:
                st.warning("🟡 Moderate Stress - Try relaxation")
            else:
                st.success("🟢 Low Stress - Keep it up!")
        except Exception as e:
            st.error(f"Analysis error: {e}")
