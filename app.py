import streamlit as st
import os
import cv2
import tempfile
from textblob import TextBlob
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# ===== PAGE CONFIG =====
st.set_page_config(page_title="Stress Detection AI", layout="centered")
st.title("🧠 Stress Detection System")
st.caption("Face + Text based Stress Analysis")

# ===== CASCADE LOADING =====
@st.cache_resource
def load_face_cascade():
    """Load face detection cascade classifier"""
    try:
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if cascade.empty():
            return None
        return cascade
    except Exception:
        return None

face_cascade = load_face_cascade()

# ===== FACE DETECTION =====
def detect_face(image_path):
    """Detect if image contains human face"""
    if face_cascade is None:
        return False
    
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=6,
            minSize=(40, 40)
        )
        return len(faces) > 0
    except Exception:
        return False

# ===== EMOTION PREDICTION =====
@st.cache_data
def predict_emotion_from_face(image_path):
    """Predict emotion based on face brightness"""
    if face_cascade is None:
        return "neutral"
    
    try:
        img = cv2.imread(image_path)
        if img is None:
            return "neutral"
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return "neutral"
        
        x, y, w, h = faces[0]
        face_region = gray[y:y+h, x:x+w]
        brightness = np.mean(face_region)
        
        if brightness > 150:
            return "happy"
        elif brightness < 100:
            return "sad"
        else:
            return "neutral"
    except Exception:
        return "neutral"

# ===== TEXT ANALYSIS =====
@st.cache_data
def analyze_text(text):
    """Analyze text sentiment"""
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity

        if polarity < -0.1:
            return "Negative", abs(polarity)
        elif polarity > 0.1:
            return "Positive", polarity
        else:
            return "Neutral", 0.0
    except Exception:
        return "Neutral", 0.0

# ===== STRESS CALCULATION =====
def calculate_stress(emotion, sentiment):
    """Calculate stress score from emotion and sentiment"""
    score = 0

    if emotion in ["sad", "angry", "fear"]:
        score += 50
    elif emotion == "neutral":
        score += 20
    elif emotion == "happy":
        score += 5
    else:
        score += 15

    if sentiment == "Negative":
        score += 40
    elif sentiment == "Neutral":
        score += 15

    return min(score, 100)

# ===== MAIN APP =====

# Show warning if face detection unavailable
if face_cascade is None:
    st.warning("⚠️ Face detection module not fully loaded. Using text analysis.")

# Input mode selection
mode = st.radio("Choose Input", ["Upload", "Camera"])
emotion = "Not detected"

# Image processing
if mode in ["Upload", "Camera"]:
    file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"]) if mode == "Upload" else st.camera_input("Capture Image")

    if file:
        st.image(file)

        try:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            tfile.write(file.read())
            tfile.flush()
            temp_path = tfile.name
            tfile.close()

            if not detect_face(temp_path):
                st.error("❌ No human face detected")
                emotion = "No Face"
            else:
                emotion = predict_emotion_from_face(temp_path).capitalize()

            try:
                os.unlink(temp_path)
            except Exception:
                pass

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            emotion = "Error"

# Text input section
st.markdown("---")
st.subheader("📝 Your Thoughts")
text = st.text_area("Enter how you feel", height=100, placeholder="Describe your current feelings...")

# Analysis button
if st.button("🔍 Analyze", use_container_width=True):
    if not text:
        st.warning("⚠️ Please enter your thoughts first")
    else:
        with st.spinner("Analyzing..."):
            sentiment, confidence = analyze_text(text)
            stress = calculate_stress(emotion, sentiment)

            st.success("✅ Analysis Complete!")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Emotion", emotion)
            with col2:
                st.metric("Sentiment", sentiment)
            with col3:
                st.metric("Stress Level", f"{stress}%")

            st.progress(stress / 100)

            if stress >= 70:
                st.error("🔴 High Stress - Consider taking a break or seeking support")
            elif stress >= 40:
                st.warning("🟡 Moderate Stress - Try relaxation techniques")
            else:
                st.success("🟢 Low Stress - You're doing great!")
