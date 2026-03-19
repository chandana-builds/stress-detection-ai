import streamlit as st
import os
import cv2
import tempfile
from textblob import TextBlob
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="Stress Detection AI", layout="centered")
st.title("🧠 Stress Detection System")
st.caption("Face + Text based Stress Analysis")

IS_DEPLOY = os.getenv("STREAMLIT_SERVER_HEADLESS") == "true" or os.getenv("STREAMLIT_APP") is not None

# -------- FACE DETECTION (STRICT) --------
@st.cache_resource
def load_face_cascade():
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

face_cascade = load_face_cascade()

def detect_face(image_path):
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
    except Exception as e:
        st.error(f"Face detection error: {str(e)}")
        return False

# -------- LOAD DEEPFACE MODEL --------
@st.cache_resource
def load_deepface_model():
    try:
        from deepface import DeepFace
        # Pre-load model to avoid timeout
        return DeepFace
    except ImportError:
        st.error("DeepFace not installed")
        return None

# -------- ANALYZE EMOTION --------
@st.cache_data
def analyze_emotion(image_path):
    try:
        DeepFace = load_deepface_model()
        if DeepFace is None:
            return "Error"
        
        result = DeepFace.analyze(
            img_path=image_path,
            actions=['emotion'],
            enforce_detection=True,
            silent=True
        )
        return result[0]['dominant_emotion']
    except Exception as e:
        return "Error"

# -------- NLP --------
@st.cache_data
def analyze_text(text):
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity

        if polarity < -0.1:
            return "Negative", abs(polarity)
        elif polarity > 0.1:
            return "Positive", polarity
        else:
            return "Neutral", 0
    except:
        return "Neutral", 0

# -------- INPUT --------
mode = st.radio("Choose Input", ["Upload", "Camera"])

emotion = "Not detected"

# -------- IMAGE --------
if mode in ["Upload", "Camera"]:
    file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"]) if mode=="Upload" else st.camera_input("Capture Image")

    if file:
        # Display image
        st.image(file)

        # Save to temporary file
        try:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            tfile.write(file.read())
            tfile.flush()
            temp_path = tfile.name
            tfile.close()

            # Check if face exists
            if not detect_face(temp_path):
                st.error("❌ No human face detected")
                emotion = "No Face"
            else:
                if not IS_DEPLOY:
                    # Local: Use full DeepFace
                    emotion = analyze_emotion(temp_path)
                else:
                    # Cloud: Use basic detection only
                    st.info("ℹ️ Using face detection only (full analysis on local)")
                    emotion = "Face Detected"
            
            # Cleanup
            import os as os_module
            try:
                os_module.unlink(temp_path)
            except:
                pass

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            emotion = "Error"

# -------- TEXT --------
st.markdown("---")
st.subheader("📝 Your Thoughts")
text = st.text_area("Enter how you feel", height=100, placeholder="Describe your current feelings...")

# -------- STRESS CALCULATION --------
def stress_calc(emotion, sentiment):
    score = 0

    # Emotion scoring
    if emotion in ["sad", "angry", "fear"]:
        score += 50
    elif emotion == "disgust":
        score += 40
    elif emotion == "neutral":
        score += 20
    elif emotion == "surprise":
        score += 15
    elif emotion == "happy":
        score += 5
    elif emotion == "Face Detected":
        score += 15

    # Sentiment scoring
    if sentiment == "Negative":
        score += 40
    elif sentiment == "Neutral":
        score += 15

    return min(score, 100)

# -------- ANALYZE --------
if st.button("🔍 Analyze", use_container_width=True):

    if not text:
        st.warning("⚠️ Please enter your thoughts first")
    else:
        with st.spinner("Analyzing..."):
            sentiment, conf = analyze_text(text)
            stress = stress_calc(emotion, sentiment)

            st.success("✅ Analysis Complete!")

            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Emotion", emotion)
            
            with col2:
                st.metric("Sentiment", sentiment)
            
            with col3:
                st.metric("Stress Level", f"{stress}%")

            # Progress bar
            st.progress(stress / 100)

            # Stress interpretation
            if stress >= 70:
                st.error("🔴 High Stress - Consider taking a break or seeking support")
            elif stress >= 40:
                st.warning("🟡 Moderate Stress - Try relaxation techniques")
            else:
                st.success("🟢 Low Stress - You're doing great!")