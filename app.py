import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
from mtcnn import MTCNN
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import tempfile
import pandas as pd

# -------- CONFIG --------
st.set_page_config(page_title="Stress Dectection AI ", layout="centered")
st.title("🧠 Multi-Modal Stress Detection System ")

detector = MTCNN()

# -------- NLP MODE --------
nlp_mode = st.selectbox(
    "Choose NLP Mode:",
    ["Fast (TextBlob)", "Advanced (AI Model)"]
)

# -------- FAST NLP --------
def analyze_fast(text):
    from textblob import TextBlob
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    if polarity < 0:
        return "Negative", abs(polarity)
    else:
        return "Positive", polarity

# -------- LAZY LOAD ADVANCED NLP --------
nlp = None

def get_nlp():
    global nlp
    if nlp is None:
        with st.spinner("Loading advanced AI model... ⏳ (one-time)"):
            from transformers import pipeline
            nlp = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
    return nlp

# -------- NLP SWITCH --------
def analyze_text(text):
    if nlp_mode == "Fast (TextBlob)":
        return analyze_fast(text)
    else:
        model = get_nlp()
        result = model(text)[0]
        sentiment = "Negative" if result['label'] == "NEGATIVE" else "Positive"
        return sentiment, result['score']

# -------- FACE CHECK --------
def is_face_present(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return False
    faces = detector.detect_faces(img)
    return len(faces) > 0

# -------- STRESS --------
def compute_stress(emotion, sentiment):
    score = 0

    if emotion in ["sad", "angry", "fear"]:
        score += 50
    elif emotion == "neutral":
        score += 20

    if sentiment == "Negative":
        score += 40

    return min(score, 100)

def suggestion(score):
    if score > 70:
        return "⚠️ High stress detected. Try rest or talk to someone."
    elif score > 40:
        return "🙂 Moderate stress. Take a short break."
    else:
        return "😊 Low stress. Keep it up!"

# -------- REAL-TIME CLASS --------
class EmotionDetector(VideoTransformerBase):
    def __init__(self):
        self.emotion = "Detecting..."

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        try:
            result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
            self.emotion = result[0]['dominant_emotion']
        except:
            pass

        cv2.putText(img, f"Emotion: {self.emotion}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img

# -------- INPUT MODE --------
mode = st.radio(
    "Choose Input Method:",
    ["Upload Image", "Capture Image", "Real-Time Webcam"]
)

image_path = None
detected_emotion = None
emotion_data = None

# -------- UPLOAD --------
if mode == "Upload Image":
    file = st.file_uploader("Upload Face Image", type=["jpg", "png", "jpeg"])

    if file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file.read())
        image_path = tfile.name
        st.image(file, caption="Uploaded Image", use_column_width=True)

# -------- CAMERA --------
elif mode == "Capture Image":
    cam_img = st.camera_input("Capture your face")

    if cam_img:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(cam_img.getvalue())
        image_path = tfile.name
        st.image(cam_img, caption="Captured Image", use_column_width=True)

# -------- REAL-TIME --------
else:
    st.info("🎥 Start webcam for live emotion detection")

    webrtc_ctx = webrtc_streamer(
        key="live",
        video_transformer_factory=EmotionDetector
    )

    if webrtc_ctx.video_transformer:
        detected_emotion = webrtc_ctx.video_transformer.emotion

# -------- TEXT --------
st.subheader("💬 Enter your thoughts")
text = st.text_area("How are you feeling?")

# -------- ANALYZE --------
if st.button("Analyze Stress"):

    if text:

        # -------- REAL-TIME --------
        if mode == "Real-Time Webcam":
            if not detected_emotion:
                st.warning("⚠️ Start webcam first")
                st.stop()
            emotion = detected_emotion

        # -------- IMAGE MODES --------
        else:
            if not image_path:
                st.warning("⚠️ Provide an image")
                st.stop()

            if not is_face_present(image_path):
                st.error("❌ No human face detected")
                st.stop()

            try:
                result = DeepFace.analyze(
                    img_path=image_path,
                    actions=['emotion'],
                    enforce_detection=True
                )
                emotion = result[0]['dominant_emotion']
                emotion_data = result[0]['emotion']
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.stop()

        # -------- TEXT --------
        sentiment, confidence = analyze_text(text)

        # -------- STRESS --------
        stress_score = compute_stress(emotion, sentiment)

        # -------- OUTPUT --------
        st.success("✅ Analysis Complete")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Emotion", emotion.capitalize())
            st.metric("Sentiment", sentiment)

        with col2:
            st.metric("Confidence", round(confidence, 2))
            st.metric("Stress Score", f"{stress_score}/100")

        # Emotion graph (only for image)
        if emotion_data:
            df = pd.DataFrame(list(emotion_data.items()), columns=["Emotion", "Score"])
            st.subheader("📊 Emotion Distribution")
            st.bar_chart(df.set_index("Emotion"))

        # Stress bar
        st.subheader("🧠 Stress Level")
        st.progress(stress_score)

        # Interpretation
        if stress_score > 70:
            st.error("High Stress ⚠️")
        elif stress_score > 40:
            st.warning("Moderate Stress")
        else:
            st.success("Low Stress 😊")

        # Suggestion
        st.subheader("💡 Recommendation")
        st.info(suggestion(stress_score))

    else:
        st.warning("⚠️ Please enter text")