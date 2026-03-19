import streamlit as st
import os
import cv2
import tempfile
from textblob import TextBlob

# -------- CONFIG --------
st.set_page_config(page_title="Stress Detection AI", layout="centered")
st.title("🧠 Multi-Modal Stress Detection System")
st.caption("AI-based Emotion + NLP Stress Detection")

# Detect deployment
IS_DEPLOY = os.getenv("STREAMLIT_SERVER_HEADLESS") == "true"

# -------- FACE DETECTION (NO TENSORFLOW) --------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def is_face_present(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return len(faces) > 0

# -------- NLP MODE --------
nlp_mode = st.selectbox(
    "Choose NLP Mode:",
    ["Fast (TextBlob)", "Advanced (AI Model)"]
)

# -------- NLP --------
def analyze_fast(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    sentiment = "Negative" if polarity < 0 else "Positive"
    return sentiment, abs(polarity)

nlp = None
def get_nlp():
    global nlp
    if nlp is None:
        from transformers import pipeline
        nlp = pipeline("sentiment-analysis")
    return nlp

def analyze_text(text):
    if nlp_mode == "Fast (TextBlob)" or IS_DEPLOY:
        return analyze_fast(text)
    else:
        model = get_nlp()
        result = model(text)[0]
        sentiment = "Negative" if result['label'] == "NEGATIVE" else "Positive"
        return sentiment, result['score']

# -------- INPUT MODE --------
mode = st.radio(
    "Choose Input Method:",
    ["Upload Image", "Capture Image", "Real-Time Webcam"]
)

emotion = "Not detected"

# -------- IMAGE INPUT --------
if mode in ["Upload Image", "Capture Image"]:

    file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"]) \
        if mode == "Upload Image" else st.camera_input("Capture Image")

    if file:
        st.image(file, use_column_width=True)

        # Save temp file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file.read())
        image_path = tfile.name

        # Face validation
        if not is_face_present(image_path):
            st.error("❌ No human face detected")
            emotion = "No Face"
        else:
            if not IS_DEPLOY:
                try:
                    from deepface import DeepFace
                    result = DeepFace.analyze(
                        img_path=image_path,
                        actions=['emotion'],
                        enforce_detection=True
                    )
                    emotion = result[0]['dominant_emotion']
                except:
                    st.error("❌ Error analyzing face")
                    emotion = "Error"
            else:
                emotion = "Face Detected (AI disabled in cloud)"

# -------- WEBCAM --------
elif mode == "Real-Time Webcam":

    if IS_DEPLOY:
        st.warning("⚠️ Webcam disabled in cloud deployment")
    else:
        from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
        from deepface import DeepFace

        class EmotionDetector(VideoTransformerBase):
            def __init__(self):
                self.emotion = "Detecting..."

            def transform(self, frame):
                img = frame.to_ndarray(format="bgr24")
                try:
                    result = DeepFace.analyze(
                        img,
                        actions=['emotion'],
                        enforce_detection=False
                    )
                    if result and 'dominant_emotion' in result[0]:
                        self.emotion = result[0]['dominant_emotion']
                except:
                    pass
                return img

        webrtc_ctx = webrtc_streamer(
            key="webcam",
            video_transformer_factory=EmotionDetector
        )

        if webrtc_ctx.video_transformer:
            emotion = webrtc_ctx.video_transformer.emotion

# -------- TEXT INPUT --------
text = st.text_area("💬 Enter your thoughts")

# -------- STRESS CALCULATION --------
def compute_stress(emotion, sentiment):
    score = 0

    if emotion in ["sad", "angry", "fear"]:
        score += 50
    elif emotion == "neutral":
        score += 20

    if sentiment == "Negative":
        score += 40

    return min(score, 100)

# -------- ANALYZE --------
if st.button("Analyze Stress"):

    if text:

        sentiment, confidence = analyze_text(text)
        stress = compute_stress(emotion, sentiment)

        st.success("✅ Analysis Complete")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Emotion", emotion)
            st.metric("Sentiment", sentiment)

        with col2:
            st.metric("Confidence", round(confidence, 2))
            st.metric("Stress Score", f"{stress}/100")

        st.subheader("🧠 Stress Level")
        st.progress(stress)

        if stress > 70:
            st.error("High Stress ⚠️")
        elif stress > 40:
            st.warning("Moderate Stress")
        else:
            st.success("Low Stress 😊")

    else:
        st.warning("⚠️ Enter text")