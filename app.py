import streamlit as st
import os
import cv2
import tempfile
from textblob import TextBlob

# -------- CONFIG --------
st.set_page_config(page_title="Stress Detection AI", layout="centered")
st.title("🧠 Multi-Modal Stress Detection System")
st.caption("AI-based Emotion + NLP Stress Detection")

IS_DEPLOY = os.getenv("STREAMLIT_SERVER_HEADLESS") == "true"

# -------- FACE DETECTION (ROBUST) --------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def is_face_present(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Try multiple scales for better accuracy
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(30, 30)
    )

    # Fallback: upscale and try again
    if len(faces) == 0:
        resized = cv2.resize(gray, None, fx=1.5, fy=1.5)
        faces = face_cascade.detectMultiScale(
            resized,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(30, 30)
        )

    return len(faces) > 0

# -------- NLP --------
def analyze_fast(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    sentiment = "Negative" if polarity < 0 else "Positive"
    return sentiment, abs(polarity)

def analyze_text(text):
    # Always safe for cloud
    return analyze_fast(text)

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

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file.read())
        image_path = tfile.name

        if not is_face_present(image_path):
            st.error("❌ No human face detected (try clearer front face)")
            emotion = "No Face"
        else:
            # Try DeepFace safely
            try:
                from deepface import DeepFace

                result = DeepFace.analyze(
                    img_path=image_path,
                    actions=['emotion'],
                    enforce_detection=True
                )
                emotion = result[0]['dominant_emotion']

            except Exception as e:
                # Cloud fallback (no crash)
                emotion = "Face Detected (AI unavailable here)"
                st.info("ℹ️ Emotion model not available in this environment")

# -------- WEBCAM --------
elif mode == "Real-Time Webcam":

    try:
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

    except:
        st.warning("⚠️ Webcam not supported in this environment")

# -------- TEXT --------
text = st.text_area("💬 Enter your thoughts")

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

        st.progress(stress)

    else:
        st.warning("⚠️ Enter text")