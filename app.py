import streamlit as st
import os

# -------- DETECT ENVIRONMENT --------
IS_DEPLOY = os.getenv("STREAMLIT_SERVER_HEADLESS") == "true"

st.set_page_config(page_title="Stress Detection AI", layout="centered")
st.title("🧠 Multi-Modal Stress Detection System")

st.caption("AI-based Emotion + NLP Stress Detection")

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
    sentiment = "Negative" if polarity < 0 else "Positive"
    return sentiment, abs(polarity)

# -------- ADVANCED NLP (ONLY LOCAL) --------
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

# -------- IMAGE MODES --------
if mode in ["Upload Image", "Capture Image"]:

    if mode == "Upload Image":
        file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    else:
        file = st.camera_input("Capture Image")

    if file:
        st.image(file, use_column_width=True)

        if not IS_DEPLOY:
            try:
                from deepface import DeepFace
                import tempfile

                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(file.read())

                result = DeepFace.analyze(
                    img_path=tfile.name,
                    actions=['emotion'],
                    enforce_detection=False
                )
                emotion = result[0]['dominant_emotion']
            except:
                emotion = "Error"
        else:
            emotion = "Disabled (Cloud Limit)"

# -------- REAL-TIME --------
if mode == "Real-Time Webcam":

    if IS_DEPLOY:
        st.warning("⚠️ Webcam disabled in cloud deployment")
    else:
        from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
        import cv2
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
                    self.emotion = result[0]['dominant_emotion']
                except:
                    pass

                cv2.putText(img, f"{self.emotion}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                return img

        webrtc_ctx = webrtc_streamer(
            key="webcam",
            video_transformer_factory=EmotionDetector
        )

        if webrtc_ctx.video_transformer:
            emotion = webrtc_ctx.video_transformer.emotion

# -------- TEXT --------
text = st.text_area("💬 Enter your thoughts")

# -------- STRESS --------
def compute_stress(emotion, sentiment):
    score = 0

    if emotion in ["sad", "angry", "fear"]:
        score += 50

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