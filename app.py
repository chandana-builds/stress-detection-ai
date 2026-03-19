import streamlit as st
import cv2
import tempfile
from textblob import TextBlob
import os
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Stress AI Pro", layout="centered")
st.title("🧠 Stress Detection System")
st.caption("Face + Text based Stress Analysis")

IS_DEPLOY = os.getenv("RENDER") == "true"

# -------- FACE DETECTION --------
@st.cache_resource
def load_cascade():
    try:
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        return cascade
    except Exception as e:
        st.error(f"Error loading face cascade: {e}")
        return None

face_cascade = load_cascade()

def detect_face(path):
    if face_cascade is None:
        return False
    
    try:
        img = cv2.imread(path)
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

# -------- TEXT ANALYSIS --------
def analyze_text(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    if polarity < 0:
        return "Negative", abs(polarity)
    elif polarity > 0:
        return "Positive", polarity
    else:
        return "Neutral", 0

# -------- INPUT --------
mode = st.radio("Input Method", ["Upload", "Camera", "Webcam"])
emotion = "Not detected"

# -------- IMAGE --------
if mode in ["Upload", "Camera"]:
    file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"]) if mode=="Upload" else st.camera_input("Capture")

    if file:
        st.image(file)

        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        temp.write(file.read())
        temp.close()

        if not detect_face(temp.name):
            st.error("❌ No human face detected")
            emotion = "No Face"
        else:
            try:
                from deepface import DeepFace

                result = DeepFace.analyze(
                    img_path=temp.name,
                    actions=['emotion'],
                    enforce_detection=True
                )
                emotion = result[0]['dominant_emotion']
            except Exception:
                emotion = "Face Detected"
        
        # Cleanup temp file
        try:
            os.unlink(temp.name)
        except Exception:
            pass

# -------- WEBCAM --------
elif mode == "Webcam":
    try:
        from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
        from deepface import DeepFace

        class Detector(VideoTransformerBase):
            def __init__(self):
                self.emotion = "Detecting"

            def transform(self, frame):
                img = frame.to_ndarray(format="bgr24")
                try:
                    result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
                    self.emotion = result[0]['dominant_emotion']
                except:
                    pass
                return img

        ctx = webrtc_streamer(key="cam", video_transformer_factory=Detector)

        if ctx.video_transformer:
            emotion = ctx.video_transformer.emotion

    except Exception:
        st.warning("Webcam not supported")

# -------- TEXT --------
text = st.text_area("Enter your thoughts")

# -------- STRESS --------
def calc_stress(emotion, sentiment):
    score = 0

    if emotion in ["sad","angry","fear"]:
        score += 50
    elif emotion == "neutral":
        score += 20

    if sentiment == "Negative":
        score += 40

    return min(score,100)

# -------- ANALYZE --------
if st.button("Analyze"):
    if text:
        sentiment, conf = analyze_text(text)
        stress = calc_stress(emotion, sentiment)

        st.success("Analysis Done")

        st.write("Emotion:", emotion)
        st.write("Sentiment:", sentiment)
        st.write("Stress:", stress)

        st.progress(stress / 100)
    else:
        st.warning("Enter text")
