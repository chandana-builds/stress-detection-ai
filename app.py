import streamlit as st
import cv2
import numpy as np
from textblob import TextBlob

st.set_page_config(page_title="AI Stress Detection", layout="centered")

st.title("🧠 Emotion & Stress Detection System")

# ---------------- FACE DETECTOR ----------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=6, minSize=(80, 80)
    )
    return faces

# ---------------- EMOTION ----------------
def detect_emotion(face):
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)

    if brightness > 140:
        return "Happy 😊"
    elif brightness < 90:
        return "Sad 😢"
    return "Neutral 😐"

# ---------------- TEXT ----------------
def analyze_text(text):
    polarity = TextBlob(text).sentiment.polarity

    if polarity < -0.3:
        return "Negative", abs(polarity)
    elif polarity > 0.3:
        return "Positive", polarity
    return "Neutral", 0

def stress_level(sentiment):
    if sentiment == "Negative":
        return 80
    elif sentiment == "Neutral":
        return 50
    return 20


# ---------------- MODE ----------------
mode = st.radio("Choose Input", ["Upload", "Camera", "Live Webcam"])

emotion = "Not detected"

# ======================================================
# 📁 UPLOAD
# ======================================================
if mode == "Upload":
    file = st.file_uploader("Upload Image", ["jpg","png","jpeg"])

    if file:
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)
        st.image(img, channels="BGR")

        faces = detect_faces(img)

        if len(faces) == 0:
            st.error("❌ No face detected")
        else:
            (x,y,w,h) = faces[0]
            face = img[y:y+h, x:x+w]

            emotion = detect_emotion(face)

            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            st.image(img, channels="BGR")

            st.success(f"Emotion: {emotion}")

# ======================================================
# 📸 CAMERA (FIXED)
# ======================================================
elif mode == "Camera":
    img_file = st.camera_input("Take Photo")

    if img_file:
        img = cv2.imdecode(np.frombuffer(img_file.read(), np.uint8), 1)

        st.image(img, channels="BGR")

        faces = detect_faces(img)

        if len(faces) == 0:
            st.error("❌ No face detected")
        else:
            (x,y,w,h) = faces[0]
            face = img[y:y+h, x:x+w]

            emotion = detect_emotion(face)
            st.success(f"Emotion: {emotion}")

# ======================================================
# 🎥 LIVE WEBCAM (REAL FIX)
# ======================================================
elif mode == "Live Webcam":
    from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
    import av

    class EmotionDetector(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")

            faces = detect_faces(img)

            for (x,y,w,h) in faces:
                face = img[y:y+h, x:x+w]
                emotion = detect_emotion(face)

                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.putText(img, emotion, (x,y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

            return img

    webrtc_streamer(key="webcam", video_transformer_factory=EmotionDetector)


# ======================================================
# 💬 TEXT
# ======================================================
st.subheader("💬 Your Thoughts")
text = st.text_area("Type here")

if st.button("Analyze"):
    if not text:
        st.warning("Enter text")
    else:
        sentiment, conf = analyze_text(text)
        stress = stress_level(sentiment)

        st.write("Emotion:", emotion)
        st.write("Sentiment:", sentiment)
        st.write("Stress:", stress)

        st.progress(stress)