"""
Local-only stress detection script.
This is for testing locally only - NOT for Streamlit Cloud.

To run locally:
    python main.py

For Streamlit Cloud deployment, use:
    streamlit run app.py
"""

import cv2
import os
import sys
from textblob import TextBlob

# Function: Analyze text sentiment
def analyze_text(text):
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity

        if polarity > 0.1:
            return "Positive", polarity
        elif polarity < -0.1:
            return "Negative", abs(polarity)
        else:
            return "Neutral", 0
    except:
        return "Neutral", 0

# Function: Determine stress level
def detect_stress(emotion, sentiment):
    stress_score = 0

    if emotion in ["angry", "fear", "sad"]:
        stress_score += 2
    elif emotion == "detected":
        stress_score += 1
    elif emotion == "neutral":
        stress_score += 1

    if sentiment == "Negative":
        stress_score += 2
    elif sentiment == "Neutral":
        stress_score += 1

    if stress_score >= 3:
        return "High Stress"
    elif stress_score == 2:
        return "Moderate Stress"
    else:
        return "Low Stress"

def main():
    if __name__ != "__main__":
        return
    
    print("⚠️  This is a local-only script.")
    print("For web deployment, use: streamlit run app.py\n")
    
    # Start webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open webcam")
        return

    print("Press 'q' to capture emotion and proceed...")

    frame = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Camera - Press q to capture", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save captured frame
    cv2.imwrite("captured_face.jpg", frame)

    # Simple emotion detection from face
    print(f"\n🎭 Face captured! (Using lightweight analysis)")
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) > 0:
        emotion = "Detected"
    else:
        emotion = "Not detected"

    print(f"🎭 Detected Emotion Status: {emotion}")

    # Text input
    text = input("\n📝 Enter how you feel (text): ").strip()

    if not text:
        sentiment = "Neutral"
        polarity = 0
    else:
        sentiment, polarity = analyze_text(text)

    print(f"📊 Text Sentiment: {sentiment} (Polarity: {polarity:.2f})")

    # Stress prediction
    stress = detect_stress(emotion, sentiment)

    print(f"\n🧠 Final Stress Level: {stress}")
    
    # Cleanup
    try:
        os.remove("captured_face.jpg")
    except:
        pass

if __name__ == "__main__":
    main()
