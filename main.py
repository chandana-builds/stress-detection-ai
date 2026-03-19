import cv2
from deepface import DeepFace
from textblob import TextBlob

# Function: Analyze text sentiment
def analyze_text(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    if polarity > 0:
        return "Positive", polarity
    elif polarity < 0:
        return "Negative", polarity
    else:
        return "Neutral", polarity

# Function: Determine stress level
def detect_stress(emotion, sentiment):
    if emotion in ["sad", "angry", "fear"] and sentiment == "Negative":
        return "High Stress"
    elif emotion in ["happy", "surprise"] and sentiment == "Positive":
        return "Low Stress"
    else:
        return "Moderate Stress"

# Start webcam
cap = cv2.VideoCapture(0)

print("Press 'q' to capture emotion and proceed...")

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

# Emotion detection using DeepFace
try:
    result = DeepFace.analyze(img_path="captured_face.jpg", actions=['emotion'])
    emotion = result[0]['dominant_emotion']
except:
    emotion = "neutral"

print(f"\nDetected Emotion: {emotion}")

# Text input
text = input("\nEnter how you feel (text): ")

sentiment, polarity = analyze_text(text)

print(f"Text Sentiment: {sentiment} (Polarity: {polarity})")

# Stress prediction
stress = detect_stress(emotion, sentiment)

print(f"\n🧠 Final Stress Level: {stress}")