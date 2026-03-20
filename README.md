# 🧠 Stress Detection AI System

A multi-modal AI application that detects **stress levels** using:
- 📸 Face detection + emotion estimation
- ✍️ Text sentiment analysis
- 🎥 Real-time webcam emotion tracking

---

## 🚀 Live Demo

👉 **Try the App Here:**  
🔗 https://huggingface.co/spaces/chandana987/stress-detection-ai

---

## 🧩 Features

- ✅ Upload Image + Analyze Emotion
- ✅ Capture Image via Camera
- ✅ Real-time Webcam Emotion Detection
- ✅ Text Sentiment Analysis (NLP)
- ✅ Combined Stress Score Calculation
- ✅ Face vs Object Differentiation
- ✅ Lightweight & Deployment-Friendly (No heavy ML models)

---

## 🛠️ Tech Stack

- Python  
- OpenCV (Face Detection)  
- Gradio (UI & Deployment)  
- TextBlob (Sentiment Analysis)  
- NumPy  

---

## 📊 How It Works

### 1. Face Detection
- Uses Haar Cascade Classifier
- Filters out non-human objects

### 2. Emotion Detection
- Based on facial intensity patterns
- Classifies into:
  - 😊 Happy  
  - 😞 Sad  
  - 😐 Neutral  
  - 😠 Angry  

### 3. Text Analysis
- Uses sentiment polarity
- Categorizes into:
  - Positive  
  - Neutral  
  - Negative  

### 4. Stress Score Calculation

Final Score =  
`0.6 × Face Emotion + 0.4 × Text Sentiment`

---

## 📷 Input Modes

| Mode            | Description |
|-----------------|------------|
| Upload Image    | Upload JPG/PNG |
| Camera Capture  | Take photo |
| Live Webcam     | Real-time detection |

---

## ⚠️ Limitations

- Emotion detection is heuristic (not deep learning)
- Accuracy depends on lighting & face clarity
- No GPU / heavy AI models used (optimized for free deployment)

---

## 🧪 Future Improvements

- Integrate ONNX-based emotion model
- Add voice-based stress detection
- Improve accuracy using deep learning

---

## 👩‍💻 Author

**Chandana**  
B.Tech Student  

---

## 📌 Deployment

Hosted on Hugging Face Spaces using Gradio

---

## ⭐ If you like this project, give it a star!
