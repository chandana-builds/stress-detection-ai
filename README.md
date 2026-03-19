# 🧠 Stress Detection AI

A web application that detects stress levels using face recognition and NLP sentiment analysis.

## Features

- **Face Detection**: Detects human faces using OpenCV Haar Cascades
- **Emotion Recognition**: Analyzes facial emotions (local only)
- **Text Sentiment Analysis**: Uses TextBlob for NLP sentiment analysis
- **Stress Scoring**: Combines emotion and sentiment for stress level prediction
- **Web Interface**: Built with Streamlit for easy deployment

## Local Development

### Requirements
- Python 3.8+
- pip or conda

### Installation

1. Clone the repository:
```bash
git clone <your-repo>
cd project-1
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running Locally

**Web App (Recommended):**
```bash
streamlit run app.py
```

**CLI Version (Local Only):**
```bash
python main.py
```

## Streamlit Cloud Deployment

### Setup Instructions

1. **Push to GitHub**:
   - Create a GitHub repository
   - Push your code: `git push origin main`

2. **Deploy on Streamlit Cloud**:
   - Go to [https://share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your GitHub repo, branch, and main file: `app.py`
   - Click "Deploy"

### Important Notes

- ✅ **Upload & Camera modes work** on Streamlit Cloud
- ❌ **Webcam mode** requires `streamlit-webrtc` (optional, not installed by default)
- ⚠️ **First load may take 2-3 minutes** due to model downloads
- 📊 **Full emotion detection** only works locally (cloud uses basic detection)

## File Structure

```
project-1/
├── app.py                    # Main Streamlit web app
├── main.py                   # Local CLI version (local-only)
├── requirements.txt          # Python dependencies
├── .streamlit/
│   └── config.toml          # Streamlit configuration
├── .gitignore               # Git ignore rules
└── README.md                # This file
```

## Troubleshooting

### Issue: Deployment fails with timeout
**Solution**: The app now uses caching and cloud has limited resources. Wait 3-5 minutes on first load.

### Issue: "No module named 'deepface'"
**Solution**: Dependencies are already in requirements.txt. Redeploy or reinstall locally.

### Issue: Camera upload works but no emotion detected in cloud
**Solution**: This is normal. Cloud has limited resources. Use local deployment for full emotion analysis.

### Issue: "No human face detected"
**Solutions**:
- Ensure good lighting
- Face should be clearly visible
- Try different angles
- Use higher resolution images

## Model Information

- **Face Detection**: Haar Cascade Classifier (OpenCV)
- **Emotion Detection**: DeepFace (local only)
- **Sentiment Analysis**: TextBlob

## Stress Score Interpretation

- 🟢 **0-35%**: Low Stress - You're doing great!
- 🟡 **35-70%**: Moderate Stress - Try relaxation techniques
- 🔴 **70-100%**: High Stress - Consider seeking support

## Performance

- **Local**: Full analysis in 2-5 seconds
- **Cloud** (first load): 2-3 minutes (model download)
- **Cloud** (subsequent loads): 5-10 seconds

## License

MIT License

## Support

For issues, create a GitHub issue or contact the developer.
