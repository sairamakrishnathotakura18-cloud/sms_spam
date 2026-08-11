# 🛡️ SpamGuard AI — Spam SMS Filtering Using Machine Learning

An intelligent, full-stack Machine Learning SMS Spam Detection web application. **SpamGuard AI** processes raw text SMS messages, extracts TF-IDF n-gram feature vectors, cleans tokens using Natural Language Processing (NLP), and accurately predicts whether a message is **Spam** or **Not Spam (Ham)** alongside confidence percentages and feature-level explanations.

---

## 🌟 Key Features

- **Real Machine Learning Model**: Trained on 5,563 real SMS messages using Scikit-Learn.
- **Multi-Model Benchmark**: Trains and compares **Multinomial Naive Bayes**, **Logistic Regression**, and **Support Vector Machine (SVM)**.
- **Explain Prediction (Feature Attribution)**: Highlighting key terms in the input SMS that contributed to the classification.
- **Batch CSV Classifier**: Upload CSV files of SMS messages, view live table classifications, and download prediction results as a CSV export.
- **Model Metrics & Confusion Matrix**: Real-time evaluation display of Accuracy (97.93%), Precision (98.46%), Recall (85.91%), F1-Score (91.76%), and a 2x2 confusion matrix grid.
- **Interactive REST API**: Flask REST backend providing `/health`, `/predict`, `/batch-predict`, and `/metrics`.
- **Modern Responsive Glassmorphism UI**: High-tech dark modern aesthetic with light mode toggle, circular progress gauge, Chart.js analytics, copyable API code snippets, and localStorage prediction history.

---

## 📁 Project Structure

```
spam-sms-detector/
│
├── frontend/
│   ├── index.html         # Main dashboard HTML interface
│   ├── style.css          # Glassmorphism, animations, & theme system
│   └── script.js          # Fetch API, Chart.js, batch CSV & localStorage logic
│
├── backend/
│   ├── app.py             # Flask REST API & static server
│   ├── model.py           # NLP text cleaner & ML inference engine with feature attribution
│   ├── train_model.py     # ML training pipeline script (NB, LR, SVM)
│   ├── requirements.txt   # Python backend dependencies
│   └── models/
│       ├── spam_model.pkl         # Primary trained model artifact
│       ├── tfidf_vectorizer.pkl   # Fitted TF-IDF vectorizer artifact
│       ├── nb_model.pkl           # Trained Naive Bayes model
│       ├── lr_model.pkl           # Trained Logistic Regression model
│       ├── svm_model.pkl          # Trained Support Vector Machine model
│       └── model_metrics.json     # Saved evaluation metrics & dataset statistics
│
├── dataset/
│   └── spam.csv           # SMS Spam Collection Dataset (5,572 rows)
│
├── README.md              # Documentation and instructions
└── .gitignore             # Ignored virtualenv & cache files
```

---

## ⚙️ Installation & Setup

### Prerequisites
- **Python 3.8+**
- **pip** package manager

### 1. Clone or Open Project Directory
```bash
cd /home/sai/seminar/spam-sms-detector
```

### 2. Create Virtual Environment & Install Dependencies
```bash
# Create python virtual environment
python3 -m venv venv

# Activate virtual environment (Linux/MacOS)
source venv/bin/activate

# Install requirements
pip install -r backend/requirements.txt
```

---

## 🤖 Model Training

The ML pipeline preprocesses text messages (lowercasing, URL/HTML removal, punctuation filtering, tokenization, stopword removal, and Porter Stemming), fits a TF-IDF Vectorizer with n-gram range `(1, 2)`, trains Naive Bayes, Logistic Regression, and SVM classifiers, and exports model artifacts.

To re-train the models:
```bash
python backend/train_model.py
```

### Model Performance Benchmarks:
| Model Algorithm | Accuracy | Precision | Recall | F1 Score | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes** | **97.93%** | **98.46%** | **85.91%** | **91.76%** | Primary Model |
| **Support Vector Machine (SVM)** | 98.20% | 97.78% | 88.59% | 92.96% | High Precision |
| **Logistic Regression** | 96.86% | 97.50% | 78.52% | 86.99% | Evaluated |

---

## 🚀 Running the Web Application

Start the combined Flask server (Backend API + Frontend Static Server):

```bash
python backend/app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 🔌 API Documentation

### 1. Health Check
- **GET** `/health`
- **Response**:
```json
{
  "model_loaded": true,
  "status": "API is running",
  "version": "1.0.0"
}
```

### 2. Predict Single Message
- **POST** `/predict`
- **Headers**: `Content-Type: application/json`
- **Request Body**:
```json
{
  "message": "Congratulations! You won a $1000 prize. Click here now!"
}
```
- **Response Body**:
```json
{
  "prediction": "Spam",
  "confidence": 97.13,
  "message": "Congratulations! You won a $1000 prize. Click here now!",
  "is_spam": true,
  "spam_probability": 97.13,
  "ham_probability": 2.87,
  "cleaned_message": "congratul prize click",
  "explanation": [
    { "word": "prize", "score": 2.512, "impact_direction": "Spam Indicator", "category": "spam" },
    { "word": "congratul", "score": 1.771, "impact_direction": "Spam Indicator", "category": "spam" },
    { "word": "click", "score": 1.099, "impact_direction": "Spam Indicator", "category": "spam" }
  ]
}
```

### 3. Model Metrics
- **GET** `/metrics`
- **Response**: Returns dataset statistics, primary model metrics, confusion matrix, model comparison metrics, and word frequencies.

---

## 🧪 Sample Messages for Testing

### 🚨 Spam Test Samples:
1. `WINNER! You have won a $1000 Walmart gift card. Claim now at http://win-now.com or call 0800-123-456`
2. `URGENT! Your bank account has been locked due to suspicious activity. Verify credentials at http://secure-bank-login.net immediately.`
3. `FreeMsg: Claim your £1000 cash prize now! Text CLAIM to 87077. Terms apply.`

### ✅ Safe (Ham) Test Samples:
1. `Hey! Are we still on for dinner tonight at 7 PM? Let me know if you want me to pick up anything.`
2. `Hi Team, the project documentation has been updated. Please review the PR before tomorrow's standup.`
3. `I am on my way home. Can you please buy some milk from the grocery store?`
