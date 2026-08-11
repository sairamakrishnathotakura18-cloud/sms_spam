import re
import os
import joblib
import json
import numpy as np

# NLTK imports with safe fallback
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Stopwords set & Stemmer instance
try:
    STOP_WORDS = set(stopwords.words('english'))
except Exception:
    STOP_WORDS = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", 
        "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", 
        "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", 
        "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", 
        "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", 
        "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", 
        "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", 
        "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", 
        "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", 
        "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", 
        "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", 
        "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", 
        "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", 
        "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", 
        "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
    }

stemmer = PorterStemmer()

def clean_text(text):
    """
    Clean raw SMS message:
    1. Lowercase text
    2. Remove URLs, email addresses, HTML tags
    3. Keep alphabetic characters & whitespace
    4. Tokenize & remove stopwords
    5. Apply Porter Stemmer
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove Email addresses
    text = re.sub(r'\S+@\S+', '', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove non-alphabet characters
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Tokenize
    words = text.split()
    
    # Filter stopwords and stem
    cleaned_words = [stemmer.stem(w) for w in words if w not in STOP_WORDS and len(w) > 1]
    
    return " ".join(cleaned_words)


class SpamClassifier:
    def __init__(self, models_dir=None):
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(__file__), 'models')
        self.models_dir = models_dir
        self.model = None
        self.vectorizer = None
        self.metrics = None
        self.load_models()

    def load_models(self):
        model_path = os.path.join(self.models_dir, 'spam_model.pkl')
        vec_path = os.path.join(self.models_dir, 'tfidf_vectorizer.pkl')
        metrics_path = os.path.join(self.models_dir, 'model_metrics.json')

        if os.path.exists(model_path) and os.path.exists(vec_path):
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vec_path)
        
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                self.metrics = json.load(f)

    def predict(self, raw_message):
        """
        Predict whether a single SMS is Spam or Ham (Not Spam),
        return prediction, confidence score, cleaned text, and influential word features.
        """
        if self.model is None or self.vectorizer is None:
            # Fallback if models are not yet trained
            self.load_models()
            if self.model is None or self.vectorizer is None:
                raise RuntimeError("Machine Learning model files not found. Please run train_model.py first.")

        if not raw_message or not raw_message.strip():
            raise ValueError("Message content cannot be empty.")

        cleaned = clean_text(raw_message)
        
        # Transform using vectorizer
        features = self.vectorizer.transform([cleaned])
        
        # Predict probability
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(features)[0]
            # probs[0] is ham (0), probs[1] is spam (1)
            spam_prob = float(probs[1])
            ham_prob = float(probs[0])
        else:
            # Fallback for models without predict_proba (e.g., decision function)
            decision = self.model.decision_function(features)[0]
            spam_prob = 1.0 / (1.0 + np.exp(-decision))
            ham_prob = 1.0 - spam_prob

        is_spam = spam_prob >= 0.5
        prediction = "Spam" if is_spam else "Not Spam (Ham)"
        confidence = float(spam_prob if is_spam else ham_prob) * 100.0
        confidence = round(confidence, 2)

        # Feature level explanation (influential words)
        explanation = self._explain_prediction(features, cleaned, is_spam)

        return {
            "prediction": prediction,
            "is_spam": bool(is_spam),
            "confidence": confidence,
            "spam_probability": round(float(spam_prob) * 100, 2),
            "ham_probability": round(float(ham_prob) * 100, 2),
            "cleaned_message": cleaned,
            "raw_message": raw_message,
            "explanation": explanation
        }

    def _explain_prediction(self, features_sparse, cleaned_text, is_spam):
        """
        Identifies top words in the input message that contributed to the Spam vs Ham decision.
        """
        if self.vectorizer is None or self.model is None:
            return []

        feature_names = self.vectorizer.get_feature_names_out()
        nonzero_indices = features_sparse.nonzero()[1]
        
        if len(nonzero_indices) == 0:
            return []

        word_scores = []
        
        # Check model coefficients or log probabilities
        if hasattr(self.model, "feature_log_prob_"):
            # Naive Bayes model
            log_prob_spam = self.model.feature_log_prob_[1]
            log_prob_ham = self.model.feature_log_prob_[0]
            diff = log_prob_spam - log_prob_ham
            
            for idx in nonzero_indices:
                word = feature_names[idx]
                tfidf_val = features_sparse[0, idx]
                impact = diff[idx] * tfidf_val
                category = "spam" if impact > 0 else "ham"
                word_scores.append({
                    "word": word,
                    "score": round(float(abs(impact)), 4),
                    "category": category,
                    "impact_direction": "Spam Indicator" if impact > 0 else "Safe Indicator"
                })
        elif hasattr(self.model, "coef_"):
            # Logistic Regression / Linear Model
            coefs = self.model.coef_[0]
            for idx in nonzero_indices:
                word = feature_names[idx]
                tfidf_val = features_sparse[0, idx]
                impact = coefs[idx] * tfidf_val
                category = "spam" if impact > 0 else "ham"
                word_scores.append({
                    "word": word,
                    "score": round(float(abs(impact)), 4),
                    "category": category,
                    "impact_direction": "Spam Indicator" if impact > 0 else "Safe Indicator"
                })
        else:
            for idx in nonzero_indices:
                word = feature_names[idx]
                word_scores.append({
                    "word": word,
                    "score": 1.0,
                    "category": "spam" if is_spam else "ham",
                    "impact_direction": "Keyword"
                })

        # Sort by score descending
        word_scores.sort(key=lambda x: x["score"], reverse=True)
        return word_scores[:10]

    def predict_batch(self, messages):
        """
        Batch prediction for a list of SMS messages.
        """
        results = []
        for msg in messages:
            try:
                pred = self.predict(msg)
                results.append(pred)
            except Exception as e:
                results.append({
                    "raw_message": msg,
                    "error": str(e),
                    "prediction": "Error",
                    "confidence": 0.0
                })
        return results

    def get_metrics(self):
        if self.metrics is None:
            self.load_models()
        return self.metrics or {}
