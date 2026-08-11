import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from model import clean_text

def train():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, '..', 'dataset', 'spam.csv')
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    print(f"1. Loading dataset from {dataset_path}...")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Please check dataset setup.")

    df = pd.read_csv(dataset_path)
    
    # Standardize column names if needed
    if 'label' not in df.columns or 'message' not in df.columns:
        # Check if columns are v1, v2 (typical Kaggle format)
        if 'v1' in df.columns and 'v2' in df.columns:
            df = df.rename(columns={'v1': 'label', 'v2': 'message'})
        else:
            df.columns = ['label', 'message'] + list(df.columns[2:])

    # Clean missing values
    df = df.dropna(subset=['label', 'message'])
    
    # Map labels ham -> 0, spam -> 1
    df['target'] = df['label'].map({'ham': 0, 'spam': 1})
    df = df.dropna(subset=['target'])
    df['target'] = df['target'].astype(int)

    print(f"Dataset loaded: {len(df)} total messages ({sum(df['target']==0)} Ham, {sum(df['target']==1)} Spam)")

    print("2. Preprocessing & Cleaning text messages...")
    df['cleaned_message'] = df['message'].apply(clean_text)

    # Filter out any messages that became empty after cleaning
    valid_mask = df['cleaned_message'].str.len() > 0
    df = df[valid_mask]

    X = df['cleaned_message']
    y = df['target']

    print("3. Train / Test Split (80% Train, 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print("4. Feature Extraction using TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("5. Training Machine Learning Models...")

    classifiers = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.2),
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        "Support Vector Machine": SVC(kernel='linear', probability=True, random_state=42)
    }

    results = {}
    saved_models = {}

    for name, clf in classifiers.items():
        print(f"   Training {name}...")
        clf.fit(X_train_tfidf, y_train)
        y_pred = clf.predict(X_test_tfidf)

        acc = round(float(accuracy_score(y_test, y_pred)) * 100, 2)
        prec = round(float(precision_score(y_test, y_pred, zero_division=0)) * 100, 2)
        rec = round(float(recall_score(y_test, y_pred, zero_division=0)) * 100, 2)
        f1 = round(float(f1_score(y_test, y_pred, zero_division=0)) * 100, 2)
        cm = confusion_matrix(y_test, y_pred).tolist()

        results[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "confusion_matrix": cm  # [[TN, FP], [FN, TP]]
        }

        # Save model pkl
        filename_prefix = name.lower().replace(" ", "_")
        model_filename = os.path.join(models_dir, f"{filename_prefix}.pkl")
        joblib.dump(clf, model_filename)
        saved_models[name] = clf

        print(f"   -> {name}: Accuracy={acc}%, Precision={prec}%, Recall={rec}%, F1={f1}%")

    # Select Primary Model (Multinomial Naive Bayes or best F1)
    primary_name = "Multinomial Naive Bayes"
    primary_model = saved_models[primary_name]
    
    primary_model_path = os.path.join(models_dir, 'spam_model.pkl')
    vec_model_path = os.path.join(models_dir, 'tfidf_vectorizer.pkl')

    joblib.dump(primary_model, primary_model_path)
    joblib.dump(vectorizer, vec_model_path)

    print("6. Extracting Word Frequencies for Analytics...")
    # Word frequency analysis for spam vs ham
    spam_tfidf = vectorizer.transform(df[df['target'] == 1]['cleaned_message'])
    ham_tfidf = vectorizer.transform(df[df['target'] == 0]['cleaned_message'])
    feature_names = vectorizer.get_feature_names_out()

    spam_scores = np.asarray(spam_tfidf.sum(axis=0)).ravel()
    ham_scores = np.asarray(ham_tfidf.sum(axis=0)).ravel()

    top_spam_indices = spam_scores.argsort()[-20:][::-1]
    top_ham_indices = ham_scores.argsort()[-20:][::-1]

    top_spam_words = [{"word": feature_names[i], "score": round(float(spam_scores[i]), 2)} for i in top_spam_indices]
    top_ham_words = [{"word": feature_names[i], "score": round(float(ham_scores[i]), 2)} for i in top_ham_indices]

    metrics_payload = {
        "primary_model": primary_name,
        "primary_metrics": results[primary_name],
        "comparison": results,
        "dataset_stats": {
            "total_messages": int(len(df)),
            "spam_messages": int(sum(df['target'] == 1)),
            "ham_messages": int(sum(df['target'] == 0)),
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test))
        },
        "top_words": {
            "spam": top_spam_words,
            "ham": top_ham_words
        }
    }

    metrics_json_path = os.path.join(models_dir, 'model_metrics.json')
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"7. Successfully saved all trained models and metrics to {models_dir}!")
    print("Done training ML Pipeline.")

if __name__ == '__main__':
    train()
