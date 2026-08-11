import os
import json
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from model import SpamClassifier

frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))
app = Flask(__name__, static_folder=frontend_dir, static_url_path='')

# Enable CORS for all routes and origins
CORS(app, resources={r"/*": {"origins": "*"}})

# Initialize Spam Classifier instance
classifier = None

def get_classifier():
    global classifier
    if classifier is None:
        classifier = SpamClassifier()
    return classifier


@app.route('/')
def serve_index():
    return send_from_directory(frontend_dir, 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    if os.path.exists(os.path.join(frontend_dir, path)):
        return send_from_directory(frontend_dir, path)
    return jsonify({"error": "File not found"}), 404



@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    """
    try:
        clf = get_classifier()
        is_ready = clf.model is not None and clf.vectorizer is not None
        return jsonify({
            "status": "API is running",
            "model_loaded": is_ready,
            "version": "1.0.0"
        }), 200
    except Exception as e:
        return jsonify({
            "status": "API is degraded",
            "error": str(e),
            "model_loaded": False
        }), 500


@app.route('/predict', methods=['POST'])
def predict_sms():
    """
    Single SMS prediction endpoint
    """
    try:
        if not request.is_json:
            return jsonify({
                "error": "Invalid Content-Type. Expected application/json."
            }), 400

        data = request.get_json(silent=True)
        if not data or 'message' not in data:
            return jsonify({
                "error": "Missing 'message' field in JSON payload."
            }), 400

        raw_message = data.get('message', '')

        if not isinstance(raw_message, str):
            return jsonify({
                "error": "'message' field must be a string."
            }), 400

        raw_message = raw_message.strip()
        if len(raw_message) == 0:
            return jsonify({
                "error": "SMS message content cannot be empty."
            }), 400

        # Safety length limit
        if len(raw_message) > 5000:
            return jsonify({
                "error": "SMS message exceeds maximum length limit of 5000 characters."
            }), 400

        clf = get_classifier()
        result = clf.predict(raw_message)

        # Standard requested output format
        response = {
          "prediction": result["prediction"],
          "confidence": result["confidence"],
          "message": raw_message,
          "is_spam": result["is_spam"],
          "spam_probability": result["spam_probability"],
          "ham_probability": result["ham_probability"],
          "cleaned_message": result["cleaned_message"],
          "explanation": result["explanation"]
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({
            "error": f"Internal Prediction Error: {str(e)}"
        }), 500


@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """
    Batch SMS prediction endpoint accepting CSV file or array of messages.
    """
    try:
        messages = []
        
        # Check if CSV file uploaded
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No selected CSV file."}), 400
            
            try:
                df = pd.read_csv(file)
                # Pick first string column or column named 'message' or 'text'
                text_col = None
                for col in ['message', 'text', 'sms', 'v2']:
                    if col in df.columns:
                        text_col = col
                        break
                if text_col is None:
                    text_col = df.select_dtypes(include=['object']).columns[0]
                
                messages = df[text_col].dropna().astype(str).tolist()
            except Exception as ex:
                return jsonify({"error": f"Failed to parse CSV file: {str(ex)}"}), 400

        elif request.is_json:
            data = request.get_json(silent=True)
            if data and 'messages' in data and isinstance(data['messages'], list):
                messages = [str(m).strip() for m in data['messages'] if str(m).strip()]

        if not messages:
            return jsonify({
                "error": "No valid messages found in payload. Provide a JSON body with 'messages' list or upload a CSV file."
            }), 400

        # Cap batch size to 500 items per request
        if len(messages) > 500:
            messages = messages[:500]

        clf = get_classifier()
        results = clf.predict_batch(messages)

        spam_count = sum(1 for r in results if r.get('is_spam') is True)
        ham_count = sum(1 for r in results if r.get('is_spam') is False)

        return jsonify({
            "total": len(results),
            "spam_count": spam_count,
            "ham_count": ham_count,
            "results": results
        }), 200

    except Exception as e:
        return jsonify({"error": f"Batch prediction failed: {str(e)}"}), 500


@app.route('/metrics', methods=['GET'])
def get_metrics():
    """
    Returns model metrics, confusion matrix, comparison data, and word frequencies.
    """
    try:
        clf = get_classifier()
        metrics = clf.get_metrics()
        return jsonify(metrics), 200
    except Exception as e:
        return jsonify({"error": f"Failed to retrieve metrics: {str(e)}"}), 500


if __name__ == '__main__':
    # Determine port from environment or default to 5000
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Spam SMS Detection Flask Server on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)
