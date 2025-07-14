import os
import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_mail import Mail, Message
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pathlib import Path
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
from serpapi_util import fetch_search_results

# Load .env
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# API Keys and Mail
serpapi_key = os.getenv('SERPAPI_KEY')
gmail_user = os.getenv("MAIL_USERNAME")

# Flask App Setup
app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"])  # <- Specific and safe

# Configure CORS to allow frontend origin with proper headers

@app.before_request
def log_request_info():
    print(f"Request method: {request.method}, Path: {request.path}")
    if request.method == 'OPTIONS':
        print("Handling preflight OPTIONS request")
        
# Mail Config
app.config.update(
    MAIL_SERVER=os.getenv("MAIL_SERVER"),
    MAIL_PORT=int(os.getenv("MAIL_PORT", 587)),
    MAIL_USERNAME=gmail_user,
    MAIL_PASSWORD=os.getenv("MAIL_PASSWORD"),
    MAIL_USE_TLS=os.getenv("MAIL_USE_TLS", "True") == "True",
    MAIL_USE_SSL=os.getenv("MAIL_USE_SSL", "False") == "True"
)
mail = Mail(app)
        
# MongoDB Setup
mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
client = MongoClient(mongo_uri)
db = client["medicalDB"]
users = db["users"]

# === 🔐 Auth & 2FA ===

def generate_code():
    return str(np.random.randint(100000, 999999))

def send_verification_email(email, code):
    try:
        msg = Message("Your Verification Code", sender=gmail_user, recipients=[email])
        msg.body = f"Your verification code is: {code}"
        mail.send(msg)
    except Exception as e:
        print(f"❌ Email sending error: {e}")

@app.route("/api/register", methods=["POST"])
def register():
    try:
        data = request.get_json(force=True)  # <-- 🔥 Add force=True
        email = data.get("email")
        password = data.get("password")

        if not email or not password:
            return jsonify({"message": "Missing email or password"}), 400

        if users.find_one({"email": email}):
            return jsonify({"message": "User already exists"}), 400

        hashed_password = generate_password_hash(password)
        users.insert_one({"email": email, "password": hashed_password})
        return jsonify({"message": "Registration successful"}), 201
    except Exception as e:
        print("Registration Error:", str(e))
        return jsonify({"message": "Registration failed", "error": str(e)}), 500


@app.route("/api/login-step1", methods=["POST"])
def login_step1():
    data = request.get_json()
    email, password = data.get("email"), data.get("password")
    user = users.find_one({"email": email})

    if not user or not check_password_hash(user["password"], password):
        return jsonify({"message": "Invalid credentials"}), 401

    code = generate_code()
    expiry = datetime.utcnow() + timedelta(minutes=5)
    users.update_one({"email": email}, {"$set": {
        "verification_code": code,
        "code_expiry": expiry
    }})

    send_verification_email(email, code)
    return jsonify({"message": "Verification code sent", "step": 2}), 200

@app.route("/api/login-step2", methods=["POST"])
def login_step2():
    data = request.get_json()
    email, code = data.get("email"), data.get("code")
    user = users.find_one({"email": email})

    if not user or user.get("verification_code") != code:
        return jsonify({"message": "Invalid verification code"}), 401

    if datetime.utcnow() > user.get("code_expiry"):
        return jsonify({"message": "Code expired"}), 401

    users.update_one({"email": email}, {"$unset": {"verification_code": "", "code_expiry": ""}})
    return jsonify({"message": "Login successful", "token": "dummy_token"}), 200

# === 🧠 Model + Prediction ===

vectorizer = None
scaler = None
models = {}

def load_models():
    global vectorizer, scaler, models
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        vectorizer = joblib.load(os.path.join(base_path, "models", "symptom_vectorizer.pkl"))
        scaler = joblib.load(os.path.join(base_path, "models", "medical_scaler.pkl"))
        models['logistic'] = joblib.load(os.path.join(base_path, "models", "best_medical_model_logistic_regression.pkl"))

        try:
            models['ensemble'] = joblib.load(os.path.join(base_path, "models", "ensemble_medical_model.pkl"))
            print("✅ Ensemble model loaded")
        except FileNotFoundError:
            print("ℹ️ Ensemble model not found, skipping...")

        print(f"✅ Vectorizer loaded with vocab size: {len(vectorizer.vocabulary_)}")
        return True
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return False

def preprocess_input(data):
    try:
        symptom_text = data.get("symptoms", "")
        bp = data.get("blood_pressure", "0/0")
        try:
            sys, dia = map(int, bp.split("/"))
            bp_avg = (sys + dia) / 2
        except:
            bp_avg = 0

        vitals = [
            bp_avg,
            float(data.get("heart_rate", 0)),
            float(data.get("age", 0)),
            float(data.get("temperature", 0)),
            float(data.get("oxygen_saturation", 0))
        ]

        symptom_vec = vectorizer.transform([symptom_text])
        vital_vec = scaler.transform([vitals])
        return np.hstack([symptom_vec.toarray(), vital_vec])
    except Exception as e:
        print(f"❌ Preprocessing error: {e}")
        return None

load_models()

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = preprocess_input(data)
        if features is None:
            return jsonify({"error": "Invalid input"}), 400

        predictions = {}
        if 'logistic' in models:
            pred = models['logistic'].predict(features)[0]
            prob = np.max(models['logistic'].predict_proba(features))
            predictions['logistic'] = {"prediction": pred, "confidence": float(prob)}

        if 'ensemble' in models:
            pred = models['ensemble'].predict(features)[0]
            prob = np.max(models['ensemble'].predict_proba(features))
            predictions['ensemble'] = {"prediction": pred, "confidence": float(prob)}

        return jsonify({"result": predictions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/treatment", methods=["POST"])
def generate_treatment():
    try:
        data = request.get_json()
        query = (
            f"{data['disease']} treatment for a {data['age']}-year-old patient "
            f"with symptoms {data['symptoms']}, blood group {data['bloodGroup']}, duration {data['duration']}"
        )
        results = fetch_search_results(query)
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=8000)
