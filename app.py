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
import gdown

# Load environment variables
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# Flask setup
app = Flask(__name__)
CORS(app, origins=[
    "http://localhost:5173",
    "https://dynamic-sunburst-5f73a6.netlify.app"
])

@app.before_request
def log_request_info():
    print(f"Request method: {request.method}, Path: {request.path}")
    if request.method == 'OPTIONS':
        print("Handling preflight OPTIONS request")

# Mail configuration
gmail_user = os.getenv("MAIL_USERNAME")
app.config.update(
    MAIL_SERVER=os.getenv("MAIL_SERVER"),
    MAIL_PORT=int(os.getenv("MAIL_PORT", 587)),
    MAIL_USERNAME=gmail_user,
    MAIL_PASSWORD=os.getenv("MAIL_PASSWORD"),
    MAIL_USE_TLS=os.getenv("MAIL_USE_TLS", "True") == "True",
    MAIL_USE_SSL=os.getenv("MAIL_USE_SSL", "False") == "True"
)
mail = Mail(app)

# MongoDB configuration
mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
client = MongoClient(mongo_uri)
db = client["medicalDB"]
users = db["users"]

# === Auth ===
def generate_code():
    return str(np.random.randint(100000, 999999))

def send_verification_email(email, code):
    try:
        msg = Message("Your Verification Code", sender=gmail_user, recipients=[email])
        msg.body = f"Your verification code is: {code}"
        mail.send(msg)
        print(f"✅ Email sent to {email} with code {code}")
    except Exception as e:
        print(f"❌ Email sending error: {e}")

@app.route("/api/register", methods=["POST"])
def register():
    try:
        data = request.get_json(force=True)
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
    try:
        data = request.get_json(force=True)
        print("Received login data:", data)

        email = data.get("email")
        password = data.get("password")

        if not email or not password:
            return jsonify({"message": "Email and password required"}), 400

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

    except Exception as e:
        print("❌ Login Step 1 Error:", str(e))
        return jsonify({"message": "Login failed", "error": str(e)}), 500

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

# === ML Models ===

def download_model_if_missing(file_id, output_path):
    if not os.path.exists(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"⬇️ Downloading {output_path}...")
        gdown.download(url, output_path, quiet=False)
        print(f"✅ Downloaded {output_path}")

def load_models():
    base = os.path.join(os.path.dirname(__file__), "models")

    # Google Drive file IDs
    logistic_id = "1JBOrSJtfZL7kKeOqOedi1e3cYMpSt2rd"
    ensemble_id = "15J6ieS97efmxGySE6c_9yMEbwZdMxpII"
    scaler_id = "1aabdJ-DvGawM5vI-B9m69IuIqeNL5Re_"
    vectorizer_id = "1lyr6Qnx3Wqr-fsWqaBQoTKra629Ac91x"

    # Paths
    logistic_path = os.path.join(base, "best_medical_model_logistic_regression.pkl")
    ensemble_path = os.path.join(base, "ensemble_medical_model.pkl")
    scaler_path = os.path.join(base, "medical_scaler.pkl")
    vectorizer_path = os.path.join(base, "symptom_vectorizer.pkl")

    # Download if missing
    download_model_if_missing(logistic_id, logistic_path)
    download_model_if_missing(ensemble_id, ensemble_path)
    download_model_if_missing(scaler_id, scaler_path)
    download_model_if_missing(vectorizer_id, vectorizer_path)

    # Load
    vectorizer = joblib.load(vectorizer_path)
    scaler = joblib.load(scaler_path)
    logistic_model = joblib.load(logistic_path)

    ensemble_model = None
    try:
        ensemble_model = joblib.load(ensemble_path)
        print("✅ Ensemble model loaded")
    except Exception as e:
        print(f"ℹ️ Ensemble model not found: {e}")

    return vectorizer, scaler, logistic_model, ensemble_model

def preprocess_input(data, vectorizer, scaler):
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

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        vectorizer, scaler, logistic_model, ensemble_model = load_models()
        features = preprocess_input(data, vectorizer, scaler)
        if features is None:
            return jsonify({"error": "Invalid input"}), 400

        predictions = {}
        if logistic_model:
            pred = logistic_model.predict(features)[0]
            prob = np.max(logistic_model.predict_proba(features))
            predictions['logistic'] = {"prediction": pred, "confidence": float(prob)}

        if ensemble_model:
            pred = ensemble_model.predict(features)[0]
            prob = np.max(ensemble_model.predict_proba(features))
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

# ✅ MongoDB Test Endpoint
@app.route("/api/ping-db", methods=["GET"])
def ping_db():
    try:
        count = users.count_documents({})
        return jsonify({"message": "MongoDB connected ✅", "user_count": count})
    except Exception as e:
        return jsonify({"error": f"MongoDB connection failed: {str(e)}"}), 500

# Start app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
