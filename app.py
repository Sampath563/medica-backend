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

# === Load Environment Variables ===
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# === Flask Setup ===
app = Flask(__name__)

# ✅ Enable CORS for all /api/* and /predict routes
# Allow Netlify frontend
CORS(app, origins=["https://dynamic-sunburst-5f73a6.netlify.app"], supports_credentials=True)

@app.before_request
def log_request_info():
    print(f"➡️ {request.method} {request.path}")
    if request.method == 'OPTIONS':
        print("🔄 Handling preflight OPTIONS request")

# === Mail Configuration ===
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

# === MongoDB Setup ===
mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
client = MongoClient(
    mongo_uri,
    tls=True,
    tlsAllowInvalidCertificates=True,  # <-- add this only temporarily if needed
    serverSelectionTimeoutMS=5000
)

db = client["medicalDB"]
users = db["users"]

# === Email Auth ===
def generate_code():
    return str(np.random.randint(100000, 999999))

def send_verification_email(email, code):
    try:
        msg = Message("Your Verification Code", sender=gmail_user, recipients=[email])
        msg.body = f"Your verification code is: {code}"
        mail.send(msg)
        print(f"✅ Email sent to {email} with code {code}")
    except Exception as e:
        print(f"❌ Email failed: {e}")
        raise

@app.route("/api/test-email", methods=["GET"])
def test_email():
    try:
        send_verification_email("bsampath563@gmail.com", "999999")
        return jsonify({"message": "Test email sent ✅"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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
        return jsonify({"message": "Registration failed", "error": str(e)}), 500

@app.route("/api/login-step1", methods=["POST"])
def login_step1():
    try:
        print("🚨 login-step1 triggered")

        data = request.get_json(force=True)
        print("📥 Received data:", data)

        email = data.get("email")
        password = data.get("password")

        if not email or not password:
            print("⚠️ Missing email or password")
            return jsonify({"message": "Email and password required"}), 400

        user = users.find_one({"email": email})
        print("👤 User found in DB:", user)

        if not user:
            print("❌ User not found")
            return jsonify({"message": "Invalid credentials"}), 401

        if not check_password_hash(user["password"], password):
            print("❌ Password mismatch")
            return jsonify({"message": "Invalid credentials"}), 401

        code = generate_code()
        expiry = datetime.utcnow() + timedelta(minutes=5)

        print("🔐 Generated code:", code)
        users.update_one({"email": email}, {"$set": {
            "verification_code": code,
            "code_expiry": expiry
        }})

        send_verification_email(email, code)
        print("✅ Email sent")

        return jsonify({"message": "Verification code sent", "step": 2}), 200
    except Exception as e:
        print("🔥 Exception occurred:", e)
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

# === Model Utilities ===
def download_model_if_missing(file_id, output_path):
    if not os.path.exists(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"⬇️ Downloading {output_path}...")
        gdown.download(url, output_path, quiet=False)
        print(f"✅ Downloaded {output_path}")

def load_models():
    base = os.path.join(os.path.dirname(__file__), "models")
    paths = {
        "logistic": ("1JBOrSJtfZL7kKeOqOedi1e3cYMpSt2rd", os.path.join(base, "best_medical_model_logistic_regression.pkl")),
        "ensemble": ("15J6ieS97efmxGySE6c_9yMEbwZdMxpII", os.path.join(base, "ensemble_medical_model.pkl")),
        "scaler": ("1aabdJ-DvGawM5vI-B9m69IuIqeNL5Re_", os.path.join(base, "medical_scaler.pkl")),
        "vectorizer": ("1lyr6Qnx3Wqr-fsWqaBQoTKra629Ac91x", os.path.join(base, "symptom_vectorizer.pkl")),
    }

    for file_id, path in paths.values():
        download_model_if_missing(file_id, path)

    try:
        vectorizer = joblib.load(paths["vectorizer"][1])
        scaler = joblib.load(paths["scaler"][1])
        logistic_model = joblib.load(paths["logistic"][1])
        ensemble_model = joblib.load(paths["ensemble"][1])
        return vectorizer, scaler, logistic_model, ensemble_model
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        raise

def preprocess_input(data, vectorizer, scaler):
    try:
        symptoms = data.get("symptoms", "")
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

        symptom_vec = vectorizer.transform([symptoms])
        vital_vec = scaler.transform([vitals])
        return np.hstack([symptom_vec.toarray(), vital_vec])
    except Exception as e:
        print(f"❌ Preprocessing error: {e}")
        return None

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        print("🔍 Predict Request:", data)

        vectorizer, scaler, logistic_model, ensemble_model = load_models()
        features = preprocess_input(data, vectorizer, scaler)
        if features is None:
            return jsonify({"error": "Invalid input"}), 400

        predictions = {}
        if logistic_model:
            pred = logistic_model.predict(features)[0]
            prob = np.max(logistic_model.predict_proba(features))
            predictions["logistic"] = {"prediction": pred, "confidence": float(prob)}

        if ensemble_model:
            pred = ensemble_model.predict(features)[0]
            prob = np.max(ensemble_model.predict_proba(features))
            predictions["ensemble"] = {"prediction": pred, "confidence": float(prob)}

        return jsonify({"result": predictions})
    except Exception as e:
        print(f"❌ Prediction error: {e}")
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

@app.route("/api/ping-db", methods=["GET"])
def ping_db():
    try:
        count = users.count_documents({})
        return jsonify({"message": "MongoDB connected ✅", "user_count": count})
    except Exception as e:
        return jsonify({"error": f"MongoDB connection failed: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
