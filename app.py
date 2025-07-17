import os
import numpy as np
import requests
import random
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_mail import Mail, Message
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pathlib import Path
import traceback
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
from serpapi_util import fetch_search_results

# === Load Environment Variables ===
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# === Flask Setup ===
app = Flask(__name__)

# === Enable CORS for frontend ===
CORS(app, resources={r"/.*": {"origins": "https://dynamic-sunburst-5f73a6.netlify.app"}}, supports_credentials=True)

@app.after_request
def after_request(response):
    return response

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
mongo_uri = os.getenv("MONGO_URI")
client = MongoClient(mongo_uri)
db = client["medicalDB"]
users = db["users"]

# === Utility: Send Email ===
def send_verification_email(email, code):
    msg = Message("Your Verification Code", sender=gmail_user, recipients=[email])
    msg.body = f"Your verification code is: {code}"
    mail.send(msg)
    print(f"✅ Email sent to {email} with code {code}")

# === Register Endpoint ===
@app.route("/api/register", methods=["POST"])
def register():
    data = request.get_json(force=True)
    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        return jsonify({"message": "Missing email or password"}), 400

    if users.find_one({"email": email}):
        return jsonify({"message": "User already exists"}), 400

    hashed_password = generate_password_hash(password)
    users.insert_one({
        "email": email,
        "password": hashed_password,
        "is_verified": False
    })

    return jsonify({"message": "Registration successful"}), 201

# === Login Step 1 ===
@app.route("/api/login-step1", methods=["POST"])
def login_step1():
    try:
        data = request.get_json()
        email, password = data.get("email"), data.get("password")

        user = users.find_one({"email": email})

        if not user:
            return jsonify({"message": "User not found"}), 404

        if not check_password_hash(user["password"], password):
            return jsonify({"message": "Invalid password"}), 401

        if user.get("is_verified", False):
            return jsonify({"token": "dummy_token"}), 200

        # If not verified, send code
        verification_code = str(random.randint(100000, 999999))
        expiry = datetime.utcnow() + timedelta(minutes=10)

        users.update_one(
            {"email": email},
            {"$set": {
                "verification_code": verification_code,
                "code_expiry": expiry
            }}
        )

        send_verification_email(email, verification_code)

        return jsonify({"step": 2, "message": "Verification code sent"}), 200

    except Exception as e:
        return jsonify({"message": "Login step 1 failed", "error": str(e)}), 500

# === Login Step 2 ===
@app.route("/api/login-step2", methods=["POST"])
def login_step2():
    try:
        data = request.get_json()
        email = data.get("email")
        code = data.get("code")

        user = users.find_one({"email": email})

        if not user:
            return jsonify({"message": "User not found"}), 404

        if user.get("is_verified", False):
            return jsonify({"message": "Already verified", "token": "dummy_token"}), 200

        if user.get("verification_code") != code:
            return jsonify({"message": "Invalid verification code"}), 401

        if datetime.utcnow() > user.get("code_expiry"):
            return jsonify({"message": "Code expired"}), 401

        users.update_one(
            {"email": email},
            {"$set": {"is_verified": True}, "$unset": {"verification_code": "", "code_expiry": ""}}
        )

        return jsonify({"message": "Login successful", "token": "dummy_token"}), 200

    except Exception as e:
        return jsonify({"message": "Login step 2 failed", "error": str(e)}), 500

# === Proxy to Hugging Face for Prediction ===
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        hf_url = "https://sampath563-medica-backend.hf.space/predict"
        response = requests.post(hf_url, json=data)
        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# === Treatment Plan Generator ===
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

# === Health Check Endpoints ===
@app.route("/api/ping-db", methods=["GET"])
def ping_db():
    try:
        count = users.count_documents({})
        return jsonify({"message": "MongoDB connected ✅", "user_count": count})
    except Exception as e:
        return jsonify({"error": f"MongoDB connection failed: {str(e)}"}), 500

# === Main Entrypoint ===
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
