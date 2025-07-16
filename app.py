import os
import numpy as np
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_mail import Mail, Message
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pathlib import Path
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
from serpapi_util import fetch_search_results

# === Load Environment Variables ===
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# === Flask Setup ===
app = Flask(__name__)

# ✅ Enable CORS for Netlify frontend
CORS(app, resources={r"/.*": {"origins": "https://dynamic-sunburst-5f73a6.netlify.app"}},
     supports_credentials=True,
     allow_headers=["Content-Type", "Authorization"])

@app.after_request
def after_request(response):
    print("🔍 Response headers:", response.headers)
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
print("🔗 Connecting to MongoDB:", mongo_uri)

try:
    client = MongoClient(mongo_uri)
    db = client["medicalDB"]
    users = db["users"]
    users.count_documents({})  # test query
    print("✅ MongoDB connected and users collection loaded")
except Exception as e:
    print(f"❌ MongoDB collection access failed: {e}")
    users = None

@app.route("/api/debug", methods=["GET"])
def debug_info():
    return jsonify({
        "mongo_uri_loaded": mongo_uri is not None,
        "users_collection_connected": users is not None
    })

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

        print(f"📅 Received registration request: email={email}")

        if not email or not password:
            return jsonify({"message": "Missing email or password"}), 400

        if users.find_one({"email": email}):
            return jsonify({"message": "User already exists"}), 400

        hashed_password = generate_password_hash(password)
        users.insert_one({"email": email, "password": hashed_password})
        return jsonify({"message": "Registration successful"}), 201
    except Exception as e:
        print(f"🔥 Exception in /api/register: {e}")
        return jsonify({"message": "Registration failed", "error": str(e)}), 500

@app.route("/api/login-step1", methods=["POST"])
def login_step1():
    try:
        data = request.get_json(force=True)
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
        print("🔥 Exception occurred in login-step1:", e)
        return jsonify({"message": "Login failed", "error": str(e)}), 500

@app.route("/api/login-step2", methods=["POST"])
def login_step2():
    try:
        data = request.get_json()
        email, code = data.get("email"), data.get("code")
        user = users.find_one({"email": email})

        if not user or user.get("verification_code") != code:
            return jsonify({"message": "Invalid verification code"}), 401

        if datetime.utcnow() > user.get("code_expiry"):
            return jsonify({"message": "Code expired"}), 401

        users.update_one({"email": email}, {"$unset": {"verification_code": "", "code_expiry": ""}})
        return jsonify({"message": "Login successful", "token": "dummy_token"}), 200
    except Exception as e:
        print("🔥 Exception in login-step2:", e)
        return jsonify({"message": "Login failed", "error": str(e)}), 500

# === PROXY PREDICTION TO HUGGING FACE ===
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        hf_url = "https://sampath563-medica-backend.hf.space/predict"
        response = requests.post(hf_url, json=data)
        return jsonify(response.json()), response.status_code
    except Exception as e:
        print(f"❌ Proxy prediction error: {e}")
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
