users.update_one(
  {"email": email},
  {"$set": {
    "verification_code": code,
    "code_expiry": datetime.utcnow() + timedelta(minutes=5)
  }}
)
