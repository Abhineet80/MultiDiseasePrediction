import firebase_admin
from firebase_admin import credentials, firestore

db = None

try:
    if not firebase_admin._apps:
        cred = credentials.Certificate("firebase-adminsdk.json")
        firebase_admin.initialize_app(cred)

    db = firestore.client()
    print("✅ Firebase connected successfully")

except Exception as e:
    print("❌ Firebase Error:", e)