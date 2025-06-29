# intent_model_utils.py
import joblib
from sentence_transformers import SentenceTransformer

# Model ve label encoder'ı yükle
intent_model = SentenceTransformer("all-MiniLM-L6-v2")
intent_classifier = joblib.load("models/intent_classifier.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

def predict_intent(user_query: str) -> str:
    vec = intent_model.encode([user_query])
    pred = intent_classifier.predict(vec)[0]
    return label_encoder.inverse_transform([pred])[0]
