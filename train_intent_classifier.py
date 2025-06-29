import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.multiclass import unique_labels
import joblib

# Veriyi yükle
df = pd.read_csv("data/intent.csv", on_bad_lines="skip")  
# Embedder ve label encoder
embedder = SentenceTransformer("all-MiniLM-L6-v2")
X = embedder.encode(df["utterance"].tolist(), show_progress_bar=True)
le = LabelEncoder()
y = le.fit_transform(df["intent"])

# Eğitim/test böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

### Logistic Regression modeli
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)

# LR için sınıf raporu
lr_labels = unique_labels(y_test, lr_pred)
print("🔹 Logistic Regression Accuracy:", lr_acc)
print(classification_report(y_test, lr_pred, labels=lr_labels, target_names=le.inverse_transform(lr_labels)))

### SVM modeli
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)
svm_acc = accuracy_score(y_test, svm_pred)

# SVM için sınıf raporu
svm_labels = unique_labels(y_test, svm_pred)
print("🔹 SVM Accuracy:", svm_acc)
print(classification_report(y_test, svm_pred, labels=svm_labels, target_names=le.inverse_transform(svm_labels)))

# En iyi modeli seç ve kaydet
if lr_acc >= svm_acc:
    print("✅ Logistic Regression seçildi ve kaydedildi.")
    joblib.dump(lr_model, "models/intent_classifier.pkl")
else:
    print("✅ SVM seçildi ve kaydedildi.")
    joblib.dump(svm_model, "models/intent_classifier.pkl")

# Ortak nesneleri kaydet
joblib.dump(le, "models/label_encoder.pkl")
joblib.dump(embedder, "models/embedder.pkl")

# Test için örnek tahmin
def predict_intent(user_query):
    vec = embedder.encode([user_query])
    best_model = joblib.load("models/intent_classifier.pkl")
    pred = best_model.predict(vec)[0]
    return le.inverse_transform([pred])[0]

sample_inputs = [
    "how can I restart a failed service automatically?",
    "i want to add a new user on my server",
    "nginx is not loading the reverse proxy properly",
    "how to setup alert rules in grafana",
    "check disk usage on linux",
    "Hello"
]

for query in sample_inputs:
    print(f"🔸 Query: {query}")
    print(f"👉 Predicted Intent: {predict_intent(query)}\n")
