import json
import random
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

INTENTS_PATH = "data/intents.json"
VEC_PATH = "models/vectorizer.joblib"
MODEL_PATH = "models/classifier.joblib"

with open(INTENTS_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

texts = [] 
labels = []  

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        texts.append(pattern.lower())
        labels.append(intent["tag"])

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

model = LogisticRegression(max_iter=1000)
model.fit(X, labels)

joblib.dump(vectorizer, VEC_PATH)
joblib.dump(model, MODEL_PATH)

print("Treinamento concluído!")
