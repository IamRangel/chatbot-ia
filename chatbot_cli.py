import json, random, joblib, os

MODELS_DIR = "models"
VEC_PATH = os.path.join(MODELS_DIR, "vectorizer.joblib")
MODEL_PATH = os.path.join(MODELS_DIR, "classifier.joblib")
INTENTS_PATH = "data/intents.json"
CONF_THRESHOLD = 0.50

vectorizer = joblib.load(VEC_PATH)
model = joblib.load(MODEL_PATH)

with open(INTENTS_PATH, "r", encoding="utf-8") as f:
    intents = json.load(f)["intents"]

def get_response(tag):
    for it in intents:
        if it["tag"] == tag:
            return random.choice(it["responses"])
    return random.choice([r for it in intents if it["tag"]=="fallback" for r in it["responses"]])

print("Chatbot iniciado! Digite 'sair' para encerrar.")
while True:
    user = input("Você: ").strip()
    if user.lower() == "sair":
        print("Chatbot: Até logo!")
        break
    X = vectorizer.transform([user.lower()])
    proba = model.predict_proba(X)[0]
    best_idx = proba.argmax()
    best_tag = model.classes_[best_idx]
    best_conf = proba[best_idx]
    if best_conf < CONF_THRESHOLD:
        resp = get_response("fallback")
    else:
        resp = get_response(best_tag)
    print("Chatbot:", resp)
