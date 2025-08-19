import streamlit as st
import json, random, joblib, os

st.set_page_config(page_title="Chatbot Gustavo", page_icon="🤖")

MODELS_DIR = "models"
VEC_PATH = os.path.join(MODELS_DIR, "vectorizer.joblib")
MODEL_PATH = os.path.join(MODELS_DIR, "classifier.joblib")
INTENTS_PATH = os.path.join("data", "intents.json")
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

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🤖 Chatbot do Gustavo")

if not st.session_state.messages:
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Olá! Eu sou o chatbot do Gustavo 🤖. Pergunte algo sobre tecnologia, programação ou IA!"
    })

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(f"{'👤' if msg['role']=='user' else '🤖'} {msg['content']}")

user_input = st.chat_input("Digite sua mensagem...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    X = vectorizer.transform([user_input.lower()])
    proba = model.predict_proba(X)[0]
    best_idx = proba.argmax()
    best_tag = model.classes_[best_idx]
    best_conf = proba[best_idx]

    if best_conf < CONF_THRESHOLD:
        response = get_response("fallback")
    else:
        response = get_response(best_tag)

    st.session_state.messages.append({"role": "assistant", "content": response})

    with st.chat_message("user"):
        st.markdown(f"👤 {user_input}")
    with st.chat_message("assistant"):
        st.markdown(f"🤖 {response}")

example_questions = [
    "O que é Inteligência Artificial?",
    "Como faço análise de dados?",
    "Qual linguagem de programação aprender primeiro?",
    "Me ensina Python"
]

st.markdown("**Exemplos de perguntas:**")
cols = st.columns(len(example_questions))
for i, q in enumerate(example_questions):
    if cols[i].button(q):
        st.session_state.messages.append({"role": "user", "content": q})
        X = vectorizer.transform([q.lower()])
        proba = model.predict_proba(X)[0]
        best_idx = proba.argmax()
        best_tag = model.classes_[best_idx]
        best_conf = proba[best_idx]

        if best_conf < CONF_THRESHOLD:
            response = get_response("fallback")
        else:
            response = get_response(best_tag)

        st.session_state.messages.append({"role": "assistant", "content": response})

        with st.chat_message("user"):
            st.markdown(f"👤 {q}")
        with st.chat_message("assistant"):
            st.markdown(f"🤖 {response}")
