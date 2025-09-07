
import streamlit as st
import requests
import json
import os
from openai import OpenAI
import re

# ---------------------------
# OpenAI Config
# ---------------------------
# keyapi = k_api  # your API key
keyapi = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
# ---------------------------
# DeepSeek Config
# ---------------------------
API_URL = "http://localhost:11434/api/chat"
MODEL_DEEPSEEK = "deepseek-r1:1.5b"
STREAM = False

# ---------------------------
# File paths
# ---------------------------
BASE_KNOWLEDGE_FILE = "knowledge.json"
USER_CHATS_DIR = "chats"

# ---------------------------
# Model logos
# ---------------------------
MODEL_LOGOS = {
    "gpt-4o": "https://upload.wikimedia.org/wikipedia/commons/0/04/ChatGPT_logo.svg",
    "deepseek-r1": "https://upload.wikimedia.org/wikipedia/commons/b/ba/Deepseek-logo-icon.svg"
}

# ---------------------------
# Helpers
# ---------------------------
def ensure_user_dir(user_id):
    user_dir = os.path.join(USER_CHATS_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)
    return user_dir

def load_base_knowledge():
    if os.path.exists(BASE_KNOWLEDGE_FILE):
        with open(BASE_KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def add_knowledge(user_messages):
    """Add new messages to knowledge.json without duplicating existing ones."""
    knowledge = load_base_knowledge()
    existing_contents = {msg["content"] for msg in knowledge}
    for msg in user_messages:
        if msg["content"] not in existing_contents:
            knowledge.append(msg)
    with open(BASE_KNOWLEDGE_FILE, "w", encoding="utf-8") as f:
        json.dump(knowledge, f, ensure_ascii=False, indent=4)
    return len(knowledge)

def list_user_chats(user_id):
    user_dir = ensure_user_dir(user_id)
    files = sorted(os.listdir(user_dir))
    chats = []
    for f in files:
        path = os.path.join(user_dir, f)
        with open(path, "r", encoding="utf-8") as file:
            messages = json.load(file)
            if messages:
                first_prompt = messages[0]["content"]
                title = first_prompt[:30] + ("..." if len(first_prompt) > 30 else "")
                model_used = messages[-1].get("model", "deepseek-r1")
            else:
                title = f"Session {f}"
                model_used = "deepseek-r1"
        chats.append((f, model_used, title))
    return chats

def load_chat(user_id, filename):
    user_dir = ensure_user_dir(user_id)
    path = os.path.join(user_dir, filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_chat(user_id, messages, session_filename):
    user_dir = ensure_user_dir(user_id)
    path = os.path.join(user_dir, session_filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=4)

def clean_deepseek_response(text):
    """Remove <think> blocks from DeepSeek responses."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def query_deepseek(messages):
    try:
        response = requests.post(API_URL, json={
            "model": MODEL_DEEPSEEK,
            "messages": messages,
            "stream": STREAM
        }).json()
        return clean_deepseek_response(response["message"]["content"])
    except Exception as e:
        return f"Error contacting DeepSeek model: {e}"

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Aali Chatbot")

if "user_id" not in st.session_state:
    st.session_state.user_id = "guest"
user_id = st.session_state.user_id

base_knowledge = load_base_knowledge()

# --- Sidebar ---
st.sidebar.title("Chats")

# Model selection
if "selected_model" not in st.session_state or st.session_state.selected_model is None:
    st.session_state.selected_model = st.sidebar.selectbox(
        "Choose a model to chat with:",
        ["gpt-4o", "deepseek-r1"]
    )
else:
    st.sidebar.text(f"Current model: {st.session_state.selected_model}")

# New chat button
if st.sidebar.button("➕ New Chat"):
    st.session_state.current_session = None
    st.session_state.messages = []
    st.session_state.selected_model = None

# Load previous chats
chats = list_user_chats(user_id)
chat_options = ["-- New Chat --"]
chat_mapping = {}
for filename, model, title in chats:
    display_title = f"{title} ({model})"
    chat_options.append(display_title)
    chat_mapping[display_title] = (filename, model)

selected = st.sidebar.radio("Previous Chats:", chat_options)

if selected != "-- New Chat --" and not st.session_state.get("current_session"):
    filename, model = chat_mapping[selected]
    st.session_state.current_session = filename
    st.session_state.selected_model = model
    st.session_state.messages = load_chat(user_id, filename)
elif selected == "-- New Chat --" and "current_session" not in st.session_state:
    st.session_state.current_session = None
    st.session_state.messages = []

# --- Chat Display ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    role = message["role"]
    avatar = None
    if role == "assistant":
        model_used = message.get("model", st.session_state.selected_model)
        avatar = MODEL_LOGOS.get(model_used)
    with st.chat_message(role, avatar=avatar):
        st.markdown(message["content"])

# --- Chat Input ---
if prompt := st.chat_input("Ask me something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    messages_to_send = [{"role": "system", "content": "You are a helpful assistant."}] + base_knowledge + st.session_state.messages

    if st.session_state.selected_model == "deepseek-r1":
        assistant_reply = query_deepseek(messages_to_send)
    else:
        client = OpenAI(api_key=keyapi)
        response = client.chat.completions.create(
            model=st.session_state.selected_model,
            messages=messages_to_send
        )
        assistant_reply = response.choices[0].message.content

    avatar = MODEL_LOGOS.get(st.session_state.selected_model)
    with st.chat_message("assistant", avatar=avatar):
        st.markdown(assistant_reply)

    st.session_state.messages.append({
        "role": "assistant",
        "content": assistant_reply,
        "model": st.session_state.selected_model
    })

    if not st.session_state.current_session:
        filename = prompt[:30].replace(" ", "_") + ".json"
        st.session_state.current_session = filename

    save_chat(user_id, st.session_state.messages, st.session_state.current_session)

# --- Add Knowledge Button ---
if st.button("💡 Add Knowledge"):
    if st.session_state.messages:
        count = add_knowledge(st.session_state.messages)
        st.success(f"Knowledge updated! Total entries: {count}")
    else:
        st.warning("No messages to add as knowledge.")
