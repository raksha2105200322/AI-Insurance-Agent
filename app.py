import streamlit as st
import json
import os
from llm_recommender import generate_recommendation
from retriever import process_documents

CHAT_FILE = "chat_history.json"

# -----------------------------------
# Function to load existing chat
# -----------------------------------
def load_chat_history():
    if os.path.exists(CHAT_FILE):
        with open(CHAT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# -----------------------------------
# Function to save chat
# -----------------------------------
def save_chat_history(history):
    with open(CHAT_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4, ensure_ascii=False)

# -----------------------------------
# Initialize or load session
# -----------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history()

# -----------------------------------
# Streamlit UI
# -----------------------------------
st.title("💬 AI Insurance Agent — Conversation Chat")

# Input section
user_query = st.text_input("Ask a customer query:")

if st.button("Send"):
    if user_query.strip():
        # Add user query
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        # Generate AI recommendation
        profile_text = "Age: 35, Married, 2 kids, looking for family coverage"
        retrieved_chunks = ["Basic Health Plan", "Family Protection Plan", "Senior Health Plus"]

        response = generate_recommendation(profile_text, retrieved_chunks)

        # Convert response dict to readable text
        response_text = json.dumps(response, indent=2)

        st.session_state.chat_history.append({"role": "assistant", "content": response_text})

        # Save to file
        save_chat_history(st.session_state.chat_history)

# -----------------------------------
# Display chat history
# -----------------------------------
st.subheader("🗂 Conversation History")

for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"🧑 **You:** {msg['content']}")
    else:
        st.markdown(f"🤖 **AI:** {msg['content']}")
