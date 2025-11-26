import streamlit as st
import google.generativeai as genai
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util

# ================================
# LOAD JSON MEMORY
# ================================
@st.cache_resource
def load_memory():
    with open("dataset.json", "r") as f:
        return json.load(f)

memory_data = load_memory()

# ================================
# FLATTEN JSON FOR EMBEDDING
# ================================
def flatten_json(data, parent_key=""):
    items = []
    for k, v in data.items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_json(v, new_key))
        elif isinstance(v, list):
            for i, item in enumerate(v):
                items.append((f"{new_key}[{i}]", str(item)))
        else:
            items.append((new_key, str(v)))
    return items

memory_pairs = flatten_json(memory_data)
memory_texts = [v for _, v in memory_pairs]

# ================================
# BUILD EMBEDDINGS
# ================================
@st.cache_resource
def load_embeddings():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(memory_texts, convert_to_tensor=True)
    return model, embeddings

model, embeddings = load_embeddings()

# ================================
# GEMINI API KEY INPUT
# ================================
st.sidebar.header("Gemini API Settings")
api_key = st.sidebar.text_input("Enter your Gemini API Key:", type="password")

if api_key:
    genai.configure(api_key=api_key)

# ================================
# KEYWORD MAP
# ================================
keyword_map = {
    "favorite food": "favorite_foods",
    "favourite food": "favorite_foods",
    "foods": "favorite_foods",
    "food": "favorite_foods",

    "class teacher": "class_teacher",
    "teacher": "class_teacher",

    "dean": "dean",
    "hod": "dean",
    "head": "dean",

    "birthday": "birthdays",
    "dob": "birthdays",
    "birth date": "birthdays",

    "gift": "gifts",
    "hobby": "hobbies",
    "likes": "likes",
    "dislike": "dislikes"
}

# ================================
# SMART CONTEXT RETRIEVAL
# ================================
def retrieve_context(query, top_k=3):
    q = query.lower()

    # keyword-based match
    for key_word, json_key in keyword_map.items():
        if key_word in q:
            matched_items = [v for k, v in memory_pairs if json_key.lower() in k.lower()]
            if matched_items:
                return matched_items

    # semantic search fallback
    query_emb = model.encode(query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_emb, embeddings)[0]
    top_idx = np.argsort(-scores.cpu().numpy())[:top_k]

    return [memory_texts[i] for i in top_idx]

# ================================
# STREAMLIT UI
# ================================
st.title("🧠 Jabez — Memory Enhanced AI Companion")

user_msg = st.text_input("Ask something to Jabez:")

if st.button("Send"):
    if not api_key:
        st.error("⚠ Please enter your Gemini API key in the sidebar!")
    else:
        context = retrieve_context(user_msg)
        full_context = "\n".join(context) if context else "[No memory found]"

        prompt = f"""
You are Jabez, a friendly emotional companion of Ramya.
Use the memory context below if available.
Respond warmly and naturally.

Memory Context:
{full_context}

User Question: {user_msg}
"""

        try:
            response = genai.GenerativeModel("models/gemini-2.5-flash").generate_content(prompt)
            st.success(response.text.strip())
        except Exception as e:
            st.error(f"Error: {e}")

