# ================================
# STEP 1: INSTALL & IMPORT
# ================================
!pip install google-generativeai sentence-transformers

import google.generativeai as genai
import json, numpy as np
from sentence_transformers import SentenceTransformer, util

# ================================
# STEP 2: LOAD JSON MEMORY
# ================================
path = "/content/dataset.json"
with open(path, "r") as f:
    memory_data = json.load(f)

# ================================
# STEP 3: FLATTEN JSON FOR RETRIEVAL
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
# STEP 4: BUILD SEMANTIC EMBEDDINGS
# ================================
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(memory_texts, convert_to_tensor=True)

# ================================
# STEP 5: CONFIGURE GEMINI API
# ================================
genai.configure(api_key="AIzaSyB-zTFs4BYaYME6ilDOsBidPtB_WHfcOsA")  # Replace with your key

# ================================
# STEP 6: SMART RETRIEVAL
# (keyword + semantic search)
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

def retrieve_context(query, top_k=3):
    q = query.lower()

    # ----- KEYWORD → JSON KEY MATCHING -----
    for key_word, json_key in keyword_map.items():
        if key_word in q:
            matched_items = [v for k, v in memory_pairs if json_key.lower() in k.lower()]
            if matched_items:
                return matched_items

    # ----- SEMANTIC RETRIEVAL (fallback) -----
    query_emb = model.encode(query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_emb, embeddings)[0]
    top_idx = np.argsort(-scores.cpu().numpy())[:top_k]

    return [memory_texts[i] for i in top_idx]

# ================================
# STEP 7: CHAT FUNCTION
# ================================
def ask_jabez(user_input):
    context = retrieve_context(user_input)

    # If context is empty → unseen/new question
    if context:
        full_context = "\n".join(context)
    else:
        full_context = "[No prior memory, answer freely based on your knowledge]"

    prompt = f"""
You are Jabez, a friendly emotional companion of Ramya.
Use the memory context below if available.
Speak naturally, warmly, and personally.

Memory Context:
{full_context}

User Question: {user_input}
"""

    try:
        response = genai.GenerativeModel("models/gemini-2.5-flash").generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"[Error] {e}"

# ================================
# STEP 8: CHAT LOOP
# ================================
print("🗣️ Jabez (Emotion + Memory) is ready! Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Jabez: Bye Ramya ❤️")
        break

    answer = ask_jabez(user_input)
    print("Jabez:", answer, "\n")
