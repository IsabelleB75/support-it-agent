from sentence_transformers import SentenceTransformer
import psycopg2
import json
import requests
import xgboost as xgb
import joblib
import numpy as np
from scipy.sparse import hstack
import re
import os

# API Mistral - depuis variable d'environnement
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
API_URL = "https://api.mistral.ai/v1/chat/completions"
MODEL = "open-mistral-7b"

# Chemins locaux
LE_QUEUE_PATH = "le_queue.pkl"
LE_URGENCY_PATH = "le_urgency.pkl"
MODEL_QUEUE_PATH = "xgboost_queue_v2/model.xgb"
MODEL_URGENCY_PATH = "xgboost_urgency_v2/model.xgb"

# Load
le_queue = joblib.load(LE_QUEUE_PATH)
le_urgency = joblib.load(LE_URGENCY_PATH)
model_queue = xgb.Booster()
model_queue.load_model(MODEL_QUEUE_PATH)
model_urgency = xgb.Booster()
model_urgency.load_model(MODEL_URGENCY_PATH)

embedder = SentenceTransformer('all-MiniLM-L6-v2')

conn_params = {
    "host": "localhost", "port": 5433, "database": "support_tech",
    "user": "bootcamp_user", "password": "bootcamp_password"
}

# Keywords pour has_* (même que prep)
keywords = {
    'network': r'\b(network|wifi|vpn|connect|internet|lan|router)\b',
    'printer': r'\b(printer|imprimante|print|scan|scanner)\b',
    'security': r'\b(security|securite|password|login|access|breach|hack|malware)\b',
    'hardware': r'\b(hardware|laptop|pc|macbook|screen|disk|ssd|cpu)\b',
    'software': r'\b(software|app|update|bug|crash|install|version)\b',
}

def extract_num_features(text):
    text_lower = text.lower()
    body_len = len(text)
    has_feats = [1 if re.search(regex, text_lower) else 0 for regex in keywords.values()]
    # mock answer_len / ratio (ou 0)
    return [body_len, body_len * 0.8, 0.8] + has_feats  # approx

def predict_queue_urgency(query_text):
    embedding = embedder.encode(query_text).reshape(1, -1)
    num_feats = extract_num_features(query_text)
    full_vec = np.hstack([embedding, np.array(num_feats).reshape(1, -1)])
    
    dmatrix = xgb.DMatrix(full_vec)
    
    pred_queue_prob = model_queue.predict(dmatrix)[0]
    pred_queue_idx = np.argmax(pred_queue_prob)
    pred_queue = le_queue.classes_[pred_queue_idx]
    
    pred_urgency_prob = model_urgency.predict(dmatrix)[0]
    pred_urgency_idx = np.argmax(pred_urgency_prob)
    pred_urgency = le_urgency.classes_[pred_urgency_idx]
    
    return pred_queue, pred_urgency

def retrieve_rag(query_text, predicted_queue=None, top_k=5):
    query_emb = embedder.encode(query_text).tolist()
    conn = psycopg2.connect(**conn_params)
    cur = conn.cursor()
    
    sql = "SELECT content FROM rag_docs"
    params = []
    
    if predicted_queue:
        sql += " WHERE metadata->>'refined_queue' = %s"
        params.append(predicted_queue)
    
    sql += " ORDER BY embedding <=> %s::vector ASC LIMIT %s;"
    params.extend([query_emb, top_k])
    
    cur.execute(sql, params)
    results = cur.fetchall()
    cur.close()
    conn.close()
    return [row[0] for row in results]

def generate_response(user_query, retrieved_contents, pred_queue, pred_urgency):
    context = "\n\n".join(retrieved_contents)
    
    messages = [
        {"role": "system", "content": "Tu es un agent support technique IT interne expert, professionnel et concis. Structure ta réponse : 1. Reconnaissance du problème 2. Diagnostic rapide basé sur tickets similaires 3. Étapes de résolution pas à pas 4. Si escalade nécessaire, propose-la."},
        {"role": "user", "content": f"Catégorie détectée : {pred_queue}\nUrgence détectée : {pred_urgency}\nContexte (tickets similaires résolus) :\n{context}\n\nQuestion utilisateur :\n{user_query}"}
    ]

    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 512
    }

    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(API_URL, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return f"Erreur API : {response.status_code} - {response.text}"

# Test full agent réel
user_query = "Problème de connexion VPN sur Windows, impossible d'accéder au serveur corporate"

pred_queue, pred_urgency = predict_queue_urgency(user_query)
print(f"Prédiction réelle XGBoost v2 : Queue = {pred_queue} | Urgence = {pred_urgency}\n")

retrieved = retrieve_rag(user_query, predicted_queue=pred_queue, top_k=5)
response = generate_response(user_query, retrieved, pred_queue, pred_urgency)

print("Réponse agent :\n")
print(response)