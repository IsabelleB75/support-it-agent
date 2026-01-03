from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from sentence_transformers import SentenceTransformer
import psycopg2
import requests
import xgboost as xgb
import joblib
import numpy as np
import re
import os
from datetime import datetime

# API Mistral - depuis variable d'environnement
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
API_URL = "https://api.mistral.ai/v1/chat/completions"
MODEL = "open-mistral-7b"

# Chemins locaux
le_queue = joblib.load("le_queue.pkl")
le_urgency = joblib.load("le_urgency.pkl")
model_queue = xgb.Booster()
model_queue.load_model("xgboost_queue_v2/model.xgb")
model_urgency = xgb.Booster()
model_urgency.load_model("xgboost_urgency_v2/model.xgb")

embedder = SentenceTransformer('all-MiniLM-L6-v2')

conn_params = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", "5433")),
    "database": "support_tech",
    "user": "bootcamp_user",
    "password": "bootcamp_password"
}

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
    return [body_len, body_len * 0.8, 0.8] + has_feats

def predict_queue_urgency(query_text):
    embedding = embedder.encode(query_text).reshape(1, -1)
    num_feats = extract_num_features(query_text)
    full_vec = np.hstack([embedding, np.array(num_feats).reshape(1, -1)])
    dmatrix = xgb.DMatrix(full_vec)
    pred_queue = le_queue.classes_[np.argmax(model_queue.predict(dmatrix)[0])]
    pred_urgency = le_urgency.classes_[np.argmax(model_urgency.predict(dmatrix)[0])]
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
        {"role": "system", "content": """Tu es un agent de support technique IT de l'entreprise TechCorp.

COORDONNEES DU SUPPORT:
- Telephone: 01 23 45 67 89
- Email: support@techcorp.fr
- Horaires: Lun-Ven 8h-18h

REGLES IMPORTANTES:
1. Reponds TOUJOURS dans la MEME LANGUE que la question (francais si question en francais)
2. Donne des solutions CONCRETES et DIRECTES - ne demande PAS d'informations supplementaires
3. Propose 2-3 etapes claires pour resoudre le probleme
4. Si le probleme necessite une intervention physique, indique les coordonnees du support
5. Sois professionnel, concis et rassurant
6. Termine par une phrase d'encouragement

FORMAT DE REPONSE:
- Salutation courte
- Diagnostic rapide
- Solutions en etapes numerotees
- Contact support si necessaire
- Conclusion positive"""},
        {"role": "user", "content": f"Categorie:{pred_queue}\nUrgence:{pred_urgency}\nContexte:\n{context}\n\nQuestion:\n{user_query}"}
    ]
    payload = {"model": MODEL, "messages": messages, "temperature": 0.7, "max_tokens": 512}
    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}
    response = requests.post(API_URL, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    return f"Erreur: {response.status_code}"

app = FastAPI(title="Agent Support IT - MLOps Bootcamp")

class Query(BaseModel):
    user_query: str

class Feedback(BaseModel):
    prediction_id: int
    correct_queue: Optional[str] = None
    correct_urgency: Optional[str] = None

def log_prediction(input_text: str, pred_queue: str, pred_urgency: str,
                   conf_queue: float = None, conf_urgency: float = None) -> int:
    """Enregistre une prediction dans la base pour monitoring et retraining"""
    try:
        conn = psycopg2.connect(**conn_params)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO prediction_logs
            (input_text, predicted_queue, predicted_urgency, confidence_queue, confidence_urgency)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
        """, (input_text, pred_queue, pred_urgency, conf_queue, conf_urgency))
        prediction_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        return prediction_id
    except Exception as e:
        print(f"Erreur logging prediction: {e}")
        return None

@app.post("/predict")
def agent(query: Query):
    pred_queue, pred_urgency = predict_queue_urgency(query.user_query)
    retrieved = retrieve_rag(query.user_query, predicted_queue=pred_queue, top_k=5)
    response = generate_response(query.user_query, retrieved, pred_queue, pred_urgency)

    # Log la prediction pour monitoring et retraining
    prediction_id = log_prediction(query.user_query, pred_queue, pred_urgency)

    return {
        "prediction_id": prediction_id,
        "query": query.user_query,
        "predicted_queue": pred_queue,
        "predicted_urgency": pred_urgency,
        "response": response
    }

@app.post("/feedback")
def submit_feedback(feedback: Feedback):
    """Enregistre le feedback utilisateur pour ameliorer le modele"""
    try:
        conn = psycopg2.connect(**conn_params)
        cur = conn.cursor()
        cur.execute("""
            UPDATE prediction_logs
            SET feedback_queue = %s, feedback_urgency = %s, feedback_at = %s
            WHERE id = %s
        """, (feedback.correct_queue, feedback.correct_urgency, datetime.now(), feedback.prediction_id))
        conn.commit()
        cur.close()
        conn.close()
        return {"status": "success", "message": "Feedback enregistre"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
