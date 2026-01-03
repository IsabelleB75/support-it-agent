from sentence_transformers import SentenceTransformer
import psycopg2
import json
import requests
import os

# API Mistral - depuis variable d'environnement
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")

# Endpoint + modèle gratuit rapide
API_URL = "https://api.mistral.ai/v1/chat/completions"
MODEL = "open-mistral-7b"  # ou "mistral-small-latest" pour meilleur

# Modèle embeddings RAG
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Connexion Postgres
conn_params = {
    "host": "localhost", "port": 5433, "database": "support_tech",
    "user": "bootcamp_user", "password": "bootcamp_password"
}

def retrieve_rag(query_text, top_k=5):
    query_emb = embedder.encode(query_text).tolist()
    conn = psycopg2.connect(**conn_params)
    cur = conn.cursor()
    sql = """
    SELECT content, metadata, embedding <=> %s::vector AS distance
    FROM rag_docs
    ORDER BY distance ASC LIMIT %s;
    """
    cur.execute(sql, (query_emb, top_k))
    results = cur.fetchall()
    cur.close()
    conn.close()
    return [row[0] for row in results]

def generate_response(user_query, retrieved_contents, pred_queue="Unknown", pred_urgency="medium"):
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
        return f"Erreur API Mistral : {response.status_code} - {response.text}"

# Test agent
user_query = "Problème de connexion VPN sur Windows, impossible d'accéder au serveur corporate"

# Mock prediction (remplace par XGBoost plus tard)
pred_queue = "Network / Infrastructure"
pred_urgency = "high"

retrieved = retrieve_rag(user_query, top_k=5)
response = generate_response(user_query, retrieved, pred_queue, pred_urgency)

print("Réponse agent (Mistral API gratuite) :\n")
print(response)