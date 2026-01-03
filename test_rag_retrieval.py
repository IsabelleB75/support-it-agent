from sentence_transformers import SentenceTransformer
import psycopg2
import json

# Paramètres connexion Postgres (adapte si besoin)
conn_params = {
    "host": "localhost",
    "port": 5433,
    "database": "support_tech",
    "user": "bootcamp_user",
    "password": "bootcamp_password"
}

# Modèle embeddings (même que pour ingestion)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Query test (change-la pour tester différents cas)
query_text = "Problème de connexion VPN sur Windows, impossible d'accéder au serveur corporate"

# Embed la query
query_embedding = model.encode(query_text).tolist()

# Connexion et recherche top-5
conn = psycopg2.connect(**conn_params)
cur = conn.cursor()

# Requête similarity (cosine avec <=>)metadata
sql = """
SELECT 
    id,
    title,
    content,
    metadata,
    embedding <=> %s::vector AS distance  -- 0 = parfait match, 1 = opposé
FROM rag_docs
ORDER BY distance ASC
LIMIT 5;
"""

cur.execute(sql, (query_embedding,))
results = cur.fetchall()

print(f"\nQuery : {query_text}\n")
print("Top 5 résultats similaires :\n")

for rank, (doc_id, title, content, metadata_json, distance) in enumerate(results, 1):
    metadata = metadata_json
    print(f"--- Rank {rank} | Score similarity : {1 - distance:.4f} (plus proche de 1 = meilleur) ---")
    print(f"ID : {doc_id}")
    print(f"Title : {title}")
    print(f"Queue raffinée : {metadata.get('refined_queue')}")
    print(f"Urgence : {metadata.get('urgency_level')}")
    print(f"Snippet content :\n{content[:600]}...\n")

cur.close()
conn.close()

print("Test retrieval terminé !")