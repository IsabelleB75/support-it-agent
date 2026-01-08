"""
Configuration partagée pour les DAGs Airflow
Les credentials sont lues depuis les variables d'environnement.
Inclut la gestion des embeddings incrémentaux (optimisation retraining).
"""
import os
import logging
import hashlib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

def get_db_connection_string():
    """
    Retourne la chaîne de connexion PostgreSQL.
    Les credentials doivent être définies dans les variables d'environnement Airflow.
    """
    host = os.getenv("POSTGRES_HOST", "host.docker.internal")
    port = os.getenv("POSTGRES_PORT", "5433")
    database = os.getenv("POSTGRES_DB", "support_tech")
    user = os.getenv("POSTGRES_USER", "bootcamp_user")
    password = os.getenv("POSTGRES_PASSWORD", "bootcamp_password")

    return f"postgresql://{user}:{password}@{host}:{port}/{database}"

def get_db_engine():
    """Retourne un engine SQLAlchemy configuré"""
    return create_engine(get_db_connection_string())

# Configuration MLflow
def get_mlflow_tracking_uri():
    return os.getenv("MLFLOW_TRACKING_URI", "http://host.docker.internal:5000")


# =============================================================================
# GESTION DES EMBEDDINGS INCREMENTAUX (optimisation retraining)
# =============================================================================

EMBEDDING_DIM = 384  # all-MiniLM-L6-v2
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def init_embeddings_table(engine):
    """Crée la table ticket_embeddings si elle n'existe pas"""
    with engine.begin() as conn:  # begin() pour auto-commit (SQLAlchemy 2.x)
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS ticket_embeddings (
                text_hash VARCHAR(64) NOT NULL,
                model_name VARCHAR(100) NOT NULL DEFAULT 'all-MiniLM-L6-v2',
                text_content TEXT NOT NULL,
                embedding vector(384),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (text_hash, model_name)
            )
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_ticket_embeddings_hash_model
            ON ticket_embeddings(text_hash, model_name)
        """))
    logging.info("Table ticket_embeddings initialisée (avec versioning modèle)")


def compute_text_hash(text: str) -> str:
    """Calcule un hash unique pour un texte"""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def get_cached_embeddings(engine, texts: list, model_name: str = DEFAULT_EMBEDDING_MODEL) -> dict:
    """
    Récupère les embeddings existants depuis la DB pour un modèle spécifique.
    Retourne un dict {text_hash: embedding_array}
    """
    if not texts:
        return {}

    hashes = [compute_text_hash(t) for t in texts]
    placeholders = ','.join([f"'{h}'" for h in hashes])

    with engine.connect() as conn:
        result = conn.execute(text(f"""
            SELECT text_hash, embedding
            FROM ticket_embeddings
            WHERE text_hash IN ({placeholders})
              AND model_name = :model_name
        """), {"model_name": model_name})
        rows = result.fetchall()

    cached = {}
    for row in rows:
        text_hash, emb_str = row
        # Parse le vecteur pgvector (format "[0.1,0.2,...]")
        if emb_str:
            emb_array = np.fromstring(emb_str.strip('[]'), sep=',')
            cached[text_hash] = emb_array

    return cached


def store_embeddings(engine, texts: list, embeddings: np.ndarray, model_name: str = DEFAULT_EMBEDDING_MODEL):
    """Stocke les nouveaux embeddings dans la DB avec versioning du modèle"""
    if len(texts) == 0:
        return

    with engine.begin() as conn:  # begin() pour auto-commit (SQLAlchemy 2.x)
        for txt, emb in zip(texts, embeddings):
            text_hash = compute_text_hash(txt)
            emb_str = '[' + ','.join(map(str, emb.tolist())) + ']'

            # UPSERT avec model_name (CAST explicite pour éviter conflit avec SQLAlchemy)
            conn.execute(text("""
                INSERT INTO ticket_embeddings (text_hash, model_name, text_content, embedding)
                VALUES (:hash, :model, :content, CAST(:emb AS vector))
                ON CONFLICT (text_hash, model_name) DO NOTHING
            """), {"hash": text_hash, "model": model_name, "content": txt[:1000], "emb": emb_str})

    logging.info(f"Stocké {len(texts)} nouveaux embeddings ({model_name})")


def get_incremental_embeddings(engine, texts: list, embedder, model_name: str = None) -> np.ndarray:
    """
    Récupère les embeddings de manière incrémentale:
    - Réutilise les embeddings existants depuis PostgreSQL (même modèle)
    - Ne calcule que les nouveaux
    - Stocke les nouveaux pour la prochaine fois

    Retourne un array numpy avec tous les embeddings dans l'ordre des texts.
    """
    init_embeddings_table(engine)

    # Détermine le nom du modèle (évite de mixer des embeddings de modèles différents)
    if model_name is None:
        # Essaie d'extraire le nom du modèle depuis l'embedder
        try:
            model_name = embedder.model_card_data.model_id if hasattr(embedder, 'model_card_data') and embedder.model_card_data else None
        except:
            pass

    # Fallback au modèle par défaut si toujours None
    if not model_name:
        model_name = DEFAULT_EMBEDDING_MODEL

    logging.info(f"Embedding model: {model_name}")

    # 1. Calcule les hashes et récupère les embeddings existants (même modèle)
    text_hashes = [compute_text_hash(t) for t in texts]
    cached = get_cached_embeddings(engine, texts, model_name)

    # 2. Identifie les textes sans embedding
    texts_to_compute = []
    indices_to_compute = []

    for i, (txt, text_hash) in enumerate(zip(texts, text_hashes)):
        if text_hash not in cached:
            texts_to_compute.append(txt)
            indices_to_compute.append(i)

    cache_hit_rate = (len(texts) - len(texts_to_compute)) / len(texts) * 100 if texts else 0
    logging.info(f"Embeddings: {len(texts) - len(texts_to_compute)}/{len(texts)} en cache ({cache_hit_rate:.1f}% hit rate)")

    # 3. Calcule les embeddings manquants
    if texts_to_compute:
        logging.info(f"Calcul de {len(texts_to_compute)} nouveaux embeddings...")
        new_embeddings = embedder.encode(texts_to_compute, show_progress_bar=True)

        # Stocke les nouveaux embeddings (avec model_name)
        store_embeddings(engine, texts_to_compute, new_embeddings, model_name)

        # Ajoute au cache local
        for txt, emb in zip(texts_to_compute, new_embeddings):
            cached[compute_text_hash(txt)] = emb

    # 4. Reconstruit l'array dans l'ordre original
    all_embeddings = np.zeros((len(texts), EMBEDDING_DIM))
    for i, text_hash in enumerate(text_hashes):
        all_embeddings[i] = cached[text_hash]

    return all_embeddings
