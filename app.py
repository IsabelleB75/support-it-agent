from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from sentence_transformers import SentenceTransformer
from contextlib import contextmanager
import psycopg2
from psycopg2 import pool
import requests
import xgboost as xgb
import joblib
import numpy as np
import re
import os
import logging
from datetime import datetime
from pathlib import Path

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Répertoire de base pour les modèles
BASE_DIR = Path(__file__).parent.resolve()

# API Mistral - depuis variable d'environnement (OBLIGATOIRE)
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    logger.warning("MISTRAL_API_KEY non définie - les réponses LLM ne fonctionneront pas")
API_URL = "https://api.mistral.ai/v1/chat/completions"
MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "30"))

# Configuration DB depuis variables d'environnement (OBLIGATOIRE en production)
conn_params = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", "5433")),
    "database": os.getenv("POSTGRES_DB", "support_tech"),
    "user": os.getenv("POSTGRES_USER"),
    "password": os.getenv("POSTGRES_PASSWORD")
}

# Vérification des credentials DB
if not conn_params["user"] or not conn_params["password"]:
    logger.warning("POSTGRES_USER ou POSTGRES_PASSWORD non définis - utilisation des valeurs par défaut (dev uniquement)")
    conn_params["user"] = conn_params["user"] or "bootcamp_user"
    conn_params["password"] = conn_params["password"] or "bootcamp_password"

# Pool de connexions DB
db_pool = None

def get_db_pool():
    global db_pool
    if db_pool is None:
        try:
            db_pool = psycopg2.pool.ThreadedConnectionPool(1, 10, **conn_params)
            logger.info("Pool de connexions DB initialisé")
        except Exception as e:
            logger.error(f"Erreur initialisation pool DB: {e}")
            raise
    return db_pool

@contextmanager
def get_db_connection():
    """Context manager pour les connexions DB avec gestion automatique des ressources"""
    pool = get_db_pool()
    conn = None
    try:
        conn = pool.getconn()
        yield conn
    except psycopg2.Error as e:
        logger.error(f"Erreur DB: {e}")
        raise HTTPException(status_code=503, detail="Service de base de données indisponible")
    finally:
        if conn:
            pool.putconn(conn)

# Lazy loading des modèles
_models = {}

def get_models():
    """Charge les modèles à la demande (lazy loading)"""
    if not _models:
        try:
            logger.info("Chargement des modèles...")
            _models['le_queue'] = joblib.load(BASE_DIR / "le_queue.pkl")
            _models['le_urgency'] = joblib.load(BASE_DIR / "le_urgency.pkl")
            _models['model_queue'] = xgb.Booster()
            _models['model_queue'].load_model(str(BASE_DIR / "xgboost_queue_v2/model.xgb"))
            _models['model_urgency'] = xgb.Booster()
            _models['model_urgency'].load_model(str(BASE_DIR / "xgboost_urgency_v2/model.xgb"))
            _models['embedder'] = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Modèles chargés avec succès")
        except FileNotFoundError as e:
            logger.error(f"Fichier modèle non trouvé: {e}")
            raise HTTPException(status_code=503, detail=f"Modèle non disponible: {e}")
        except Exception as e:
            logger.error(f"Erreur chargement modèles: {e}")
            raise HTTPException(status_code=503, detail="Erreur chargement des modèles")
    return _models

keywords = {
    'network': r'\b(network|wifi|vpn|connect|internet|lan|router)\b',
    'printer': r'\b(printer|imprimante|print|scan|scanner)\b',
    'security': r'\b(security|securite|password|login|access|breach|hack|malware)\b',
    'hardware': r'\b(hardware|laptop|pc|macbook|screen|disk|ssd|cpu)\b',
    'software': r'\b(software|app|update|bug|crash|install|version)\b',
}

def extract_num_features(text: str) -> list:
    """Extrait les features numériques du texte pour la classification"""
    text_lower = text.lower()
    body_len = len(text)
    # Ratio estimé de mots (approximation)
    word_ratio = body_len * 0.8
    # Score de complexité fixe (à améliorer si nécessaire)
    complexity_score = 0.8
    has_feats = [1 if re.search(regex, text_lower) else 0 for regex in keywords.values()]
    return [body_len, word_ratio, complexity_score] + has_feats

def predict_queue_urgency(query_text: str) -> tuple:
    """Prédit la queue et l'urgence pour une requête donnée"""
    models = get_models()
    embedding = models['embedder'].encode(query_text).reshape(1, -1)
    num_feats = extract_num_features(query_text)
    full_vec = np.hstack([embedding, np.array(num_feats).reshape(1, -1)])
    dmatrix = xgb.DMatrix(full_vec)
    pred_queue = models['le_queue'].classes_[np.argmax(models['model_queue'].predict(dmatrix)[0])]
    pred_urgency = models['le_urgency'].classes_[np.argmax(models['model_urgency'].predict(dmatrix)[0])]
    return pred_queue, pred_urgency

def retrieve_rag(query_text: str, predicted_queue: str = None, top_k: int = 5,
                 similarity_threshold: float = 0.4) -> list:
    """
    Récupère les documents RAG pertinents depuis PGVector.

    Utilise un seuil de similarité pour éviter de retourner des documents non pertinents.
    La distance cosinus dans PGVector: 0 = identique, 2 = opposé
    On convertit en score de similarité: 1 - (distance/2)
    """
    models = get_models()
    query_emb = models['embedder'].encode(query_text).tolist()

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # Récupère le contenu ET le score de distance
            sql = "SELECT content, embedding <=> %s::vector as distance FROM rag_docs"
            params = [query_emb]

            if predicted_queue:
                sql += " WHERE metadata->>'refined_queue' = %s"
                params.append(predicted_queue)

            sql += " ORDER BY distance ASC LIMIT %s;"
            params.append(top_k * 2)  # On récupère plus pour filtrer ensuite

            cur.execute(sql, params)
            results = cur.fetchall()

            # Filtre par seuil de similarité
            # Distance cosinus: 0 = identique, 2 = opposé
            # Similarité = 1 - (distance / 2), donc threshold 0.7 = distance max 0.6
            max_distance = (1 - similarity_threshold) * 2

            filtered_results = []
            for content, distance in results:
                if distance <= max_distance:
                    filtered_results.append(content)
                    if len(filtered_results) >= top_k:
                        break

            # Si pas assez de résultats pertinents dans la queue, cherche globalement
            if len(filtered_results) < 2 and predicted_queue:
                logger.info(f"Pas assez de docs pertinents dans {predicted_queue}, recherche globale...")
                sql_global = """
                    SELECT content, embedding <=> %s::vector as distance
                    FROM rag_docs
                    ORDER BY distance ASC LIMIT %s
                """
                cur.execute(sql_global, [query_emb, top_k])
                global_results = cur.fetchall()

                for content, distance in global_results:
                    if distance <= max_distance and content not in filtered_results:
                        filtered_results.append(content)
                        if len(filtered_results) >= top_k:
                            break

            if not filtered_results:
                logger.warning(f"Aucun document RAG pertinent trouvé (seuil: {similarity_threshold})")

            return filtered_results

def generate_response(user_query: str, retrieved_contents: list, pred_queue: str,
                      pred_urgency: str, conversation_history: list = None) -> str:
    """Génère une réponse via l'API Mistral avec le contexte RAG"""

    if not MISTRAL_API_KEY:
        logger.error("Clé API Mistral non configurée")
        return "Erreur: Service de génération de réponse non configuré. Contactez le support."

    # Gestion du cas sans documents pertinents
    if not retrieved_contents:
        context = "[AUCUN DOCUMENT PERTINENT TROUVE DANS LA BASE DE CONNAISSANCES]"
    else:
        context = "\n\n".join(retrieved_contents)

    system_prompt = """Tu es un agent de support technique IT de l'entreprise TechCorp.

COORDONNEES DU SUPPORT:
- Telephone: 01 23 45 67 89
- Email: support@techcorp.fr
- Horaires: Lun-Ven 8h-18h

REGLES STRICTES (A RESPECTER ABSOLUMENT):
1. Reponds TOUJOURS dans la MEME LANGUE que la question
2. Si le CONTEXTE indique "[AUCUN DOCUMENT PERTINENT TROUVE]" ou ne contient PAS d'information pertinente:
   - Donne des conseils de depannage GENERIQUES et BASIQUES pour le type de probleme (VPN, reseau, etc.)
   - Recommande de contacter le support pour une assistance personnalisee
   - NE JAMAIS mentionner de failles de securite, violations de donnees ou incidents graves
3. Si le CONTEXTE contient des informations pertinentes, utilise-les
4. NE JAMAIS inventer de solutions techniques specifiques ou d'informations sensibles
5. Pour les problemes simples (acces VPN, mot de passe, connexion), donne des etapes basiques:
   - Verifier les identifiants
   - Redemarrer l'application/appareil
   - Verifier la connexion internet
   - Contacter le support si ca persiste

FORMAT DE REPONSE:
- Salutation courte (utilise "Bonjour" sans nom)
- Diagnostic rapide
- Solutions en etapes numerotees (basiques et securisees)
- Contact support si necessaire
- Conclusion positive

NE JAMAIS utiliser de placeholders comme [Name], [tel_num], etc. Utilise les vraies coordonnees ci-dessus.
NE JAMAIS mentionner de failles de securite, breaches, ou violations de donnees pour des problemes d'acces simples."""

    # Format Mistral (OpenAI-like): system en premier message
    messages = [{"role": "system", "content": system_prompt}]

    # Ajouter l'historique de conversation si présent (avec validation)
    if conversation_history:
        for msg in conversation_history:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                messages.append({"role": msg["role"], "content": msg["content"]})

    # Ajouter la nouvelle question avec contexte RAG
    messages.append({"role": "user", "content": f"Categorie:{pred_queue}\nUrgence:{pred_urgency}\nContexte:\n{context}\n\nQuestion:\n{user_query}"})

    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 512
    }
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(API_URL, json=payload, headers=headers, timeout=API_TIMEOUT)
        response.raise_for_status()

        data = response.json()
        if 'choices' in data and len(data['choices']) > 0 and 'message' in data['choices'][0]:
            return data['choices'][0]['message']['content']
        else:
            logger.error(f"Réponse API malformée: {data}")
            return "Erreur: Réponse du service invalide. Veuillez réessayer."

    except requests.Timeout:
        logger.error("Timeout appel API Mistral")
        return "Erreur: Le service met trop de temps à répondre. Veuillez réessayer."
    except requests.RequestException as e:
        logger.error(f"Erreur appel API Mistral: {e}")
        return f"Erreur: Service temporairement indisponible. Contactez le support au 01 23 45 67 89."

app = FastAPI(title="Agent Support IT - MLOps Bootcamp")

class Query(BaseModel):
    user_query: str = Field(..., min_length=1, max_length=5000, description="Question de l'utilisateur")
    conversation_history: Optional[list] = None  # Liste de {"role": "user/assistant", "content": "..."}

    @field_validator('user_query')
    @classmethod
    def validate_query(cls, v):
        if not v or not v.strip():
            raise ValueError("La question ne peut pas être vide")
        return v.strip()

# Queues et urgences valides
VALID_QUEUES = ["network", "hardware", "software", "security", "printer", "other"]
VALID_URGENCIES = ["low", "medium", "high", "critical"]

class Feedback(BaseModel):
    prediction_id: int = Field(..., gt=0, description="ID de la prédiction")
    correct_queue: Optional[str] = Field(None, description="Queue correcte si différente")
    correct_urgency: Optional[str] = Field(None, description="Urgence correcte si différente")

    @field_validator('correct_queue')
    @classmethod
    def validate_queue(cls, v):
        if v is not None and v not in VALID_QUEUES:
            raise ValueError(f"Queue invalide. Valeurs acceptées: {VALID_QUEUES}")
        return v

    @field_validator('correct_urgency')
    @classmethod
    def validate_urgency(cls, v):
        if v is not None and v not in VALID_URGENCIES:
            raise ValueError(f"Urgence invalide. Valeurs acceptées: {VALID_URGENCIES}")
        return v

def log_prediction(input_text: str, pred_queue: str, pred_urgency: str,
                   response: str = None, language: str = None,
                   conf_queue: float = None, conf_urgency: float = None) -> Optional[int]:
    """Enregistre une prediction dans la base pour monitoring et retraining"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO prediction_logs
                    (input_text, predicted_queue, predicted_urgency, response, language, confidence_queue, confidence_urgency)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (input_text, pred_queue, pred_urgency, response, language, conf_queue, conf_urgency))
                prediction_id = cur.fetchone()[0]
                conn.commit()
                return prediction_id
    except Exception as e:
        logger.error(f"Erreur logging prediction: {e}")
        return None

@app.post("/predict")
def agent(query: Query):
    pred_queue, pred_urgency = predict_queue_urgency(query.user_query)

    # Construire la requête RAG avec contexte si historique existe
    if query.conversation_history:
        # Valider le format de conversation_history (doit être liste de dicts avec role/content)
        try:
            valid_history = [m for m in query.conversation_history if isinstance(m, dict) and "role" in m and "content" in m]
            first_user_msg = next((m["content"] for m in valid_history if m["role"] == "user"), "")
            rag_query = f"{first_user_msg} {query.user_query}"
        except (TypeError, KeyError):
            rag_query = query.user_query
    else:
        rag_query = query.user_query

    retrieved = retrieve_rag(rag_query, predicted_queue=pred_queue, top_k=5)

    # Mistral reçoit: historique + docs RAG + nouveau message
    response = generate_response(query.user_query, retrieved, pred_queue, pred_urgency, query.conversation_history)

    # Log la prediction pour monitoring et retraining (avec réponse pour feedback loop complet)
    prediction_id = log_prediction(query.user_query, pred_queue, pred_urgency, response=response, language='en')

    return {
        "prediction_id": prediction_id,
        "query": query.user_query,
        "predicted_queue": pred_queue,
        "predicted_urgency": pred_urgency,
        "response": response,
        "rag_sources": retrieved[:3]  # Sources complètes pour vérification
    }

@app.post("/feedback")
def submit_feedback(feedback: Feedback):
    """Enregistre le feedback utilisateur pour ameliorer le modele"""
    if feedback.correct_queue is None and feedback.correct_urgency is None:
        raise HTTPException(status_code=400, detail="Au moins correct_queue ou correct_urgency doit être fourni")

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Vérifier que la prediction existe
                cur.execute("SELECT id FROM prediction_logs WHERE id = %s", (feedback.prediction_id,))
                if cur.fetchone() is None:
                    raise HTTPException(status_code=404, detail=f"Prediction {feedback.prediction_id} non trouvée")

                cur.execute("""
                    UPDATE prediction_logs
                    SET feedback_queue = %s, feedback_urgency = %s, feedback_at = %s
                    WHERE id = %s
                """, (feedback.correct_queue, feedback.correct_urgency, datetime.now(), feedback.prediction_id))
                conn.commit()
                return {"status": "success", "message": "Feedback enregistré"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur feedback: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de l'enregistrement du feedback")

@app.get("/health")
def health():
    """Vérifie l'état de santé de l'application"""
    health_status = {
        "status": "ok",
        "components": {}
    }

    # Vérifier la connexion DB
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        health_status["components"]["database"] = "ok"
    except Exception as e:
        health_status["status"] = "degraded"
        health_status["components"]["database"] = f"error: {str(e)}"

    # Vérifier que les modèles peuvent être chargés
    try:
        get_models()
        health_status["components"]["models"] = "ok"
    except Exception as e:
        health_status["status"] = "degraded"
        health_status["components"]["models"] = f"error: {str(e)}"

    # Vérifier la clé API
    if MISTRAL_API_KEY:
        health_status["components"]["mistral_api"] = "configured"
    else:
        health_status["status"] = "degraded"
        health_status["components"]["mistral_api"] = "not configured"

    return health_status

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
