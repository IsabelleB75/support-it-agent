-- schema.sql - Structure de la base de donnees PostgreSQL
-- Projet MLOps Support IT - Jedha Bootcamp

-- Table principale des tickets (donnees d'entrainement)
CREATE TABLE IF NOT EXISTS tickets_tech_en_enriched (
    id SERIAL PRIMARY KEY,
    subject TEXT,
    body_clean TEXT,
    answer_clean TEXT,
    refined_queue VARCHAR(100),
    urgency_level VARCHAR(50),
    queue VARCHAR(100),
    language VARCHAR(10),
    body_length INTEGER,
    answer_length INTEGER,
    response_ratio FLOAT,
    has_network INTEGER DEFAULT 0,
    has_printer INTEGER DEFAULT 0,
    has_security INTEGER DEFAULT 0,
    has_hardware INTEGER DEFAULT 0,
    has_software INTEGER DEFAULT 0
);

-- Table des logs de predictions (alimentee par l'API FastAPI)
CREATE TABLE IF NOT EXISTS prediction_logs (
    id SERIAL PRIMARY KEY,
    input_text TEXT NOT NULL,
    predicted_queue VARCHAR(100),
    predicted_urgency VARCHAR(50),
    confidence_queue FLOAT,
    confidence_urgency FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    feedback_queue VARCHAR(100),
    feedback_urgency VARCHAR(50),
    feedback_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_prediction_logs_created_at ON prediction_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_prediction_logs_feedback ON prediction_logs(feedback_queue) WHERE feedback_queue IS NOT NULL;

-- Table des logs de drift (alimentee par le DAG monitoring)
CREATE TABLE IF NOT EXISTS drift_logs (
    id SERIAL PRIMARY KEY,
    check_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    drift_detected BOOLEAN,
    drift_share FLOAT,
    report_path TEXT,
    UNIQUE(check_date)
);

-- Table RAG pour les embeddings (pgvector)
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS rag_docs (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding vector(384),
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_rag_embedding ON rag_docs USING ivfflat (embedding vector_cosine_ops);
