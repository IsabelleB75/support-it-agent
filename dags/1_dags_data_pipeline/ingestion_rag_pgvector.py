from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
import pandas as pd
from sqlalchemy import text
from sentence_transformers import SentenceTransformer
import json
import logging
from sklearn.model_selection import train_test_split
import numpy as np
import sys
sys.path.insert(0, '/opt/airflow/dags')
from utils.db_config import get_db_engine

default_args = {
    'owner': 'jedha_bootcamp',
    'depends_on_past': False,
    'retries': 1,
}

dag = DAG(
    'ingestion_rag_pgvector',
    default_args=default_args,
    description='Ingestion embeddings RAG dans PGVector (sur train split)',
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['rag', 'pgvector', 'embeddings', 'tickets'],
)

def ingest_rag_to_pgvector():
    # Connexion Postgres depuis configuration
    engine = get_db_engine()

    logging.info("Lecture des données enrichies...")
    df = pd.read_sql("SELECT * FROM tickets_tech_en_enriched", engine)
    logging.info(f"Total tickets: {len(df)}")

    # Split train (on utilise seulement train pour l'index RAG -> évite leakage)
    train, _ = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        stratify=df[['queue', 'urgency_level']]
    )
    logging.info(f"Utilisation de {len(train)} tickets pour l'index RAG (train split)")

    # Modèle embeddings (léger, ~80 MB, rapide)
    logging.info("Chargement du modèle SentenceTransformer (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    def ticket_to_rag_row(row, idx):
        # Combine problème + réponse pour le contenu RAG
        subject = row['subject'] if pd.notna(row['subject']) else ''
        body = row['body_clean'] if pd.notna(row['body_clean']) else ''
        answer = row['answer_clean'] if pd.notna(row['answer_clean']) else ''

        content = f"""Problème :
{subject}
{body[:1000]}

Réponse agent typique :
{answer[:800]}""".strip()

        metadata = {
            "refined_queue": row['refined_queue'],
            "urgency_level": row['urgency_level'],
            "language": row['language'] if pd.notna(row['language']) else 'en'
        }

        return {
            "id": f"ticket_{idx}",
            "title": f"{row['refined_queue']} - {subject[:80]}",
            "content": content,
            "metadata": json.dumps(metadata)
        }

    # Génère les rows (sans embeddings pour l'instant)
    logging.info("Préparation des documents RAG...")
    rag_rows = [ticket_to_rag_row(row, idx) for idx, (_, row) in enumerate(train.iterrows())]

    # Génère les embeddings en batch (plus efficace)
    logging.info("Génération des embeddings en batch...")
    contents = [r['content'] for r in rag_rows]
    embeddings = model.encode(contents, show_progress_bar=True, batch_size=64)

    # Ajoute les embeddings aux rows
    for i, emb in enumerate(embeddings):
        rag_rows[i]['embedding'] = emb.tolist()

    logging.info(f"Génération de {len(rag_rows)} embeddings terminée.")

    # Ingestion en batch avec SQL brut (pgvector nécessite format spécial)
    logging.info("Ingestion dans PGVector...")

    with engine.begin() as conn:
        # Vide la table existante
        conn.execute(text("TRUNCATE TABLE rag_docs"))

        # Insert par batch
        batch_size = 100
        for i in range(0, len(rag_rows), batch_size):
            batch = rag_rows[i:i+batch_size]

            for row in batch:
                embedding_str = '[' + ','.join(map(str, row['embedding'])) + ']'

                # Escape les quotes dans content et title
                safe_id = row['id'].replace("'", "''")
                safe_title = row['title'].replace("'", "''")
                safe_content = row['content'].replace("'", "''")
                safe_metadata = row['metadata'].replace("'", "''")

                sql = f"""
                    INSERT INTO rag_docs (id, title, content, embedding, metadata)
                    VALUES ('{safe_id}', '{safe_title}', '{safe_content}', '{embedding_str}'::vector, '{safe_metadata}'::jsonb)
                """
                conn.execute(text(sql))

            logging.info(f"Batch {i//batch_size + 1}/{(len(rag_rows)//batch_size)+1} inséré")

    logging.info("Ingestion RAG dans PGVector terminée !")

    # Vérification finale
    with engine.begin() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM rag_docs"))
        count = result.scalar()
        logging.info(f"Total documents RAG indexés: {count}")

ingest_rag_task = PythonOperator(
    task_id='ingest_embeddings_rag',
    python_callable=ingest_rag_to_pgvector,
    dag=dag,
)

ingest_rag_task
