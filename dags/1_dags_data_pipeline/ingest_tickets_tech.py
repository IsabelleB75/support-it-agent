from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
import pandas as pd
import logging
from langdetect import detect, LangDetectException
import sys
sys.path.insert(0, '/opt/airflow/dags')
from utils.db_config import get_db_engine

default_args = {
    'owner': 'jedha_bootcamp',
    'depends_on_past': False,
    'retries': 1,
}

dag = DAG(
    'ingest_tickets_tech_en',
    default_args=default_args,
    description='Ingest filtered technical support tickets into Postgres',
    schedule_interval=None,  # manual trigger pour l'instant
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['ingestion', 'postgres', 'tickets'],
)

def detect_language(text):
    """Détecte la langue réelle du texte (pas le metadata)"""
    try:
        if pd.isna(text) or len(str(text).strip()) < 20:
            return 'unknown'
        return detect(str(text)[:500])  # 500 chars suffisent pour détecter
    except LangDetectException:
        return 'unknown'

def ingest_to_postgres():
    # Chemin vers ton parquet filtré
    parquet_path = '/opt/airflow/data/train_tech_en.parquet'

    # Connexion Postgres depuis configuration
    engine = get_db_engine()

    logging.info("Lecture du parquet...")
    df = pd.read_parquet(parquet_path)
    initial_count = len(df)

    # Détection de langue réelle sur le body
    logging.info("Détection de langue en cours...")
    df['detected_lang'] = df['body'].apply(detect_language)

    # Stats avant filtrage
    lang_stats = df['detected_lang'].value_counts()
    logging.info(f"Langues détectées:\n{lang_stats}")

    # Garde uniquement l'anglais confirmé
    df_clean = df[df['detected_lang'] == 'en'].copy()
    df_clean = df_clean.drop(columns=['detected_lang'])  # Pas besoin de stocker

    filtered_count = initial_count - len(df_clean)
    logging.info(f"Filtrage langue: {initial_count} → {len(df_clean)} tickets ({filtered_count} non-anglais supprimés)")

    logging.info(f"Ingestion de {len(df_clean)} lignes dans tickets_tech_en...")
    df_clean.to_sql(
        name='tickets_tech_en',
        con=engine,
        if_exists='replace',  # ou 'append' si tu veux accumuler
        index=False,
        method='multi',       # plus rapide pour gros volumes
        chunksize=1000
    )
    logging.info("Ingestion terminée !")

ingest_task = PythonOperator(
    task_id='ingest_tickets_to_postgres',
    python_callable=ingest_to_postgres,
    dag=dag,
)

ingest_task
