from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
import pandas as pd
from sqlalchemy import create_engine
import logging

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
    start_date=datetime(2025, 12, 1),
    catchup=False,
    tags=['ingestion', 'postgres', 'tickets'],
)

def ingest_to_postgres():
    # Chemin vers ton parquet filtré
    parquet_path = '/opt/airflow/data/train_tech_en.parquet'

    # Connexion Postgres (bootcamp_postgres sur port 5433)
    engine = create_engine('postgresql://bootcamp_user:bootcamp_password@host.docker.internal:5433/support_tech')

    logging.info("Lecture du parquet...")
    df = pd.read_parquet(parquet_path)

    logging.info(f"Ingestion de {len(df)} lignes dans tickets_tech_en...")
    df.to_sql(
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
