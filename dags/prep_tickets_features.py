from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
import pandas as pd
from sqlalchemy import create_engine
import re
import logging
from evidently import Report
from evidently.presets import DataSummaryPreset, DataDriftPreset

default_args = {
    'owner': 'jedha_bootcamp',
    'depends_on_past': False,
    'retries': 1,
}

dag = DAG(
    'prep_tickets_features',
    default_args=default_args,
    description='Feature engineering on technical support tickets',
    schedule_interval=None,  # manual pour l'instant
    start_date=datetime(2025, 12, 1),
    catchup=False,
    tags=['preparation', 'features', 'postgres'],
)

def prep_features():
    # Connexion Postgres (bootcamp_postgres sur port 5433)
    engine = create_engine('postgresql://bootcamp_user:bootcamp_password@host.docker.internal:5433/support_tech')

    logging.info("Lecture depuis tickets_tech_en...")
    query = "SELECT * FROM tickets_tech_en"
    df = pd.read_sql(query, engine)

    # 1. Nettoyage léger
    def clean_text(text):
        if pd.isna(text):
            return ""
        text = re.sub(r'<.*?>', '', text)               # supprime HTML
        text = re.sub(r'\s+', ' ', text)                # espaces multiples
        text = text.strip()
        # Optionnel : anonymisation plus poussée
        text = re.sub(r'\d{3}-\d{3}-\d{4}', '<tel>', text)  # numéros US
        return text

    df['body_clean'] = df['body'].apply(clean_text)
    df['answer_clean'] = df['answer'].apply(clean_text)

    # 2. Features texte
    df['body_length'] = df['body_clean'].str.len()
    df['answer_length'] = df['answer_clean'].str.len()
    df['response_ratio'] = df['answer_length'] / (df['body_length'] + 1)  # évite /0

    # 3. Keywords pour sous-classes sujet (exemples simples, tu peux étendre)
    keywords = {
        'network':     r'\b(network|wifi|vpn|connect|internet|lan|router)\b',
        'printer':     r'\b(printer|imprimante|print|scan|scanner)\b',
        'security':    r'\b(security|securite|password|login|access|breach|hack|malware)\b',
        'hardware':    r'\b(hardware|laptop|pc|macbook|screen|disk|ssd|cpu)\b',
        'software':    r'\b(software|app|update|bug|crash|install|version)\b',
    }

    for k, regex in keywords.items():
        df[f'has_{k}'] = df['body_clean'].str.contains(regex, case=False, na=False).astype(int)

    # 4. Sauvegarde dans nouvelle table
    logging.info(f"Enrichissement terminé : {len(df)} lignes. Sauvegarde...")
    df.to_sql(
        name='tickets_tech_en_enriched',
        con=engine,
        if_exists='replace',
        index=False,
        method='multi',
        chunksize=1000
    )
    logging.info("Table tickets_tech_en_enriched créée !")

    # 5. Rapport Evidently (Data Quality + Data Drift)
    logging.info("Génération du rapport Evidently...")

    # Sélectionne uniquement les colonnes numériques pour Evidently
    numeric_cols = ['body_length', 'answer_length', 'response_ratio',
                    'has_network', 'has_printer', 'has_security', 'has_hardware', 'has_software']
    df_numeric = df[numeric_cols].copy()

    # Data Summary Report (qualité des données)
    quality_report = Report([DataSummaryPreset()])
    quality_result = quality_report.run(current_data=df_numeric, reference_data=None)
    quality_result.save_html("/opt/airflow/data/evidently_quality_report.html")
    logging.info("Rapport Data Summary sauvegardé !")

    # Data Drift Report (compare première moitié vs seconde moitié)
    drift_report = Report([DataDriftPreset()])
    drift_result = drift_report.run(reference_data=df_numeric.head(9000), current_data=df_numeric.tail(8893))
    drift_result.save_html("/opt/airflow/data/evidently_drift_report.html")
    logging.info("Rapport Data Drift sauvegardé !")

prep_task = PythonOperator(
    task_id='prepare_features',
    python_callable=prep_features,
    dag=dag,
)

prep_task
