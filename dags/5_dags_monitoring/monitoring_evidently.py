from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
import pandas as pd
from sqlalchemy import text
from evidently import Report
from evidently.presets import DataDriftPreset
import logging
import json
import os
import sys
sys.path.insert(0, '/opt/airflow/dags')
from utils.db_config import get_db_engine

default_args = {
    'owner': 'jedha_bootcamp',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'monitoring_evidently_drift',
    default_args=default_args,
    description='Monitoring drift avec Evidently - declenche retraining si drift detecte',
    schedule_interval='0 6 * * *',  # Tous les jours a 6h
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['monitoring', 'evidently', 'drift', 'mlops'],
)

def check_drift(**context):
    """Verifie le drift entre donnees reference et production"""
    engine = get_db_engine()

    logging.info("Chargement des donnees de reference (train)...")
    # Donnees de reference = train set original
    df_reference = pd.read_sql("""
        SELECT subject, body_clean, refined_queue, urgency_level,
               body_length, answer_length, response_ratio,
               has_network, has_printer, has_security, has_hardware, has_software
        FROM tickets_tech_en_enriched
        ORDER BY RANDOM()
        LIMIT 5000
    """, engine)

    logging.info("Chargement des donnees de production (logs recents)...")
    # Donnees de production = logs des 7 derniers jours
    try:
        df_production = pd.read_sql("""
            SELECT input_text as subject, input_text as body_clean,
                   predicted_queue as refined_queue, predicted_urgency as urgency_level,
                   LENGTH(input_text) as body_length,
                   LENGTH(input_text) as answer_length,
                   1.0 as response_ratio,
                   CASE WHEN input_text ILIKE '%network%' OR input_text ILIKE '%vpn%' THEN 1 ELSE 0 END as has_network,
                   CASE WHEN input_text ILIKE '%printer%' THEN 1 ELSE 0 END as has_printer,
                   CASE WHEN input_text ILIKE '%security%' OR input_text ILIKE '%password%' THEN 1 ELSE 0 END as has_security,
                   CASE WHEN input_text ILIKE '%hardware%' OR input_text ILIKE '%laptop%' THEN 1 ELSE 0 END as has_hardware,
                   CASE WHEN input_text ILIKE '%software%' OR input_text ILIKE '%install%' THEN 1 ELSE 0 END as has_software
            FROM prediction_logs
            WHERE created_at >= NOW() - INTERVAL '7 days'
            LIMIT 1000
        """, engine)
    except Exception as e:
        logging.warning(f"Table prediction_logs non trouvee: {e}")
        logging.info("Creation de donnees de production simulees...")
        # Simulation si pas de logs production
        df_production = df_reference.sample(n=min(500, len(df_reference)), random_state=42)
        # Ajoute un leger drift simule
        df_production['body_length'] = df_production['body_length'] * 1.2

    if len(df_production) < 100:
        logging.warning("Pas assez de donnees production pour analyse drift")
        context['ti'].xcom_push(key='drift_detected', value=False)
        context['ti'].xcom_push(key='drift_score', value=0.0)
        return

    logging.info(f"Reference: {len(df_reference)} samples | Production: {len(df_production)} samples")

    # Colonnes numeriques pour le drift
    num_cols = ['body_length', 'answer_length', 'response_ratio',
                'has_network', 'has_printer', 'has_security', 'has_hardware', 'has_software']

    # Prepare les DataFrames
    df_ref_num = df_reference[num_cols].fillna(0)
    df_prod_num = df_production[num_cols].fillna(0)

    # Rapport Evidently (API v0.7.x)
    logging.info("Generation du rapport Evidently...")
    report = Report(metrics=[
        DataDriftPreset(),
    ])

    snapshot = report.run(reference_data=df_ref_num, current_data=df_prod_num)

    # Extrait les resultats (API v0.7.x: snapshot.dict())
    result_dict = snapshot.dict()

    # Trouve le metric DriftedColumnsCount pour le drift share
    drift_share = 0.0
    for m in result_dict.get('metrics', []):
        if 'DriftedColumnsCount' in str(m.get('metric_id', '')):
            value = m.get('value', {})
            drift_share = value.get('share', 0.0)
            break

    # Dataset drift si plus de 50% des colonnes ont drift
    dataset_drift = drift_share > 0.5

    logging.info(f"=== RESULTATS DRIFT ===")
    logging.info(f"Dataset Drift Detecte: {dataset_drift}")
    logging.info(f"Proportion de features en drift: {drift_share:.2%}")

    # Sauvegarde le rapport HTML (dans /tmp accessible en écriture)
    report_path = "/tmp/evidently_reports"
    os.makedirs(report_path, exist_ok=True)
    report_file = f"{report_path}/drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    snapshot.save_html(report_file)
    logging.info(f"Rapport sauvegarde: {report_file}")

    # Push resultats pour decision
    context['ti'].xcom_push(key='drift_detected', value=dataset_drift)
    context['ti'].xcom_push(key='drift_score', value=drift_share)

    # Log dans la base
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO drift_logs (check_date, drift_detected, drift_share, report_path)
            VALUES (:check_date, :drift_detected, :drift_share, :report_path)
            ON CONFLICT DO NOTHING
        """), {
            'check_date': datetime.now(),
            'drift_detected': dataset_drift,
            'drift_share': drift_share,
            'report_path': report_file
        })

def decide_retraining(**context):
    """Decide si le retraining doit etre declenche - retourne le task_id a executer"""
    drift_detected = context['ti'].xcom_pull(key='drift_detected', task_ids='check_drift')
    drift_score = context['ti'].xcom_pull(key='drift_score', task_ids='check_drift')

    logging.info(f"Drift detecte: {drift_detected} | Score: {drift_score}")

    # Seuil: si plus de 5% des features ont drift, on retraine (abaisse pour tests)
    DRIFT_THRESHOLD = 0.05

    if drift_detected or (drift_score and drift_score > DRIFT_THRESHOLD):
        logging.info(f"ALERTE: Drift significant detecte ({drift_score:.2%} > {DRIFT_THRESHOLD:.2%})")
        logging.info("Declenchement de l'optimisation hyperparametres puis retraining...")
        return 'trigger_hyperparameter_tuning'  # Task ID a executer
    else:
        logging.info("Pas de drift significant, modele stable")
        return 'no_retraining'  # Task ID a executer

def no_retraining_needed():
    """Task finale si pas de retraining"""
    logging.info("Monitoring termine - Pas de retraining necessaire")

check_drift_task = PythonOperator(
    task_id='check_drift',
    python_callable=check_drift,
    provide_context=True,
    dag=dag,
)

decide_task = BranchPythonOperator(
    task_id='decide_retraining',
    python_callable=decide_retraining,
    provide_context=True,
    dag=dag,
)

trigger_hyperparams = TriggerDagRunOperator(
    task_id='trigger_hyperparameter_tuning',
    trigger_dag_id='hyperparameter_tuning_ray_tune',
    dag=dag,
)

no_retrain_task = PythonOperator(
    task_id='no_retraining',
    python_callable=no_retraining_needed,
    dag=dag,
)

check_drift_task >> decide_task
decide_task >> [trigger_hyperparams, no_retrain_task]
