from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sentence_transformers import SentenceTransformer
import mlflow
import mlflow.xgboost
import logging
import joblib
import os
import sys
sys.path.insert(0, '/opt/airflow/dags')
from utils.db_config import get_db_engine, get_mlflow_tracking_uri

default_args = {
    'owner': 'jedha_bootcamp',
    'depends_on_past': False,
    'retries': 1,
}

dag = DAG(
    'classification_xgboost_v2_embeddings',
    default_args=default_args,
    description='Classification V2: Embeddings + Sample Weight (improved)',
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['classification', 'xgboost', 'mlflow', 'embeddings', 'v2'],
)

def train_and_log_model_v2():
    # Connexion Postgres depuis configuration
    engine = get_db_engine()

    logging.info("Lecture des données enrichies...")
    df = pd.read_sql("SELECT * FROM tickets_tech_en_enriched", engine)

    # Features text + numériques
    df['text_combined'] = df['subject'].fillna('') + " " + df['body_clean'].fillna('')

    num_features = ['body_length', 'answer_length', 'response_ratio',
                    'has_network', 'has_printer', 'has_security', 'has_hardware', 'has_software']

    # Encodage targets
    le_queue = LabelEncoder()
    le_urgency = LabelEncoder()
    df['queue_encoded'] = le_queue.fit_transform(df['refined_queue'])
    df['urgency_encoded'] = le_urgency.fit_transform(df['urgency_level'])

    # Split stratifié
    X = df[['text_combined'] + num_features]
    y_queue = df['queue_encoded']
    y_urgency = df['urgency_encoded']

    X_train, X_test, yq_train, yq_test, yu_train, yu_test = train_test_split(
        X, y_queue, y_urgency, test_size=0.15, random_state=42,
        stratify=df[['refined_queue', 'urgency_level']]
    )

    X_train, X_val, yq_train, yq_val, yu_train, yu_val = train_test_split(
        X_train, yq_train, yu_train, test_size=0.15, random_state=42,
        stratify=pd.DataFrame({'queue': yq_train, 'urgency': yu_train})
    )

    logging.info(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # ============================================
    # AMÉLIORATION 1: Embeddings au lieu de TF-IDF
    # ============================================
    logging.info("Chargement du modèle SentenceTransformer (all-MiniLM-L6-v2)...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    logging.info("Génération des embeddings pour train...")
    X_train_embed = embedder.encode(X_train['text_combined'].tolist(), show_progress_bar=True)
    logging.info("Génération des embeddings pour val...")
    X_val_embed = embedder.encode(X_val['text_combined'].tolist(), show_progress_bar=True)
    logging.info("Génération des embeddings pour test...")
    X_test_embed = embedder.encode(X_test['text_combined'].tolist(), show_progress_bar=True)

    # Combine embeddings + num features
    X_train_full = np.hstack([X_train_embed, X_train[num_features].values])
    X_val_full = np.hstack([X_val_embed, X_val[num_features].values])
    X_test_full = np.hstack([X_test_embed, X_test[num_features].values])

    logging.info(f"Shape features: {X_train_full.shape} (384 embed + 8 num = 392)")

    # ============================================
    # AMÉLIORATION 2: Sample weights pour urgency
    # ============================================
    sample_weights_urgency = compute_sample_weight('balanced', yu_train)
    logging.info("Sample weights calculés pour urgency (classe 'low' boostée)")

    # MLflow setup
    mlflow.set_tracking_uri(get_mlflow_tracking_uri())
    mlflow.set_experiment("tickets_classification_bootcamp")

    with mlflow.start_run(run_name="xgboost_v2_embeddings") as run:
        # Params
        params = {
            'objective': 'multi:softprob',
            'num_class': len(le_queue.classes_),
            'eval_metric': 'mlogloss',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'random_state': 42,
        }
        mlflow.log_params(params)
        mlflow.log_param("embedding_model", "all-MiniLM-L6-v2")
        mlflow.log_param("embedding_dim", 384)
        mlflow.log_param("urgency_sample_weight", "balanced")

        # Modèle pour refined_queue (6 classes)
        logging.info("Entraînement modèle Queue (6 classes) avec embeddings...")
        model_queue = XGBClassifier(**params)
        model_queue.fit(X_train_full, yq_train, eval_set=[(X_val_full, yq_val)], verbose=False)

        # Prédictions queue
        pred_queue_test = model_queue.predict(X_test_full)

        # Metrics queue
        acc_queue = accuracy_score(yq_test, pred_queue_test)
        f1_queue = f1_score(yq_test, pred_queue_test, average='weighted')
        mlflow.log_metric("accuracy_queue", acc_queue)
        mlflow.log_metric("f1_queue", f1_queue)

        report_queue = classification_report(yq_test, pred_queue_test, target_names=le_queue.classes_)
        logging.info(f"=== Classification Report Queue ===\n{report_queue}")
        with open("classification_report_queue.txt", "w") as f:
            f.write(report_queue)
        mlflow.log_artifact("classification_report_queue.txt")
        os.remove("classification_report_queue.txt")

        # Modèle pour urgency_level (3 classes) AVEC sample_weight
        logging.info("Entraînement modèle Urgency (3 classes) avec sample_weight...")
        params_urgency = params.copy()
        params_urgency['num_class'] = 3
        model_urgency = XGBClassifier(**params_urgency)

        # FIT avec sample_weight pour booster les classes minoritaires
        model_urgency.fit(
            X_train_full, yu_train,
            sample_weight=sample_weights_urgency,  # <-- AMÉLIORATION
            eval_set=[(X_val_full, yu_val)],
            verbose=False
        )

        pred_urgency_test = model_urgency.predict(X_test_full)
        acc_urgency = accuracy_score(yu_test, pred_urgency_test)
        f1_urgency = f1_score(yu_test, pred_urgency_test, average='weighted')
        mlflow.log_metric("accuracy_urgency", acc_urgency)
        mlflow.log_metric("f1_urgency", f1_urgency)

        report_urgency = classification_report(yu_test, pred_urgency_test, target_names=le_urgency.classes_)
        logging.info(f"=== Classification Report Urgency ===\n{report_urgency}")
        with open("classification_report_urgency.txt", "w") as f:
            f.write(report_urgency)
        mlflow.log_artifact("classification_report_urgency.txt")
        os.remove("classification_report_urgency.txt")

        # Log models dans MLflow
        mlflow.xgboost.log_model(model_queue, "xgboost_queue_v2")
        mlflow.xgboost.log_model(model_urgency, "xgboost_urgency_v2")

        # Sauvegarde artifacts dans MLflow
        # On sauvegarde l'embedder info (pas le modèle lui-même car trop gros)
        with open("embedder_info.txt", "w") as f:
            f.write("Model: sentence-transformers/all-MiniLM-L6-v2\n")
            f.write("Dimension: 384\n")
            f.write("Usage: embedder = SentenceTransformer('all-MiniLM-L6-v2')\n")
        mlflow.log_artifact("embedder_info.txt")
        os.remove("embedder_info.txt")

        joblib.dump(le_queue, "le_queue.pkl")
        joblib.dump(le_urgency, "le_urgency.pkl")
        mlflow.log_artifact("le_queue.pkl")
        mlflow.log_artifact("le_urgency.pkl")
        os.remove("le_queue.pkl")
        os.remove("le_urgency.pkl")

        logging.info(f"=== RESULTATS FINAUX V2 ===")
        logging.info(f"Queue     - Accuracy: {acc_queue:.4f} | F1: {f1_queue:.4f}")
        logging.info(f"Urgency   - Accuracy: {acc_urgency:.4f} | F1: {f1_urgency:.4f}")
        logging.info("Entraînement V2 terminé et loggé dans MLflow !")

train_task = PythonOperator(
    task_id='train_xgboost_v2_embeddings',
    python_callable=train_and_log_model_v2,
    dag=dag,
)

train_task
