from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
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
import json
import os

default_args = {
    'owner': 'jedha_bootcamp',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'retraining_pipeline',
    default_args=default_args,
    description='Pipeline de retraining automatique declenche par drift',
    schedule_interval=None,  # Declenche par monitoring DAG
    start_date=datetime(2025, 12, 1),
    catchup=False,
    tags=['retraining', 'mlflow', 'xgboost', 'mlops'],
)

def load_data(**context):
    """Charge les donnees enrichies + nouveaux logs production"""
    import subprocess
    import glob

    engine = create_engine('postgresql://bootcamp_user:bootcamp_password@host.docker.internal:5433/support_tech')
    PROJECT_PATH = "/opt/airflow/project"
    last_training_date = None

    # 1. Cherche le dataset le plus récent versionné par DVC
    try:
        # Pull les dernières versions depuis DVC
        subprocess.run(["dvc", "pull"], cwd=PROJECT_PATH, check=False)

        # Trouve le fichier le plus récent
        dataset_files = sorted(glob.glob(f"{PROJECT_PATH}/data/training_data_*.parquet"))
        if dataset_files:
            latest_dataset = dataset_files[-1]
            logging.info(f"Chargement du dataset DVC le plus recent: {latest_dataset}")
            df = pd.read_parquet(latest_dataset)
            logging.info(f"Dataset DVC charge: {len(df)} tickets")

            # Extrait la date du dernier training pour filtrer les nouveaux feedbacks
            filename = os.path.basename(latest_dataset)
            # Format: training_data_YYYYMMDD_HHMMSS.parquet
            date_str = filename.replace("training_data_", "").replace(".parquet", "")
            last_training_date = datetime.strptime(date_str, '%Y%m%d_%H%M%S')
        else:
            raise FileNotFoundError("Aucun dataset DVC trouve")
    except Exception as e:
        logging.info(f"Pas de dataset DVC disponible ({e}), chargement depuis PostgreSQL...")
        df = pd.read_sql("SELECT * FROM tickets_tech_en_enriched", engine)
        logging.info(f"Donnees historiques: {len(df)} tickets")

    # Ajoute les logs de production recents (si feedback disponible)
    try:
        # Si on a un dataset DVC, on charge seulement les feedbacks DEPUIS le dernier training
        # Sinon, on prend les 30 derniers jours
        if last_training_date:
            date_filter = f"created_at >= '{last_training_date.strftime('%Y-%m-%d %H:%M:%S')}'"
            logging.info(f"Chargement feedbacks depuis: {last_training_date}")
        else:
            date_filter = "created_at >= NOW() - INTERVAL '30 days'"
            logging.info("Chargement feedbacks des 30 derniers jours")

        df_logs = pd.read_sql(f"""
            SELECT input_text as subject, input_text as body_clean,
                   feedback_queue as refined_queue, feedback_urgency as urgency_level,
                   LENGTH(input_text) as body_length, 0 as answer_length, 1.0 as response_ratio,
                   '' as answer_clean, 'en' as language, '' as queue
            FROM prediction_logs
            WHERE feedback_queue IS NOT NULL
              AND {date_filter}
        """, engine)
        if len(df_logs) > 0:
            logging.info(f"Ajout de {len(df_logs)} feedbacks production")
            # Ajoute les colonnes manquantes avec valeurs par defaut
            for col in df.columns:
                if col not in df_logs.columns:
                    df_logs[col] = 0 if df[col].dtype in ['int64', 'float64'] else ''
            df = pd.concat([df, df_logs[df.columns]], ignore_index=True)
    except Exception as e:
        logging.warning(f"Pas de feedback production disponible: {e}")

    logging.info(f"Total donnees pour entrainement: {len(df)}")

    # Sauvegarde temporaire
    df.to_parquet('/tmp/training_data.parquet', index=False)
    context['ti'].xcom_push(key='data_size', value=len(df))

    # Versioning DVC du dataset combiné
    try:
        import subprocess
        PROJECT_PATH = "/opt/airflow/project"
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dataset_filename = f"training_data_{timestamp}.parquet"
        dataset_path = f"{PROJECT_PATH}/data/{dataset_filename}"

        # Crée le dossier data s'il n'existe pas
        os.makedirs(f"{PROJECT_PATH}/data", exist_ok=True)

        # Copie le dataset avec timestamp
        subprocess.run(["cp", "/tmp/training_data.parquet", dataset_path], check=True)

        # DVC add + push
        subprocess.run(["dvc", "add", f"data/{dataset_filename}"], cwd=PROJECT_PATH, check=True)
        subprocess.run(["dvc", "push"], cwd=PROJECT_PATH, check=True)

        logging.info(f"Dataset versionne avec DVC: {dataset_filename}")
        context['ti'].xcom_push(key='dataset_version', value=dataset_filename)
    except Exception as e:
        logging.warning(f"Erreur versioning DVC dataset: {e}")

def train_models(**context):
    """Entraine les modeles XGBoost avec embeddings"""
    logging.info("Chargement des donnees...")
    df = pd.read_parquet('/tmp/training_data.parquet')

    df['text_combined'] = df['subject'].fillna('') + " " + df['body_clean'].fillna('')

    num_features = ['body_length', 'answer_length', 'response_ratio',
                    'has_network', 'has_printer', 'has_security', 'has_hardware', 'has_software']

    # Filtre les classes avec moins de 2 exemples (requis pour stratify)
    queue_counts = df['refined_queue'].value_counts()
    valid_queues = queue_counts[queue_counts >= 2].index
    df = df[df['refined_queue'].isin(valid_queues)]
    logging.info(f"Apres filtrage classes rares: {len(df)} tickets")

    # Encodage targets
    le_queue = LabelEncoder()
    le_urgency = LabelEncoder()
    df['queue_encoded'] = le_queue.fit_transform(df['refined_queue'])
    df['urgency_encoded'] = le_urgency.fit_transform(df['urgency_level'])

    # Split
    X = df[['text_combined'] + num_features]
    y_queue = df['queue_encoded']
    y_urgency = df['urgency_encoded']

    X_train, X_test, yq_train, yq_test, yu_train, yu_test = train_test_split(
        X, y_queue, y_urgency, test_size=0.15, random_state=42,
        stratify=y_queue  # Stratifie uniquement sur queue
    )

    logging.info(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # Embeddings
    logging.info("Generation des embeddings...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    X_train_embed = embedder.encode(X_train['text_combined'].tolist(), show_progress_bar=True)
    X_test_embed = embedder.encode(X_test['text_combined'].tolist(), show_progress_bar=True)

    X_train_full = np.hstack([X_train_embed, X_train[num_features].values])
    X_test_full = np.hstack([X_test_embed, X_test[num_features].values])

    # Sample weights
    sample_weights = compute_sample_weight('balanced', yu_train)

    # MLflow - utilise le serveur externe
    mlflow.set_tracking_uri("http://host.docker.internal:5000")
    mlflow.set_experiment("tickets_classification_bootcamp")

    run_name = f"retraining_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Charger les hyperparametres optimises par Ray Tune (si disponibles)
    HYPERPARAMS_FILE = "/opt/airflow/data/best_hyperparams.json"
    hyperparams_loaded = False

    if os.path.exists(HYPERPARAMS_FILE):
        try:
            with open(HYPERPARAMS_FILE, 'r') as f:
                saved_params = json.load(f)
            hyperparams_loaded = True
            logging.info(f"Hyperparametres Ray Tune charges: {HYPERPARAMS_FILE}")
        except Exception as e:
            logging.warning(f"Erreur chargement hyperparametres: {e}")

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("triggered_by", "drift_detection")
        mlflow.log_param("data_size", len(df))
        mlflow.log_param("hyperparams_source", "ray_tune" if hyperparams_loaded else "default")

        # Log la version du dataset utilisé
        dataset_version = context['ti'].xcom_pull(key='dataset_version', task_ids='load_data')
        if dataset_version:
            mlflow.log_param("dataset_version", dataset_version)

        if hyperparams_loaded:
            # Utilise les hyperparametres optimises par Ray Tune
            logging.info("Utilisation des hyperparametres Ray Tune optimises")

            queue_params = saved_params.get("queue", {})
            urgency_params = saved_params.get("urgency", {})

            # Modele Queue avec params optimises
            logging.info("Entrainement modele Queue (params Ray Tune)...")
            model_queue = XGBClassifier(
                objective='multi:softprob',
                num_class=len(le_queue.classes_),
                eval_metric='mlogloss',
                random_state=42,
                n_jobs=-1,
                **{k: v for k, v in queue_params.items() if k in ['max_depth', 'learning_rate', 'n_estimators', 'min_child_weight', 'subsample', 'colsample_bytree']}
            )
            model_queue.fit(X_train_full, yq_train)
            mlflow.log_params({f"queue_{k}": v for k, v in queue_params.items()})

            # Modele Urgency avec params optimises
            logging.info("Entrainement modele Urgency (params Ray Tune)...")
            model_urgency = XGBClassifier(
                objective='multi:softprob',
                num_class=3,
                eval_metric='mlogloss',
                random_state=42,
                n_jobs=-1,
                **{k: v for k, v in urgency_params.items() if k in ['max_depth', 'learning_rate', 'n_estimators', 'min_child_weight', 'subsample', 'colsample_bytree']}
            )
            model_urgency.fit(X_train_full, yu_train, sample_weight=sample_weights)
            mlflow.log_params({f"urgency_{k}": v for k, v in urgency_params.items()})

        else:
            # Parametres par defaut (pas de Ray Tune disponible)
            logging.info("Pas d'hyperparametres Ray Tune - Utilisation parametres par defaut")

            default_params = {
                'objective': 'multi:softprob',
                'eval_metric': 'mlogloss',
                'random_state': 42,
                'n_jobs': -1,
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 200
            }

            # Train Queue
            logging.info("Entrainement modele Queue...")
            model_queue = XGBClassifier(**default_params, num_class=len(le_queue.classes_))
            model_queue.fit(X_train_full, yq_train)
            mlflow.log_params({f"queue_{k}": v for k, v in default_params.items()})

            # Train Urgency
            logging.info("Entrainement modele Urgency...")
            model_urgency = XGBClassifier(**default_params, num_class=3)
            model_urgency.fit(X_train_full, yu_train, sample_weight=sample_weights)
            mlflow.log_params({f"urgency_{k}": v for k, v in default_params.items()})

        # Evaluation des deux modeles
        pred_queue = model_queue.predict(X_test_full)
        acc_queue = accuracy_score(yq_test, pred_queue)
        f1_queue = f1_score(yq_test, pred_queue, average='weighted')
        mlflow.log_metric("accuracy_queue", acc_queue)
        mlflow.log_metric("f1_queue", f1_queue)

        pred_urgency = model_urgency.predict(X_test_full)
        acc_urgency = accuracy_score(yu_test, pred_urgency)
        f1_urgency = f1_score(yu_test, pred_urgency, average='weighted')
        mlflow.log_metric("accuracy_urgency", acc_urgency)
        mlflow.log_metric("f1_urgency", f1_urgency)

        logging.info(f"=== RESULTATS RETRAINING ===")
        logging.info(f"Queue     - Accuracy: {acc_queue:.4f} | F1: {f1_queue:.4f}")
        logging.info(f"Urgency   - Accuracy: {acc_urgency:.4f} | F1: {f1_urgency:.4f}")

        # Sauvegarde locale des modeles (pour deploiement)
        MODEL_PATH = "/opt/airflow/data/models"
        os.makedirs(MODEL_PATH, exist_ok=True)
        model_queue.save_model(f"{MODEL_PATH}/xgboost_queue_retrained.json")
        model_urgency.save_model(f"{MODEL_PATH}/xgboost_urgency_retrained.json")
        joblib.dump(le_queue, f"{MODEL_PATH}/le_queue.pkl")
        joblib.dump(le_urgency, f"{MODEL_PATH}/le_urgency.pkl")
        logging.info(f"Modeles sauvegardes dans {MODEL_PATH}")

        # Log models dans MLflow (optionnel, peut echouer si pas de shared storage)
        try:
            mlflow.xgboost.log_model(model_queue, "xgboost_queue_retrained")
            mlflow.xgboost.log_model(model_urgency, "xgboost_urgency_retrained")
            mlflow.log_artifact(f"{MODEL_PATH}/le_queue.pkl")
            mlflow.log_artifact(f"{MODEL_PATH}/le_urgency.pkl")
        except Exception as e:
            logging.warning(f"Impossible de logger artifacts MLflow (non bloquant): {e}")

        context['ti'].xcom_push(key='run_id', value=run.info.run_id)
        context['ti'].xcom_push(key='f1_queue', value=f1_queue)
        context['ti'].xcom_push(key='f1_urgency', value=f1_urgency)

def validate_and_promote(**context):
    """Valide les metriques et promeut le modele si amelioration"""
    run_id = context['ti'].xcom_pull(key='run_id', task_ids='train_models')
    f1_queue = context['ti'].xcom_pull(key='f1_queue', task_ids='train_models')
    f1_urgency = context['ti'].xcom_pull(key='f1_urgency', task_ids='train_models')

    # Seuils minimaux
    MIN_F1_QUEUE = 0.70
    MIN_F1_URGENCY = 0.65

    logging.info(f"Validation des metriques...")
    logging.info(f"F1 Queue: {f1_queue:.4f} (min: {MIN_F1_QUEUE})")
    logging.info(f"F1 Urgency: {f1_urgency:.4f} (min: {MIN_F1_URGENCY})")

    if f1_queue >= MIN_F1_QUEUE and f1_urgency >= MIN_F1_URGENCY:
        logging.info("Metriques validees - Promotion du modele en Production")

        mlflow.set_tracking_uri("http://host.docker.internal:5000")

        # Enregistre le modele dans le Model Registry
        model_uri = f"runs:/{run_id}/xgboost_queue_retrained"
        mv = mlflow.register_model(model_uri, "support_queue_classifier")
        logging.info(f"Modele Queue enregistre: version {mv.version}")

        model_uri_urg = f"runs:/{run_id}/xgboost_urgency_retrained"
        mv_urg = mlflow.register_model(model_uri_urg, "support_urgency_classifier")
        logging.info(f"Modele Urgency enregistre: version {mv_urg.version}")

        # Note: La promotion en Production necessite MLflow avec backend store
        logging.info("Modeles prets pour deploiement")
    else:
        logging.warning("Metriques insuffisantes - Modele non promu")
        logging.warning("Verification manuelle requise")

def deploy_model(**context):
    """Sauvegarde le modele avec DVC et declenche le CI/CD"""
    import subprocess
    import requests

    f1_queue = context['ti'].xcom_pull(key='f1_queue', task_ids='train_models')
    f1_urgency = context['ti'].xcom_pull(key='f1_urgency', task_ids='train_models')

    # Seuils minimaux
    MIN_F1_QUEUE = 0.70
    MIN_F1_URGENCY = 0.65

    if f1_queue < MIN_F1_QUEUE or f1_urgency < MIN_F1_URGENCY:
        logging.warning("Modele non deploye - metriques insuffisantes")
        return

    logging.info("=== DEPLOIEMENT AUTOMATIQUE ===")

    # Chemin des modeles sauvegardes par train_models
    MODEL_PATH = "/opt/airflow/data/models"

    # 1. DVC push vers S3 (optionnel - si DVC est configure)
    try:
        PROJECT_PATH = "/opt/airflow/project"  # Path vers le repo git

        if os.path.exists(PROJECT_PATH):
            # Copie les modeles vers le repo (depuis le bon chemin)
            subprocess.run(["cp", f"{MODEL_PATH}/le_queue.pkl", f"{PROJECT_PATH}/le_queue.pkl"], check=True)
            subprocess.run(["cp", f"{MODEL_PATH}/le_urgency.pkl", f"{PROJECT_PATH}/le_urgency.pkl"], check=True)

            # DVC add + push
            subprocess.run(["dvc", "add", "le_queue.pkl"], cwd=PROJECT_PATH, check=True)
            subprocess.run(["dvc", "add", "le_urgency.pkl"], cwd=PROJECT_PATH, check=True)
            subprocess.run(["dvc", "push"], cwd=PROJECT_PATH, check=True)

            logging.info("DVC push vers S3 OK")
        else:
            logging.warning(f"Repertoire projet non trouve: {PROJECT_PATH} - DVC skip")
    except Exception as e:
        logging.warning(f"Erreur DVC push (non bloquant): {e}")

    # 2. Declenche le CI/CD via GitHub API
    try:
        GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
        if not GITHUB_TOKEN:
            logging.warning("GITHUB_TOKEN non configure - CI/CD non declenche")
            logging.info(f"Modeles disponibles dans: {MODEL_PATH}")
            return

        # Trigger workflow_dispatch
        url = "https://api.github.com/repos/IsabelleB75/support-it-agent/actions/workflows/ci-cd.yaml/dispatches"
        headers = {
            "Authorization": f"token {GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json"
        }
        data = {"ref": "main"}

        response = requests.post(url, headers=headers, json=data)

        if response.status_code == 204:
            logging.info("CI/CD declenche avec succes!")
        else:
            logging.error(f"Erreur CI/CD: {response.status_code} - {response.text}")
    except Exception as e:
        logging.error(f"Erreur declenchement CI/CD: {e}")


def cleanup():
    """Nettoie les fichiers temporaires"""
    import os
    for f in ['/tmp/training_data.parquet', '/tmp/le_queue.pkl', '/tmp/le_urgency.pkl']:
        if os.path.exists(f):
            os.remove(f)
    logging.info("Fichiers temporaires nettoyes")

load_task = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    provide_context=True,
    dag=dag,
)

train_task = PythonOperator(
    task_id='train_models',
    python_callable=train_models,
    provide_context=True,
    dag=dag,
)

validate_task = PythonOperator(
    task_id='validate_and_promote',
    python_callable=validate_and_promote,
    provide_context=True,
    dag=dag,
)

deploy_task = PythonOperator(
    task_id='deploy_model',
    python_callable=deploy_model,
    provide_context=True,
    dag=dag,
)

cleanup_task = PythonOperator(
    task_id='cleanup',
    python_callable=cleanup,
    dag=dag,
)

load_task >> train_task >> validate_task >> deploy_task >> cleanup_task
