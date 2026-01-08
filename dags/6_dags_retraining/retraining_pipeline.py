from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
import pandas as pd
import numpy as np
from sqlalchemy import text
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
import sys
sys.path.insert(0, '/opt/airflow/dags')
from utils.db_config import get_db_engine, get_mlflow_tracking_uri, get_incremental_embeddings

default_args = {
    'owner': 'jedha_bootcamp',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


def get_run_tmp_dir(context):
    """Retourne un repertoire tmp unique par run (anti-concurrence)"""
    run_id = context['dag_run'].run_id if context.get('dag_run') else datetime.now().strftime('%Y%m%d_%H%M%S')
    # Sanitize run_id (peut contenir des caracteres speciaux)
    safe_run_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in run_id)
    return f"/opt/airflow/data/tmp/{safe_run_id}"

dag = DAG(
    'retraining_pipeline',
    default_args=default_args,
    description='Pipeline de retraining automatique declenche par drift',
    schedule_interval=None,  # Declenche par monitoring DAG
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,  # Evite concurrence sur repo Git/DVC partage
    tags=['retraining', 'mlflow', 'xgboost', 'mlops'],
)

def load_data(**context):
    """Charge les donnees enrichies - utilise le dataset de hyperparameter_tuning si fourni"""
    import subprocess
    import glob

    engine = get_db_engine()
    PROJECT_PATH = "/opt/airflow/project"

    # Verifie si un dataset_version a ete fourni par hyperparameter_tuning
    dataset_version = context['dag_run'].conf.get('dataset_version') if context.get('dag_run') else None

    if dataset_version and dataset_version != 'None':
        # Utilise le MEME dataset que hyperparameter_tuning (coherence donnees)
        logging.info("=" * 60)
        logging.info(f"UTILISATION DU DATASET DE HYPERPARAMETER_TUNING: {dataset_version}")
        logging.info("=" * 60)

        subprocess.run(["dvc", "pull"], cwd=PROJECT_PATH, check=False)
        dataset_path = f"{PROJECT_PATH}/data/{dataset_version}"

        if os.path.exists(dataset_path):
            df = pd.read_parquet(dataset_path)
            logging.info(f"Dataset charge: {len(df)} tickets (meme que tuning)")
            context['ti'].xcom_push(key='dataset_version', value=dataset_version)
        else:
            logging.warning(f"Dataset {dataset_version} non trouve, fallback...")
            dataset_version = None

    if not dataset_version or dataset_version == 'None':
        # Fallback: cree un nouveau dataset (cas ou retraining est lance seul)
        logging.info("=" * 60)
        logging.info("CREATION NOUVEAU DATASET (pas de tuning precedent)")
        logging.info("=" * 60)

        last_training_date = None

        # 1. Cherche le dataset le plus récent versionné par DVC
        try:
            subprocess.run(["dvc", "pull"], cwd=PROJECT_PATH, check=False)
            dataset_files = sorted(glob.glob(f"{PROJECT_PATH}/data/training_data_*.parquet"))
            if dataset_files:
                latest_dataset = dataset_files[-1]
                logging.info(f"Chargement du dataset DVC le plus recent: {latest_dataset}")
                df = pd.read_parquet(latest_dataset)
                logging.info(f"Dataset DVC charge: {len(df)} tickets")
                filename = os.path.basename(latest_dataset)
                date_str = filename.replace("training_data_", "").replace(".parquet", "")
                last_training_date = datetime.strptime(date_str, '%Y%m%d_%H%M%S')
            else:
                raise FileNotFoundError("Aucun dataset DVC trouve")
        except Exception as e:
            logging.info(f"Pas de dataset DVC disponible ({e}), chargement depuis PostgreSQL...")
            df = pd.read_sql("SELECT * FROM tickets_tech_en_enriched", engine)
            logging.info(f"Donnees historiques: {len(df)} tickets")

        # 2. Ajoute les feedbacks de production
        try:
            if last_training_date:
                date_filter = f"created_at >= '{last_training_date.strftime('%Y-%m-%d %H:%M:%S')}'"
            else:
                date_filter = "created_at >= NOW() - INTERVAL '30 days'"

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
                for col in df.columns:
                    if col not in df_logs.columns:
                        df_logs[col] = 0 if df[col].dtype in ['int64', 'float64'] else ''
                df = pd.concat([df, df_logs[df.columns]], ignore_index=True)
        except Exception as e:
            logging.warning(f"Pas de feedback production disponible: {e}")

        # 3. Versionne ce nouveau dataset
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            dataset_filename = f"training_data_{timestamp}.parquet"
            dataset_path = f"{PROJECT_PATH}/data/{dataset_filename}"
            os.makedirs(f"{PROJECT_PATH}/data", exist_ok=True)
            df.to_parquet(dataset_path, index=False)
            subprocess.run(["dvc", "add", f"data/{dataset_filename}"], cwd=PROJECT_PATH, check=True)
            subprocess.run(["dvc", "push"], cwd=PROJECT_PATH, check=True)
            logging.info(f"Nouveau dataset versionne: {dataset_filename}")
            context['ti'].xcom_push(key='dataset_version', value=dataset_filename)
        except Exception as e:
            logging.warning(f"Erreur versioning DVC: {e}")

    logging.info(f"Total donnees pour entrainement: {len(df)}")

    # Sauvegarde temporaire - repertoire unique par run (anti-concurrence)
    SHARED_DATA = get_run_tmp_dir(context)
    os.makedirs(SHARED_DATA, exist_ok=True)
    logging.info(f"Repertoire temporaire du run: {SHARED_DATA}")

    df.to_parquet(f'{SHARED_DATA}/training_data.parquet', index=False)
    context['ti'].xcom_push(key='data_size', value=len(df))
    context['ti'].xcom_push(key='tmp_dir', value=SHARED_DATA)

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
        subprocess.run(["cp", f"{SHARED_DATA}/training_data.parquet", dataset_path], check=True)

        # DVC add + push
        subprocess.run(["dvc", "add", f"data/{dataset_filename}"], cwd=PROJECT_PATH, check=True)
        subprocess.run(["dvc", "push"], cwd=PROJECT_PATH, check=True)

        logging.info(f"Dataset versionne avec DVC: {dataset_filename}")
        context['ti'].xcom_push(key='dataset_version', value=dataset_filename)
    except Exception as e:
        logging.warning(f"Erreur versioning DVC dataset: {e}")

def train_models(**context):
    """Entraine les modeles XGBoost avec embeddings"""
    # Recupere le repertoire specifique de ce run
    SHARED_DATA = context['ti'].xcom_pull(key='tmp_dir', task_ids='load_data')
    logging.info(f"Chargement des donnees depuis: {SHARED_DATA}")
    df = pd.read_parquet(f'{SHARED_DATA}/training_data.parquet')

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

    # EMBEDDINGS INCREMENTAUX - réutilise les embeddings existants depuis PostgreSQL
    logging.info("=" * 50)
    logging.info("EMBEDDINGS INCREMENTAUX (optimisation)")
    logging.info("=" * 50)
    engine = get_db_engine()
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    # Calcule/récupère TOUS les embeddings d'un coup (avant split)
    all_texts = df['text_combined'].tolist()
    all_embeddings = get_incremental_embeddings(engine, all_texts, embedder)

    logging.info(f"Embeddings shape: {all_embeddings.shape}")

    # Split avec les embeddings pré-calculés
    X_num = df[num_features].values
    y_queue = df['queue_encoded'].values
    y_urgency = df['urgency_encoded'].values

    # Split indices (pour garder correspondance embeddings/features)
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.15, random_state=42,
        stratify=y_queue
    )

    # Applique le split
    X_train_embed = all_embeddings[train_idx]
    X_test_embed = all_embeddings[test_idx]
    X_train_num = X_num[train_idx]
    X_test_num = X_num[test_idx]
    yq_train, yq_test = y_queue[train_idx], y_queue[test_idx]
    yu_train, yu_test = y_urgency[train_idx], y_urgency[test_idx]

    logging.info(f"Train: {len(train_idx)} | Test: {len(test_idx)}")

    # Combine embeddings + features numériques
    X_train_full = np.hstack([X_train_embed, X_train_num])
    X_test_full = np.hstack([X_test_embed, X_test_num])

    # Sample weights
    sample_weights = compute_sample_weight('balanced', yu_train)

    # MLflow - utilise le serveur externe
    mlflow.set_tracking_uri(get_mlflow_tracking_uri())
    mlflow.set_experiment("tickets_classification_bootcamp")

    run_name = f"retraining_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Charger les hyperparametres optimises par Ray Tune depuis MLflow (source de verite)
    HYPERPARAMS_FILE = "/opt/airflow/data/best_hyperparams.json"
    hyperparams_loaded = False
    saved_params = {}

    # 1. Essaie de recuperer depuis MLflow via run_id specifique (passe par tuning DAG)
    try:
        mlflow.set_tracking_uri(get_mlflow_tracking_uri())
        client = mlflow.tracking.MlflowClient()

        # Recupere le run_id passe par le DAG de tuning (via dag_run.conf)
        tuning_run_id = context['dag_run'].conf.get('tuning_run_id') if context.get('dag_run') else None

        if tuning_run_id and tuning_run_id != 'None':
            # Utilise le run_id specifique (plus robuste que "latest")
            logging.info(f"Recuperation hyperparams depuis MLflow run specifique: {tuning_run_id}")
            artifact_path = client.download_artifacts(
                tuning_run_id,
                "hyperparams/best_hyperparams.json",
                "/tmp"
            )
            with open(artifact_path, 'r') as f:
                saved_params = json.load(f)
            hyperparams_loaded = True
            logging.info("Hyperparametres charges depuis MLflow artifact (run_id specifique)")
        else:
            # Fallback: cherche le dernier run de hyperparameter_tuning
            logging.info("Pas de run_id dans conf, recherche du dernier run tuning...")
            experiment = client.get_experiment_by_name("hyperparameter_tuning")
            if experiment:
                runs = client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    order_by=["start_time DESC"],
                    max_results=1
                )
                if runs:
                    latest_run = runs[0]
                    logging.info(f"Recuperation hyperparams depuis MLflow run: {latest_run.info.run_id}")
                    artifact_path = client.download_artifacts(
                        latest_run.info.run_id,
                        "hyperparams/best_hyperparams.json",
                        "/tmp"
                    )
                    with open(artifact_path, 'r') as f:
                        saved_params = json.load(f)
                    hyperparams_loaded = True
                    logging.info("Hyperparametres charges depuis MLflow artifact (latest run)")
    except Exception as e:
        logging.warning(f"MLflow artifact non disponible: {e}")

    # 2. Fallback: fichier local
    if not hyperparams_loaded and os.path.exists(HYPERPARAMS_FILE):
        try:
            with open(HYPERPARAMS_FILE, 'r') as f:
                saved_params = json.load(f)
            hyperparams_loaded = True
            logging.info(f"Hyperparametres charges depuis fichier local: {HYPERPARAMS_FILE}")
        except Exception as e:
            logging.warning(f"Erreur chargement fichier local: {e}")

    # Recupere le tuning_run_id pour tracabilite
    tuning_run_id = context['dag_run'].conf.get('tuning_run_id') if context.get('dag_run') else None

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("triggered_by", "drift_detection")
        mlflow.log_param("data_size", len(df))
        mlflow.log_param("hyperparams_source", "ray_tune" if hyperparams_loaded else "default")

        # Log tuning_run_id pour tracabilite complete
        if tuning_run_id and tuning_run_id != 'None':
            mlflow.log_param("tuning_run_id", tuning_run_id)

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
                num_class=len(le_urgency.classes_),  # Dynamique, pas hardcodé
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
            model_urgency = XGBClassifier(**default_params, num_class=len(le_urgency.classes_))
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
    """Valide les metriques et promeut le modele en Production si amelioration"""
    run_id = context['ti'].xcom_pull(key='run_id', task_ids='train_models')
    f1_queue = context['ti'].xcom_pull(key='f1_queue', task_ids='train_models')
    f1_urgency = context['ti'].xcom_pull(key='f1_urgency', task_ids='train_models')

    # Seuils minimaux
    MIN_F1_QUEUE = 0.70
    MIN_F1_URGENCY = 0.65

    logging.info(f"Validation des metriques...")
    logging.info(f"F1 Queue: {f1_queue:.4f} (min: {MIN_F1_QUEUE})")
    logging.info(f"F1 Urgency: {f1_urgency:.4f} (min: {MIN_F1_URGENCY})")

    # Client MLflow pour logging et promotion (cree une seule fois)
    mlflow.set_tracking_uri(get_mlflow_tracking_uri())
    client = mlflow.tracking.MlflowClient()

    if f1_queue >= MIN_F1_QUEUE and f1_urgency >= MIN_F1_URGENCY:
        logging.info("Metriques validees - Promotion du modele en Production")

        try:
            # Enregistre et promeut le modele Queue
            model_uri = f"runs:/{run_id}/xgboost_queue_retrained"
            mv = mlflow.register_model(model_uri, "support_queue_classifier")
            logging.info(f"Modele Queue enregistre: version {mv.version}")

            # VRAIE PROMOTION vers Production
            client.transition_model_version_stage(
                name="support_queue_classifier",
                version=mv.version,
                stage="Production",
                archive_existing_versions=True  # Archive les anciennes versions Production
            )
            logging.info(f"Modele Queue v{mv.version} promu en PRODUCTION")

            # Enregistre et promeut le modele Urgency
            model_uri_urg = f"runs:/{run_id}/xgboost_urgency_retrained"
            mv_urg = mlflow.register_model(model_uri_urg, "support_urgency_classifier")
            logging.info(f"Modele Urgency enregistre: version {mv_urg.version}")

            # VRAIE PROMOTION vers Production
            client.transition_model_version_stage(
                name="support_urgency_classifier",
                version=mv_urg.version,
                stage="Production",
                archive_existing_versions=True
            )
            logging.info(f"Modele Urgency v{mv_urg.version} promu en PRODUCTION")

            # Push validation status pour deploy_model
            context['ti'].xcom_push(key='promotion_status', value='success')
            context['ti'].xcom_push(key='queue_version', value=mv.version)
            context['ti'].xcom_push(key='urgency_version', value=mv_urg.version)

            # Log dans le run MLflow pour tracabilite complete (via Client API, plus robuste)
            # Utilise set_tag pour les metadonnees (plus semantique que log_param)
            try:
                client.set_tag(run_id, "promotion_status", "success")
                client.set_tag(run_id, "queue_model_version", mv.version)
                client.set_tag(run_id, "urgency_model_version", mv_urg.version)
                logging.info("Promotion status et versions logges dans MLflow (tags)")
            except Exception as log_e:
                logging.warning(f"Impossible de logger dans MLflow: {log_e}")

        except Exception as e:
            logging.error(f"Erreur promotion MLflow: {e}")
            context['ti'].xcom_push(key='promotion_status', value='failed')
            # Log l'echec dans MLflow (via Client API)
            try:
                client.set_tag(run_id, "promotion_status", "failed")
                client.set_tag(run_id, "promotion_error", str(e)[:250])
            except:
                pass
    else:
        logging.warning("Metriques insuffisantes - Modele non promu")
        logging.warning("Verification manuelle requise")
        context['ti'].xcom_push(key='promotion_status', value='rejected')
        # Log le rejet dans MLflow (via Client API)
        try:
            client.set_tag(run_id, "promotion_status", "rejected")
            client.set_tag(run_id, "rejection_reason", f"F1 Queue={f1_queue:.4f} < {MIN_F1_QUEUE} or F1 Urgency={f1_urgency:.4f} < {MIN_F1_URGENCY}")
        except:
            pass

def check_promotion_success(**context):
    """ShortCircuit: retourne True si promotion OK, False sinon (skip deploy)"""
    promotion_status = context['ti'].xcom_pull(key='promotion_status', task_ids='validate_and_promote')
    if promotion_status == 'success':
        logging.info("Promotion validee - deploiement autorise")
        return True
    else:
        logging.warning(f"Promotion non validee ({promotion_status}) - skip deploiement")
        return False

def deploy_model(**context):
    """Sauvegarde TOUS les artefacts modele avec DVC et declenche le CI/CD"""
    import subprocess
    import requests

    # Plus besoin de verifier ici - ShortCircuitOperator l'a deja fait
    f1_queue = context['ti'].xcom_pull(key='f1_queue', task_ids='train_models')
    f1_urgency = context['ti'].xcom_pull(key='f1_urgency', task_ids='train_models')

    logging.info("=== DEPLOIEMENT AUTOMATIQUE ===")
    logging.info(f"F1 Queue: {f1_queue:.4f} | F1 Urgency: {f1_urgency:.4f}")

    # Chemin des modeles sauvegardes par train_models
    MODEL_PATH = "/opt/airflow/data/models"
    PROJECT_PATH = "/opt/airflow/project"

    # 1. DVC push vers S3 - TOUS LES FICHIERS MODELE
    try:
        if os.path.exists(PROJECT_PATH):
            # Cree les dossiers si necessaire
            os.makedirs(f"{PROJECT_PATH}/xgboost_queue_v2", exist_ok=True)
            os.makedirs(f"{PROJECT_PATH}/xgboost_urgency_v2", exist_ok=True)

            # Copie les 4 fichiers modele vers le repo
            # 1. Label Encoders (.pkl)
            subprocess.run(["cp", f"{MODEL_PATH}/le_queue.pkl", f"{PROJECT_PATH}/le_queue.pkl"], check=True)
            subprocess.run(["cp", f"{MODEL_PATH}/le_urgency.pkl", f"{PROJECT_PATH}/le_urgency.pkl"], check=True)

            # 2. Modeles XGBoost (.json -> renommes vers le format attendu par app.py)
            subprocess.run(["cp", f"{MODEL_PATH}/xgboost_queue_retrained.json",
                          f"{PROJECT_PATH}/xgboost_queue_v2/model.xgb"], check=True)
            subprocess.run(["cp", f"{MODEL_PATH}/xgboost_urgency_retrained.json",
                          f"{PROJECT_PATH}/xgboost_urgency_v2/model.xgb"], check=True)

            logging.info("Fichiers modele copies vers le repo")

            # DVC add pour TOUS les artefacts
            subprocess.run(["dvc", "add", "le_queue.pkl"], cwd=PROJECT_PATH, check=True)
            subprocess.run(["dvc", "add", "le_urgency.pkl"], cwd=PROJECT_PATH, check=True)
            subprocess.run(["dvc", "add", "xgboost_queue_v2"], cwd=PROJECT_PATH, check=True)
            subprocess.run(["dvc", "add", "xgboost_urgency_v2"], cwd=PROJECT_PATH, check=True)

            # Push vers S3
            subprocess.run(["dvc", "push"], cwd=PROJECT_PATH, check=True)

            logging.info("DVC push vers S3 OK - 4 artefacts (2 encoders + 2 modeles XGBoost)")

            # IMPORTANT: Git commit/push des pointeurs .dvc pour que CI/CD trouve les bons hash
            logging.info("Git commit/push des fichiers DVC...")
            # git add -A pour capturer .dvc, dvc.lock, .gitignore (plus robuste que *.dvc)
            subprocess.run(["git", "add", "-A"], cwd=PROJECT_PATH, check=True)

            # Verifie s'il y a des changements a commiter (evite commit/push/CI inutiles)
            status_result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=PROJECT_PATH, capture_output=True, text=True
            )
            has_changes = bool(status_result.stdout.strip())

            if has_changes:
                # Commit avec message descriptif
                queue_version = context['ti'].xcom_pull(key='queue_version', task_ids='validate_and_promote') or "N/A"
                urgency_version = context['ti'].xcom_pull(key='urgency_version', task_ids='validate_and_promote') or "N/A"
                commit_msg = f"[Airflow] Update DVC pointers - Queue v{queue_version}, Urgency v{urgency_version}"
                subprocess.run(["git", "commit", "-m", commit_msg], cwd=PROJECT_PATH, check=True)

                # Push vers GitHub (prerequis: git remote configure avec credentials/SSH)
                subprocess.run(["git", "push", "origin", "main"], cwd=PROJECT_PATH, check=True)
                logging.info("Git push des pointeurs DVC OK")
            else:
                logging.info("Aucun changement DVC detecte - skip git commit/push")

            dvc_success = True
            git_has_changes = has_changes
        else:
            logging.warning(f"Repertoire projet non trouve: {PROJECT_PATH} - DVC skip")
            dvc_success = False
            git_has_changes = False
    except Exception as e:
        logging.error(f"Erreur DVC/Git: {e}")
        dvc_success = False
        git_has_changes = False

    # 2. Declenche le CI/CD via GitHub API
    # Note: on declenche meme si DVC echoue car les modeles sont aussi dans MLflow
    if not dvc_success:
        logging.warning("DVC/Git a echoue - les modeles sont disponibles via MLflow")
        logging.info("Declenchement du CI/CD quand meme (modeles MLflow)...")
    elif not git_has_changes:
        logging.info("Aucun changement DVC detecte - declenchement CI/CD pour deploiement MLflow...")

    # Continue vers le declenchement CI/CD

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


def cleanup(**context):
    """Nettoie les fichiers temporaires du run specifique (pas les autres runs)"""
    import shutil
    # Recupere le repertoire specifique de CE run
    SHARED_DATA = context['ti'].xcom_pull(key='tmp_dir', task_ids='load_data')
    if not SHARED_DATA:
        SHARED_DATA = get_run_tmp_dir(context)

    try:
        if os.path.exists(SHARED_DATA):
            shutil.rmtree(SHARED_DATA)
            logging.info(f"Repertoire temporaire nettoye: {SHARED_DATA}")
        else:
            logging.info(f"Repertoire deja absent: {SHARED_DATA}")
    except Exception as e:
        logging.warning(f"Erreur cleanup: {e}")

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

# ShortCircuit: skip deploy si promotion failed
check_deploy_task = ShortCircuitOperator(
    task_id='check_promotion_success',
    python_callable=check_promotion_success,
    provide_context=True,
    dag=dag,
)

deploy_task = PythonOperator(
    task_id='deploy_model',
    python_callable=deploy_model,
    provide_context=True,
    pool='deploy_pool',  # Pool taille 1 pour eviter concurrence inter-DAG sur Git/DVC
    dag=dag,
)

cleanup_task = PythonOperator(
    task_id='cleanup',
    python_callable=cleanup,
    provide_context=True,
    dag=dag,
)

# Pipeline: validate → check → deploy (skip si check=False) → cleanup
load_task >> train_task >> validate_task >> check_deploy_task >> deploy_task >> cleanup_task
