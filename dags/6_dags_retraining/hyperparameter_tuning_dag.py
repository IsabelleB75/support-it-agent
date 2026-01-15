"""
DAG Airflow pour optimisation des hyperparametres avec Ray Tune
Declenche par monitoring si drift > 30%, puis lance le retraining
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer
import json
import logging
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

dag = DAG(
    'hyperparameter_tuning_ray_tune',
    default_args=default_args,
    description='Optimisation hyperparametres XGBoost avec Ray Tune puis retraining',
    schedule_interval=None,  # Declenche par monitoring
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,  # Evite conflit port Ray dashboard (8265) + fichiers partages
    tags=['hyperparameters', 'ray-tune', 'optimization', 'mlops'],
)

# Configuration
SAMPLE_SIZE = 3000  # Subset pour rapidite dans Airflow
N_TRIALS = 10
OUTPUT_FILE = "/opt/airflow/data/best_hyperparams.json"


def get_run_tmp_dir(context):
    """Retourne un repertoire tmp unique par run (anti-concurrence)"""
    run_id = context['dag_run'].run_id if context.get('dag_run') else datetime.now().strftime('%Y%m%d_%H%M%S')
    # Sanitize run_id (peut contenir des caracteres speciaux)
    safe_run_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in run_id)
    return f"/opt/airflow/data/tmp/{safe_run_id}"


def load_and_prepare_data(**context):
    """Charge les donnees ENRICHIES (DVC + feedbacks) et prepare les features"""
    import subprocess
    import glob
    from sqlalchemy import text

    logging.info("=" * 60)
    logging.info("CHARGEMENT DES DONNEES ENRICHIES POUR RAY TUNE")
    logging.info("=" * 60)

    engine = get_db_engine()
    PROJECT_PATH = "/opt/airflow/project"
    last_training_date = None

    # 1. Charge le dataset DVC existant (comme retraining)
    try:
        subprocess.run(["dvc", "pull"], cwd=PROJECT_PATH, check=False)
        dataset_files = sorted(glob.glob(f"{PROJECT_PATH}/data/training_data_*.parquet"))
        if dataset_files:
            latest_dataset = dataset_files[-1]
            logging.info(f"Dataset DVC charge: {latest_dataset}")
            df = pd.read_parquet(latest_dataset)
            # Extrait la date pour filtrer les nouveaux feedbacks
            filename = os.path.basename(latest_dataset)
            date_str = filename.replace("training_data_", "").replace(".parquet", "")
            last_training_date = datetime.strptime(date_str, '%Y%m%d_%H%M%S')
        else:
            raise FileNotFoundError("Aucun dataset DVC")
    except Exception as e:
        logging.info(f"Pas de dataset DVC ({e}), chargement depuis PostgreSQL...")
        df = pd.read_sql("SELECT * FROM tickets_tech_en_enriched", engine)

    logging.info(f"Dataset initial: {len(df)} tickets")

    # 2. Ajoute les feedbacks de production (prediction_logs)
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
        logging.warning(f"Pas de feedback disponible: {e}")

    logging.info(f"Dataset ENRICHI total: {len(df)} tickets")

    # 3. Sauvegarde le nouveau dataset dans DVC (pour que retraining utilise le meme)
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dataset_filename = f"training_data_{timestamp}.parquet"
        dataset_path = f"{PROJECT_PATH}/data/{dataset_filename}"
        os.makedirs(f"{PROJECT_PATH}/data", exist_ok=True)

        df.to_parquet(dataset_path, index=False)
        subprocess.run(["dvc", "add", f"data/{dataset_filename}"], cwd=PROJECT_PATH, check=True)
        subprocess.run(["dvc", "push"], cwd=PROJECT_PATH, check=True)

        logging.info(f"Dataset enrichi versionne DVC: {dataset_filename}")
        context['ti'].xcom_push(key='dataset_version', value=dataset_filename)
    except Exception as e:
        logging.warning(f"Erreur versioning DVC: {e}")
        context['ti'].xcom_push(key='dataset_version', value=None)

    # Echantillonnage stratifie pour rapidite
    if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
        df = df.groupby(['refined_queue', 'urgency_level'], group_keys=False).apply(
            lambda x: x.sample(n=max(1, int(len(x) * SAMPLE_SIZE / len(df))), random_state=42)
        ).reset_index(drop=True)
        logging.info(f"Echantillon pour tuning: {len(df)} tickets")

    # Preparation features
    df['text_combined'] = df['subject'].fillna('') + " " + df['body_clean'].fillna('')

    num_features = ['body_length', 'answer_length', 'response_ratio',
                    'has_network', 'has_printer', 'has_security', 'has_hardware', 'has_software']

    # Encodage targets
    le_queue = LabelEncoder()
    le_urgency = LabelEncoder()
    df['queue_encoded'] = le_queue.fit_transform(df['refined_queue'])
    df['urgency_encoded'] = le_urgency.fit_transform(df['urgency_level'])

    # EMBEDDINGS INCREMENTAUX - réutilise les embeddings existants depuis PostgreSQL
    logging.info("=" * 50)
    logging.info("EMBEDDINGS INCREMENTAUX (optimisation)")
    logging.info("=" * 50)
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = get_incremental_embeddings(engine, df['text_combined'].tolist(), embedder)

    # Features completes
    X = np.hstack([embeddings, df[num_features].values])

    # Split
    X_train, X_test, yq_train, yq_test, yu_train, yu_test = train_test_split(
        X, df['queue_encoded'].values, df['urgency_encoded'].values,
        test_size=0.2, random_state=42
    )

    # Sauvegarde temporaire - repertoire unique par run (anti-concurrence)
    SHARED_DATA = get_run_tmp_dir(context)
    os.makedirs(SHARED_DATA, exist_ok=True)
    logging.info(f"Repertoire temporaire du run: {SHARED_DATA}")

    np.save(f'{SHARED_DATA}/X_train.npy', X_train)
    np.save(f'{SHARED_DATA}/X_test.npy', X_test)
    np.save(f'{SHARED_DATA}/yq_train.npy', yq_train)
    np.save(f'{SHARED_DATA}/yq_test.npy', yq_test)
    np.save(f'{SHARED_DATA}/yu_train.npy', yu_train)
    np.save(f'{SHARED_DATA}/yu_test.npy', yu_test)

    context['ti'].xcom_push(key='n_classes_queue', value=int(len(le_queue.classes_)))
    context['ti'].xcom_push(key='n_classes_urgency', value=int(len(le_urgency.classes_)))
    context['ti'].xcom_push(key='data_size', value=len(df))
    context['ti'].xcom_push(key='tmp_dir', value=SHARED_DATA)

    logging.info("Donnees preparees et sauvegardees")


def run_ray_tune_optimization(**context):
    """Execute Ray Tune pour trouver les meilleurs hyperparametres"""
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search.optuna import OptunaSearch

    logging.info("=" * 60)
    logging.info("LANCEMENT RAY TUNE OPTIMIZATION")
    logging.info("=" * 60)

    # Chargement des donnees preparees (repertoire unique par run)
    SHARED_DATA = context['ti'].xcom_pull(key='tmp_dir', task_ids='load_data')
    logging.info(f"Chargement depuis: {SHARED_DATA}")
    X_train = np.load(f'{SHARED_DATA}/X_train.npy')
    X_test = np.load(f'{SHARED_DATA}/X_test.npy')
    yq_train = np.load(f'{SHARED_DATA}/yq_train.npy')
    yq_test = np.load(f'{SHARED_DATA}/yq_test.npy')
    yu_train = np.load(f'{SHARED_DATA}/yu_train.npy')
    yu_test = np.load(f'{SHARED_DATA}/yu_test.npy')

    n_classes_queue = context['ti'].xcom_pull(key='n_classes_queue', task_ids='load_data')
    n_classes_urgency = context['ti'].xcom_pull(key='n_classes_urgency', task_ids='load_data')

    logging.info(f"Donnees chargees: X_train={X_train.shape}, classes_queue={n_classes_queue}, classes_urgency={n_classes_urgency}")

    # Initialisation Ray (dashboard desactive pour eviter conflit de port si crash)
    ray.init(
        ignore_reinit_error=True,
        num_cpus=4,
        include_dashboard=False  # Evite port 8265 occupe apres crash
    )

    try:
        # Espace de recherche
        search_space = {
            "max_depth": tune.choice([3, 4, 5, 6, 7, 8]),
            "learning_rate": tune.loguniform(0.01, 0.3),
            "n_estimators": tune.choice([50, 100, 150, 200]),
            "min_child_weight": tune.choice([1, 3, 5, 7]),
            "subsample": tune.uniform(0.6, 1.0),
            "colsample_bytree": tune.uniform(0.6, 1.0),
        }

        # --- Recherche Queue ---
        logging.info("Ray Tune - Optimisation modele QUEUE...")

        scheduler_queue = ASHAScheduler(max_t=10, grace_period=1, reduction_factor=2)

        def train_queue(config):
            model = XGBClassifier(
                **config,
                objective='multi:softprob',
                num_class=n_classes_queue,
                eval_metric='mlogloss',
                random_state=42,
                n_jobs=2
            )
            try:
                # error_score=0 pour eviter NaN quand un fold echoue
                scores = cross_val_score(model, X_train, yq_train, cv=3, scoring='f1_weighted', n_jobs=1, error_score=0)
                score = scores.mean()
                # Si score est NaN ou 0, retourne une petite valeur pour que Optuna continue
                if np.isnan(score) or score == 0:
                    score = 0.01
                return {"f1_score": score}
            except Exception as e:
                return {"f1_score": 0.01}

        analysis_queue = tune.run(
            train_queue,
            config=search_space,
            num_samples=N_TRIALS,
            scheduler=scheduler_queue,
            search_alg=OptunaSearch(),
            metric="f1_score",
            mode="max",
            verbose=1,
            resources_per_trial={"cpu": 1},
            max_concurrent_trials=2,
        )

        # Gestion du cas ou aucun trial n'a reussi
        if analysis_queue.best_config is None:
            logging.warning("Queue - Aucun trial valide, utilisation des parametres par defaut")
            best_queue = {"max_depth": 5, "learning_rate": 0.1, "n_estimators": 100,
                          "min_child_weight": 3, "subsample": 0.8, "colsample_bytree": 0.8}
            score_queue = 0.0
        else:
            best_queue = analysis_queue.best_config
            score_queue = analysis_queue.best_result.get("f1_score", 0.0) if analysis_queue.best_result else 0.0
        logging.info(f"Queue - Best F1: {score_queue:.4f}")
        logging.info(f"Queue - Best params: {best_queue}")

        # --- Recherche Urgency (CV coherent avec Queue) ---
        logging.info("Ray Tune - Optimisation modele URGENCY (avec CV)...")

        scheduler_urgency = ASHAScheduler(max_t=10, grace_period=1, reduction_factor=2)

        def train_urgency(config):
            model = XGBClassifier(
                **config,
                objective='multi:softprob',
                num_class=n_classes_urgency,
                eval_metric='mlogloss',
                random_state=42,
                n_jobs=2
            )
            try:
                # error_score=0 pour eviter NaN quand un fold echoue
                scores = cross_val_score(model, X_train, yu_train, cv=3, scoring='f1_weighted', n_jobs=1, error_score=0)
                score = scores.mean()
                if np.isnan(score) or score == 0:
                    score = 0.01
                return {"f1_score": score}
            except Exception as e:
                return {"f1_score": 0.01}

        analysis_urgency = tune.run(
            train_urgency,
            config=search_space,
            num_samples=N_TRIALS,
            scheduler=scheduler_urgency,
            search_alg=OptunaSearch(),
            metric="f1_score",
            mode="max",
            verbose=1,
            resources_per_trial={"cpu": 1},
            max_concurrent_trials=2,
        )

        # Gestion du cas ou aucun trial n'a reussi
        if analysis_urgency.best_config is None:
            logging.warning("Urgency - Aucun trial valide, utilisation des parametres par defaut")
            best_urgency = {"max_depth": 5, "learning_rate": 0.1, "n_estimators": 100,
                            "min_child_weight": 3, "subsample": 0.8, "colsample_bytree": 0.8}
            score_urgency = 0.0
        else:
            best_urgency = analysis_urgency.best_config
            score_urgency = analysis_urgency.best_result.get("f1_score", 0.0) if analysis_urgency.best_result else 0.0
        logging.info(f"Urgency - Best F1: {score_urgency:.4f}")
        logging.info(f"Urgency - Best params: {best_urgency}")

        # Push results
        context['ti'].xcom_push(key='best_queue', value=best_queue)
        context['ti'].xcom_push(key='best_urgency', value=best_urgency)
        context['ti'].xcom_push(key='score_queue', value=score_queue)
        context['ti'].xcom_push(key='score_urgency', value=score_urgency)

    finally:
        # Toujours fermer Ray proprement
        ray.shutdown()
        logging.info("Ray shutdown complete")


def save_hyperparameters(**context):
    """Sauvegarde les meilleurs hyperparametres dans MLflow (source de verite) + JSON local"""
    import mlflow

    logging.info("=" * 60)
    logging.info("SAUVEGARDE DES HYPERPARAMETRES")
    logging.info("=" * 60)

    best_queue = context['ti'].xcom_pull(key='best_queue', task_ids='ray_tune_optimization')
    best_urgency = context['ti'].xcom_pull(key='best_urgency', task_ids='ray_tune_optimization')
    score_queue = context['ti'].xcom_pull(key='score_queue', task_ids='ray_tune_optimization')
    score_urgency = context['ti'].xcom_pull(key='score_urgency', task_ids='ray_tune_optimization')
    data_size = context['ti'].xcom_pull(key='data_size', task_ids='load_data')

    # Prepare hyperparams dict
    hyperparams = {
        "queue": best_queue,
        "urgency": best_urgency,
        "scores": {
            "queue": score_queue,
            "urgency": score_urgency
        },
        "sample_size": SAMPLE_SIZE,
        "n_trials": N_TRIALS,
        "method": "Ray Tune + Optuna + ASHA (CV coherent)",
        "timestamp": datetime.now().isoformat(),
        "triggered_by": "drift_detection"
    }

    # Sauvegarde JSON local (backup)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(hyperparams, f, indent=2)
    logging.info(f"Hyperparametres sauvegardes localement: {OUTPUT_FILE}")

    # Log dans MLflow (SOURCE DE VERITE pour le retraining)
    mlflow_run_id = None
    try:
        mlflow.set_tracking_uri(get_mlflow_tracking_uri())
        mlflow.set_experiment("hyperparameter_tuning")

        with mlflow.start_run(run_name=f"ray_tune_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
            mlflow_run_id = run.info.run_id

            # Log params
            mlflow.log_param("method", "Ray Tune + Optuna + ASHA")
            mlflow.log_param("sample_size", SAMPLE_SIZE)
            mlflow.log_param("n_trials", N_TRIALS)
            mlflow.log_param("data_size", data_size)
            mlflow.log_param("triggered_by", "drift_detection")
            mlflow.log_param("evaluation_method", "cross_validation_3fold")

            for k, v in best_queue.items():
                mlflow.log_param(f"queue_{k}", v)
            for k, v in best_urgency.items():
                mlflow.log_param(f"urgency_{k}", v)

            # Log metrics
            mlflow.log_metric("best_f1_queue", score_queue)
            mlflow.log_metric("best_f1_urgency", score_urgency)

            # Log hyperparams JSON comme ARTEFACT (source de verite pour retraining)
            mlflow.log_artifact(OUTPUT_FILE, artifact_path="hyperparams")

            logging.info(f"MLflow run_id: {mlflow_run_id}")
            logging.info("Hyperparametres enregistres comme artefact MLflow")

    except Exception as e:
        logging.warning(f"Erreur MLflow (non bloquant): {e}")

    # Push run_id pour que retraining puisse recuperer l'artefact
    context['ti'].xcom_push(key='mlflow_tuning_run_id', value=mlflow_run_id)

    # Resume
    logging.info("=" * 60)
    logging.info("RESULTATS OPTIMISATION HYPERPARAMETRES")
    logging.info("=" * 60)
    logging.info(f"Queue   - F1: {score_queue:.4f}")
    logging.info(f"Urgency - F1: {score_urgency:.4f}")
    logging.info("=" * 60)
    logging.info("Declenchement du RETRAINING avec nouveaux hyperparametres...")


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


# Tasks
load_task = PythonOperator(
    task_id='load_data',
    python_callable=load_and_prepare_data,
    provide_context=True,
    dag=dag,
)

ray_tune_task = PythonOperator(
    task_id='ray_tune_optimization',
    python_callable=run_ray_tune_optimization,
    provide_context=True,
    dag=dag,
)

save_task = PythonOperator(
    task_id='save_hyperparameters',
    python_callable=save_hyperparameters,
    provide_context=True,
    dag=dag,
)

cleanup_task = PythonOperator(
    task_id='cleanup',
    python_callable=cleanup,
    provide_context=True,
    dag=dag,
)

trigger_retraining = TriggerDagRunOperator(
    task_id='trigger_retraining',
    trigger_dag_id='retraining_pipeline',
    # Passe le run_id MLflow + dataset_version pour que retraining utilise exactement les memes donnees
    conf={
        "tuning_run_id": "{{ ti.xcom_pull(task_ids='save_hyperparameters', key='mlflow_tuning_run_id') }}",
        "dataset_version": "{{ ti.xcom_pull(task_ids='load_data', key='dataset_version') }}"
    },
    dag=dag,
)

# Pipeline complet
load_task >> ray_tune_task >> save_task >> cleanup_task >> trigger_retraining
