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
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer
import json
import logging
import os

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
    start_date=datetime(2025, 12, 1),
    catchup=False,
    tags=['hyperparameters', 'ray-tune', 'optimization', 'mlops'],
)

# Configuration
SAMPLE_SIZE = 3000  # Subset pour rapidite dans Airflow
N_TRIALS = 10
OUTPUT_FILE = "/opt/airflow/data/best_hyperparams.json"


def load_and_prepare_data(**context):
    """Charge les donnees et prepare les features avec embeddings"""
    logging.info("=" * 60)
    logging.info("CHARGEMENT DES DONNEES POUR RAY TUNE")
    logging.info("=" * 60)

    engine = create_engine('postgresql://bootcamp_user:bootcamp_password@host.docker.internal:5433/support_tech')
    df = pd.read_sql("SELECT * FROM tickets_tech_en_enriched", engine)

    logging.info(f"Dataset complet: {len(df)} tickets")

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

    # Embeddings
    logging.info("Generation des embeddings (SentenceTransformer)...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedder.encode(df['text_combined'].tolist(), show_progress_bar=True)

    # Features completes
    X = np.hstack([embeddings, df[num_features].values])

    # Split
    X_train, X_test, yq_train, yq_test, yu_train, yu_test = train_test_split(
        X, df['queue_encoded'].values, df['urgency_encoded'].values,
        test_size=0.2, random_state=42
    )

    # Sauvegarde temporaire
    np.save('/tmp/X_train.npy', X_train)
    np.save('/tmp/X_test.npy', X_test)
    np.save('/tmp/yq_train.npy', yq_train)
    np.save('/tmp/yq_test.npy', yq_test)
    np.save('/tmp/yu_train.npy', yu_train)
    np.save('/tmp/yu_test.npy', yu_test)

    context['ti'].xcom_push(key='n_classes_queue', value=int(len(le_queue.classes_)))
    context['ti'].xcom_push(key='n_classes_urgency', value=int(len(le_urgency.classes_)))
    context['ti'].xcom_push(key='data_size', value=len(df))

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

    # Chargement des donnees preparees
    X_train = np.load('/tmp/X_train.npy')
    X_test = np.load('/tmp/X_test.npy')
    yq_train = np.load('/tmp/yq_train.npy')
    yq_test = np.load('/tmp/yq_test.npy')
    yu_train = np.load('/tmp/yu_train.npy')
    yu_test = np.load('/tmp/yu_test.npy')

    n_classes_queue = context['ti'].xcom_pull(key='n_classes_queue', task_ids='load_data')
    n_classes_urgency = context['ti'].xcom_pull(key='n_classes_urgency', task_ids='load_data')

    logging.info(f"Donnees chargees: X_train={X_train.shape}, classes_queue={n_classes_queue}, classes_urgency={n_classes_urgency}")

    # Initialisation Ray avec dashboard accessible
    ray.init(
        ignore_reinit_error=True,
        num_cpus=4,
        dashboard_host="0.0.0.0",  # Accessible depuis l'exterieur
        dashboard_port=8265
    )

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
        scores = cross_val_score(model, X_train, yq_train, cv=3, scoring='f1_weighted', n_jobs=1)
        return {"f1_score": scores.mean()}

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

    best_queue = analysis_queue.best_config
    score_queue = analysis_queue.best_result["f1_score"]
    logging.info(f"Queue - Best F1: {score_queue:.4f}")
    logging.info(f"Queue - Best params: {best_queue}")

    # --- Recherche Urgency ---
    logging.info("Ray Tune - Optimisation modele URGENCY...")

    scheduler_urgency = ASHAScheduler(max_t=10, grace_period=1, reduction_factor=2)
    sample_weights = compute_sample_weight('balanced', yu_train)

    def train_urgency(config):
        model = XGBClassifier(
            **config,
            objective='multi:softprob',
            num_class=n_classes_urgency,
            eval_metric='mlogloss',
            random_state=42,
            n_jobs=2
        )
        model.fit(X_train, yu_train, sample_weight=sample_weights)
        preds = model.predict(X_test)
        f1 = f1_score(yu_test, preds, average='weighted')
        return {"f1_score": f1}

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

    best_urgency = analysis_urgency.best_config
    score_urgency = analysis_urgency.best_result["f1_score"]
    logging.info(f"Urgency - Best F1: {score_urgency:.4f}")
    logging.info(f"Urgency - Best params: {best_urgency}")

    ray.shutdown()

    # Push results
    context['ti'].xcom_push(key='best_queue', value=best_queue)
    context['ti'].xcom_push(key='best_urgency', value=best_urgency)
    context['ti'].xcom_push(key='score_queue', value=score_queue)
    context['ti'].xcom_push(key='score_urgency', value=score_urgency)


def save_hyperparameters(**context):
    """Sauvegarde les meilleurs hyperparametres dans JSON et MLflow"""
    import mlflow

    logging.info("=" * 60)
    logging.info("SAUVEGARDE DES HYPERPARAMETRES")
    logging.info("=" * 60)

    best_queue = context['ti'].xcom_pull(key='best_queue', task_ids='ray_tune_optimization')
    best_urgency = context['ti'].xcom_pull(key='best_urgency', task_ids='ray_tune_optimization')
    score_queue = context['ti'].xcom_pull(key='score_queue', task_ids='ray_tune_optimization')
    score_urgency = context['ti'].xcom_pull(key='score_urgency', task_ids='ray_tune_optimization')
    data_size = context['ti'].xcom_pull(key='data_size', task_ids='load_data')

    # Sauvegarde JSON
    hyperparams = {
        "queue": best_queue,
        "urgency": best_urgency,
        "scores": {
            "queue": score_queue,
            "urgency": score_urgency
        },
        "sample_size": SAMPLE_SIZE,
        "n_trials": N_TRIALS,
        "method": "Ray Tune + Optuna + ASHA",
        "timestamp": datetime.now().isoformat(),
        "triggered_by": "drift_detection"
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(hyperparams, f, indent=2)

    logging.info(f"Hyperparametres sauvegardes: {OUTPUT_FILE}")

    # Log dans MLflow
    try:
        mlflow.set_tracking_uri("http://host.docker.internal:5000")
        mlflow.set_experiment("hyperparameter_tuning")

        with mlflow.start_run(run_name=f"ray_tune_auto_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_param("method", "Ray Tune + Optuna + ASHA")
            mlflow.log_param("sample_size", SAMPLE_SIZE)
            mlflow.log_param("n_trials", N_TRIALS)
            mlflow.log_param("data_size", data_size)
            mlflow.log_param("triggered_by", "drift_detection")

            for k, v in best_queue.items():
                mlflow.log_param(f"queue_{k}", v)
            for k, v in best_urgency.items():
                mlflow.log_param(f"urgency_{k}", v)

            mlflow.log_metric("best_f1_queue", score_queue)
            mlflow.log_metric("best_f1_urgency", score_urgency)

        logging.info("Resultats enregistres dans MLflow")
    except Exception as e:
        logging.warning(f"Erreur MLflow (non bloquant): {e}")

    # Resume
    logging.info("=" * 60)
    logging.info("RESULTATS OPTIMISATION HYPERPARAMETRES")
    logging.info("=" * 60)
    logging.info(f"Queue   - F1: {score_queue:.4f}")
    logging.info(f"Urgency - F1: {score_urgency:.4f}")
    logging.info("=" * 60)
    logging.info("Declenchement du RETRAINING avec nouveaux hyperparametres...")


def cleanup(**context):
    """Nettoie les fichiers temporaires"""
    for f in ['/tmp/X_train.npy', '/tmp/X_test.npy',
              '/tmp/yq_train.npy', '/tmp/yq_test.npy',
              '/tmp/yu_train.npy', '/tmp/yu_test.npy']:
        if os.path.exists(f):
            os.remove(f)
    logging.info("Fichiers temporaires nettoyes")


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
    dag=dag,
)

# Pipeline complet
load_task >> ray_tune_task >> save_task >> cleanup_task >> trigger_retraining
