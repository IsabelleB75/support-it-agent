"""
Script Ray Tune pour optimisation des hyperparametres XGBoost
A executer une fois (ou occasionnellement) pour trouver les meilleurs params
Les resultats sont sauvegardes dans MLflow et reutilises par le DAG de retraining
"""

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer
import mlflow
import joblib
import json
import os
from sqlalchemy import create_engine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration - FULL DATASET
SAMPLE_SIZE = None  # None = tout le dataset (~17k)
N_TRIALS = 15  # Nombre de trials
MAX_CONCURRENT = 1  # 1 seul trial a la fois pour economiser RAM
MLFLOW_URI = "http://78.47.129.250:5000"
# Sauvegarde dans le repertoire scripts (accessible en ecriture)
OUTPUT_DIR = "/home/isabelle/projects/bootcamp/jedha/projet_mlops_support/scripts"

def load_and_prepare_data():
    """Charge les donnees et prepare les features"""
    logger.info("Chargement des donnees...")

    # Connexion PostgreSQL
    engine = create_engine('postgresql://bootcamp_user:bootcamp_password@localhost:5433/support_tech')
    df = pd.read_sql("SELECT * FROM tickets_tech_en_enriched", engine)

    logger.info(f"Dataset complet: {len(df)} tickets")

    # Echantillonnage stratifie si SAMPLE_SIZE est defini
    if SAMPLE_SIZE is not None and len(df) > SAMPLE_SIZE:
        df = df.groupby(['refined_queue', 'urgency_level'], group_keys=False).apply(
            lambda x: x.sample(n=max(1, int(len(x) * SAMPLE_SIZE / len(df.dropna(subset=['refined_queue', 'urgency_level'])))),
                              random_state=42),
            include_groups=False
        ).reset_index(drop=True)
        logger.info(f"Echantillon pour tuning: {len(df)} tickets")
    else:
        logger.info(f"Utilisation du dataset COMPLET: {len(df)} tickets")

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
    logger.info("Generation des embeddings...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedder.encode(df['text_combined'].tolist(), show_progress_bar=True)

    # Features completes
    X = np.hstack([embeddings, df[num_features].values])
    y_queue = df['queue_encoded'].values
    y_urgency = df['urgency_encoded'].values

    # Split
    X_train, X_test, yq_train, yq_test, yu_train, yu_test = train_test_split(
        X, y_queue, y_urgency, test_size=0.2, random_state=42
    )

    return {
        'X_train': X_train, 'X_test': X_test,
        'yq_train': yq_train, 'yq_test': yq_test,
        'yu_train': yu_train, 'yu_test': yu_test,
        'le_queue': le_queue, 'le_urgency': le_urgency,
        'n_classes_queue': len(le_queue.classes_),
        'n_classes_urgency': len(le_urgency.classes_)
    }


def train_queue_model(config, data):
    """Fonction d'entrainement pour Ray Tune - modele Queue"""
    model = XGBClassifier(
        max_depth=config["max_depth"],
        learning_rate=config["learning_rate"],
        n_estimators=config["n_estimators"],
        min_child_weight=config["min_child_weight"],
        subsample=config["subsample"],
        colsample_bytree=config["colsample_bytree"],
        objective='multi:softprob',
        num_class=data['n_classes_queue'],
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=2  # Limite CPU pour economiser memoire
    )

    # Cross-validation pour score robuste
    scores = cross_val_score(model, data['X_train'], data['yq_train'],
                            cv=3, scoring='f1_weighted', n_jobs=1)

    return {"f1_score": scores.mean()}


def train_urgency_model(config, data):
    """Fonction d'entrainement pour Ray Tune - modele Urgency"""
    sample_weights = compute_sample_weight('balanced', data['yu_train'])

    model = XGBClassifier(
        max_depth=config["max_depth"],
        learning_rate=config["learning_rate"],
        n_estimators=config["n_estimators"],
        min_child_weight=config["min_child_weight"],
        subsample=config["subsample"],
        colsample_bytree=config["colsample_bytree"],
        objective='multi:softprob',
        num_class=data['n_classes_urgency'],
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=2  # Limite CPU pour economiser memoire
    )

    model.fit(data['X_train'], data['yu_train'], sample_weight=sample_weights)
    preds = model.predict(data['X_test'])
    f1 = f1_score(data['yu_test'], preds, average='weighted')

    return {"f1_score": f1}


def run_ray_tune_search(data, model_type="queue"):
    """Execute la recherche Ray Tune"""

    # Espace de recherche des hyperparametres
    search_space = {
        "max_depth": tune.choice([3, 4, 5, 6, 7, 8]),
        "learning_rate": tune.loguniform(0.01, 0.3),
        "n_estimators": tune.choice([50, 100, 150, 200, 250]),
        "min_child_weight": tune.choice([1, 3, 5, 7]),
        "subsample": tune.uniform(0.6, 1.0),
        "colsample_bytree": tune.uniform(0.6, 1.0),
    }

    # Scheduler ASHA pour early stopping des mauvais trials
    scheduler = ASHAScheduler(
        max_t=10,
        grace_period=1,
        reduction_factor=2
    )

    # Optuna pour recherche intelligente
    search_alg = OptunaSearch()

    train_fn = train_queue_model if model_type == "queue" else train_urgency_model

    logger.info(f"Lancement Ray Tune pour modele {model_type}...")

    # Wrap la fonction pour passer les donnees
    def trainable(config):
        return train_fn(config, data)

    # Lancement de la recherche - limite la concurrence pour economiser la RAM
    analysis = tune.run(
        trainable,
        config=search_space,
        num_samples=N_TRIALS,
        scheduler=scheduler,
        search_alg=search_alg,
        metric="f1_score",
        mode="max",
        verbose=1,
        resources_per_trial={"cpu": 1},  # 1 CPU par trial
        max_concurrent_trials=MAX_CONCURRENT,  # Max 2 en parallele
    )

    best_config = analysis.best_config
    best_score = analysis.best_result["f1_score"]

    logger.info(f"Meilleurs hyperparametres {model_type}: {best_config}")
    logger.info(f"Meilleur F1 score: {best_score:.4f}")

    return best_config, best_score


def save_hyperparams(best_params_queue, best_params_urgency, scores):
    """Sauvegarde les hyperparametres dans MLflow et localement"""

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Sauvegarde locale JSON
    hyperparams = {
        "queue": best_params_queue,
        "urgency": best_params_urgency,
        "scores": scores,
        "sample_size": SAMPLE_SIZE if SAMPLE_SIZE else "full_dataset",
        "n_trials": N_TRIALS
    }

    with open(f"{OUTPUT_DIR}/best_hyperparams.json", "w") as f:
        json.dump(hyperparams, f, indent=2)

    logger.info(f"Hyperparametres sauvegardes dans {OUTPUT_DIR}/best_hyperparams.json")

    # Sauvegarde MLflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("hyperparameter_tuning")

    with mlflow.start_run(run_name="ray_tune_xgboost"):
        # Log parametres
        mlflow.log_param("sample_size", SAMPLE_SIZE)
        mlflow.log_param("n_trials", N_TRIALS)
        mlflow.log_param("method", "Ray Tune + Optuna + ASHA")

        # Log best params queue
        for k, v in best_params_queue.items():
            mlflow.log_param(f"queue_{k}", v)

        # Log best params urgency
        for k, v in best_params_urgency.items():
            mlflow.log_param(f"urgency_{k}", v)

        # Log scores
        mlflow.log_metric("best_f1_queue", scores["queue"])
        mlflow.log_metric("best_f1_urgency", scores["urgency"])

        # Log artifact (avec gestion erreur permissions)
        try:
            mlflow.log_artifact(f"{OUTPUT_DIR}/best_hyperparams.json")
            logger.info("Artifact sauvegarde dans MLflow")
        except PermissionError as e:
            logger.warning(f"Impossible de sauvegarder l'artifact dans MLflow: {e}")
            logger.info("Les parametres et metriques sont quand meme enregistres")

        logger.info("Hyperparametres enregistres dans MLflow")


def main():
    """Point d'entree principal"""
    logger.info("=" * 60)
    logger.info("OPTIMISATION HYPERPARAMETRES AVEC RAY TUNE")
    logger.info("=" * 60)

    # Initialisation Ray avec dashboard accessible
    ray.init(
        ignore_reinit_error=True,
        object_store_memory=2 * 1024 * 1024 * 1024,  # 2GB max pour object store
        include_dashboard=True,
        dashboard_host="0.0.0.0",  # Accessible depuis l'exterieur
        dashboard_port=8265,
    )

    try:
        # Chargement donnees
        data = load_and_prepare_data()

        # Recherche hyperparametres Queue
        best_params_queue, score_queue = run_ray_tune_search(data, "queue")

        # Recherche hyperparametres Urgency
        best_params_urgency, score_urgency = run_ray_tune_search(data, "urgency")

        # Sauvegarde
        scores = {"queue": score_queue, "urgency": score_urgency}
        save_hyperparams(best_params_queue, best_params_urgency, scores)

        logger.info("=" * 60)
        logger.info("RESULTATS FINAUX")
        logger.info("=" * 60)
        logger.info(f"Queue - Best F1: {score_queue:.4f}")
        logger.info(f"  Params: {best_params_queue}")
        logger.info(f"Urgency - Best F1: {score_urgency:.4f}")
        logger.info(f"  Params: {best_params_urgency}")
        logger.info("=" * 60)

    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
