# Classification XGBoost + MLflow - Documentation Complète

## Ce que fait ce DAG (résumé rapide)

1. **Lit la table enrichie** (`tickets_tech_en_enriched`)
2. **Split stratifié** des données (Train/Val/Test)
3. **TF-IDF + features numériques** combinés
4. **Deux modèles XGBoost** (un pour queue, un pour urgency)
5. **Log tout dans MLflow** (params, metrics, reports, modèles, artifacts)

---

## C'est quoi un Split Stratifié ?

### Le problème du split classique

Imaginons qu'on a 100 tickets avec cette distribution:
- 70 tickets "Software"
- 20 tickets "Security"
- 10 tickets "Hardware"

Avec un split **aléatoire classique**, on pourrait avoir par malchance:
- Train: 60 Software, 20 Security, 5 Hardware
- Test: 10 Software, 0 Security, 5 Hardware ← **Pas de Security dans le test !**

### La solution : Split Stratifié

Le split **stratifié** garantit que **chaque ensemble (train/val/test) garde la même proportion de classes** que l'original.

```
Données originales:           Train (72%):              Test (15%):
┌─────────────────┐           ┌─────────────────┐       ┌─────────────────┐
│ Software   70%  │    ──▶    │ Software   70%  │       │ Software   70%  │
│ Security   20%  │           │ Security   20%  │       │ Security   20%  │
│ Hardware   10%  │           │ Hardware   10%  │       │ Hardware   10%  │
└─────────────────┘           └─────────────────┘       └─────────────────┘
                              Mêmes proportions !       Mêmes proportions !
```

### Dans notre code

```python
X_train, X_test, yq_train, yq_test, yu_train, yu_test = train_test_split(
    X, y_queue, y_urgency,
    test_size=0.15,
    random_state=42,
    stratify=df[['refined_queue', 'urgency_level']]  # ← Stratification sur les 2 targets
)
```

Le paramètre `stratify=` dit à sklearn de garder les proportions des classes.

### Pourquoi c'est important ?

| Sans stratification | Avec stratification |
|---------------------|---------------------|
| Classes rares peuvent disparaître du test | Toutes les classes sont représentées |
| Métriques biaisées | Métriques fiables |
| Modèle mal évalué | Évaluation réaliste |

C'est **crucial** quand on a des classes déséquilibrées (comme "Service Outages" qui ne représente que 2.6% des tickets).

---

## Vue d'ensemble

Ce document décrit le pipeline complet de classification des tickets de support technique avec tracking MLflow et Model Registry.

---

## Architecture Globale

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           PIPELINE MLOPS COMPLET                                     │
│                                                                                      │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐           │
│  │  PostgreSQL │───▶│   Airflow   │───▶│   MLflow    │───▶│  Registry   │           │
│  │   (Data)    │    │   (DAGs)    │    │ (Tracking)  │    │ (Versions)  │           │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘           │
│       5433              8082               5001              intégré                 │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Les 3 DAGs Airflow

```
┌────────────────────┐     ┌────────────────────┐     ┌────────────────────────────┐
│  DAG 1: Ingestion  │────▶│  DAG 2: Features   │────▶│  DAG 3: Classification     │
│                    │     │                    │     │                            │
│  - Parquet → SQL   │     │  - Nettoyage text  │     │  - TF-IDF + XGBoost        │
│  - 17,893 tickets  │     │  - Keywords detect │     │  - MLflow tracking         │
│                    │     │  - Evidently       │     │  - Model Registry          │
└────────────────────┘     └────────────────────┘     └────────────────────────────┘
     ingest_tickets             prep_tickets           classification_xgboost
       _tech_en                  _features                   _mlflow
```

---

## DAG Classification - Détails

### Informations générales

| Paramètre | Valeur |
|-----------|--------|
| **DAG ID** | `classification_tickets_xgboost_mlflow` |
| **Owner** | `jedha_bootcamp` |
| **Schedule** | Manuel (`None`) |
| **Tags** | `classification`, `xgboost`, `mlflow`, `tickets` |

### Workflow du DAG

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                    DAG: classification_tickets_xgboost_mlflow                     │
│                                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │                    Task: train_xgboost_and_log_mlflow                        │ │
│  │                                                                              │ │
│  │   1. Lecture données enrichies depuis PostgreSQL                            │ │
│  │      └─▶ SELECT * FROM tickets_tech_en_enriched (17,893 lignes)             │ │
│  │                                                                              │ │
│  │   2. Préparation features                                                    │ │
│  │      └─▶ text_combined = subject + body_clean                               │ │
│  │      └─▶ 8 features numériques (body_length, has_network, etc.)             │ │
│  │                                                                              │ │
│  │   3. Encodage des targets                                                    │ │
│  │      └─▶ LabelEncoder pour refined_queue (6 classes)                        │ │
│  │      └─▶ LabelEncoder pour urgency_level (3 classes)                        │ │
│  │                                                                              │ │
│  │   4. Split stratifié                                                         │ │
│  │      └─▶ Train: 12,927 (72%) | Val: 2,282 (13%) | Test: 2,684 (15%)         │ │
│  │                                                                              │ │
│  │   5. TF-IDF Vectorization                                                    │ │
│  │      └─▶ max_features=5000, ngram_range=(1,2)                               │ │
│  │                                                                              │ │
│  │   6. Entraînement XGBoost Queue (6 classes)                                  │ │
│  │      └─▶ ~8 minutes                                                          │ │
│  │                                                                              │ │
│  │   7. Entraînement XGBoost Urgency (3 classes)                                │ │
│  │      └─▶ ~5 minutes                                                          │ │
│  │                                                                              │ │
│  │   8. Logging dans MLflow                                                     │ │
│  │      └─▶ Params, Metrics, Models, Artifacts                                  │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Features utilisées

### Features textuelles (TF-IDF)

| Paramètre | Valeur |
|-----------|--------|
| Source | `subject` + `body_clean` combinés |
| max_features | 5000 |
| ngram_range | (1, 2) |

### Features numériques

| Feature | Description |
|---------|-------------|
| `body_length` | Longueur du corps du ticket |
| `answer_length` | Longueur de la réponse |
| `response_ratio` | Ratio réponse/problème |
| `has_network` | Contient mots-clés réseau (wifi, vpn, router...) |
| `has_printer` | Contient mots-clés imprimante |
| `has_security` | Contient mots-clés sécurité (password, login...) |
| `has_hardware` | Contient mots-clés hardware (laptop, disk...) |
| `has_software` | Contient mots-clés software (app, update, bug...) |

---

## Targets (Classes à prédire)

### refined_queue (6 classes)

| Classe | Distribution |
|--------|-------------|
| Software / Product | 36.9% |
| General Technical Support | 26.1% |
| Security / Access | 23.3% |
| Network / Infrastructure | 6.7% |
| Hardware / Device | 4.4% |
| Service Outages | 2.6% |

### urgency_level (3 classes)

| Niveau | Distribution |
|--------|-------------|
| high | 48.6% |
| medium | 37.8% |
| low | 13.6% |

---

## Paramètres XGBoost

```python
params = {
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 200,
    'random_state': 42,
}
```

---

## Résultats Finaux

### Métriques globales

| Modèle | Accuracy | F1 Score (weighted) |
|--------|----------|---------------------|
| **Queue (6 classes)** | **89.87%** | **89.88%** |
| **Urgency (3 classes)** | **68.74%** | **66.55%** |

### Classification Report - Queue

```
                           precision    recall  f1-score   support

General Technical Support       0.75      0.94      0.83       702
        Hardware / Device       1.00      1.00      1.00       118
 Network / Infrastructure       1.00      1.00      1.00       179
        Security / Access       1.00      1.00      1.00       625
          Service Outages       0.95      0.54      0.69        69
       Software / Product       0.95      0.80      0.87       991

                 accuracy                           0.90      2684
                macro avg       0.94      0.88      0.90      2684
             weighted avg       0.91      0.90      0.90      2684
```

### Classification Report - Urgency

```
              precision    recall  f1-score   support

        high       0.67      0.89      0.76      1304
         low       0.91      0.26      0.40       365
      medium       0.71      0.58      0.64      1015

    accuracy                           0.69      2684
   macro avg       0.76      0.58      0.60      2684
weighted avg       0.71      0.69      0.67      2684
```

---

## MLflow - Configuration

### Tracking URI

```python
# Écriture directe sur disque (évite problèmes de permission Docker)
mlflow.set_tracking_uri("file:///opt/airflow/mlruns")
mlflow.set_experiment("tickets_classification_bootcamp")
```

### Volumes Docker

```yaml
# docker-compose-airflow.yaml
volumes:
  - ./dags:/opt/airflow/dags
  - ./logs:/opt/airflow/logs
  - ./data:/opt/airflow/data
  - ./mlruns:/opt/airflow/mlruns  # MLflow artifacts
```

### Lancer l'UI MLflow

```bash
cd /home/isabelle/projects/bootcamp/jedha/projet_mlops_support
mlflow ui --backend-store-uri file:///home/isabelle/projects/bootcamp/jedha/projet_mlops_support/mlruns --host 0.0.0.0 --port 5001
```

Accès: `http://<IP>:5001`

---

## Artifacts sauvegardés dans MLflow

| Artifact | Description |
|----------|-------------|
| `xgboost_queue/` | Modèle XGBoost pour refined_queue (6 classes) |
| `xgboost_urgency/` | Modèle XGBoost pour urgency_level (3 classes) |
| `tfidf_vectorizer.pkl` | Vectorizer TF-IDF (5000 features) |
| `le_queue.pkl` | LabelEncoder pour queue |
| `le_urgency.pkl` | LabelEncoder pour urgency |
| `classification_report_queue.txt` | Rapport détaillé queue |
| `classification_report_urgency.txt` | Rapport détaillé urgency |

### Avantages de tout centraliser dans MLflow

- **Versioning complet** : chaque run garde son modèle + vectorizer + encoders
- **Reproductibilité** : on peut revenir à n'importe quelle version
- **Déploiement facile** : un seul Run ID pour tout récupérer

---

## Model Registry

### Modèles enregistrés

| Registry Name | Version | Accuracy | F1 Score |
|---------------|---------|----------|----------|
| `ticket_queue_classifier` | v1 | 89.87% | 89.88% |
| `ticket_urgency_classifier` | v1 | 68.74% | 66.55% |

### Workflow Model Registry

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           MODEL REGISTRY WORKFLOW                                │
│                                                                                  │
│   ┌──────────┐      ┌──────────┐      ┌────────────┐      ┌──────────┐          │
│   │  MLflow  │─────▶│ Register │─────▶│  Staging   │─────▶│Production│          │
│   │   Run    │      │  Model   │      │   Tests    │      │  Deploy  │          │
│   └──────────┘      └──────────┘      └────────────┘      └──────────┘          │
│                           │                                     │               │
│                           ▼                                     ▼               │
│                    ticket_queue_classifier v1          API FastAPI / Batch      │
│                    ticket_urgency_classifier v1                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### À quoi sert le Model Registry ?

| Sans Registry | Avec Registry |
|---------------|---------------|
| "Quel modèle est en prod ?" → Run ID obscur | "Quel modèle est en prod ?" → `v3 Production` |
| Pas de contrôle des versions | Versions numérotées (v1, v2, v3...) |
| Déploiement manuel | Transitions: Staging → Production → Archived |
| Difficile de revenir en arrière | Rollback facile vers version précédente |

---

## Utilisation des modèles

### Charger depuis le Run ID

```python
import mlflow

# Charger les modèles depuis un run spécifique
model_queue = mlflow.pyfunc.load_model('runs:/50ac57fb70834c5ca4addd76191d4b71/xgboost_queue')
model_urgency = mlflow.pyfunc.load_model('runs:/50ac57fb70834c5ca4addd76191d4b71/xgboost_urgency')
```

### Charger depuis le Registry (recommandé)

```python
import mlflow

# Charger depuis le Model Registry (plus propre, versionné)
model_queue = mlflow.pyfunc.load_model("models:/ticket_queue_classifier/1")
model_urgency = mlflow.pyfunc.load_model("models:/ticket_urgency_classifier/1")

# Ou charger la version "Production"
model_queue = mlflow.pyfunc.load_model("models:/ticket_queue_classifier/Production")
```

### Charger les artifacts (vectorizer, encoders)

```python
import mlflow
import joblib
import os

# Télécharger les artifacts
run_id = "50ac57fb70834c5ca4addd76191d4b71"
artifacts_path = mlflow.artifacts.download_artifacts(run_id=run_id)

# Charger le vectorizer et les encoders
tfidf = joblib.load(os.path.join(artifacts_path, "tfidf_vectorizer.pkl"))
le_queue = joblib.load(os.path.join(artifacts_path, "le_queue.pkl"))
le_urgency = joblib.load(os.path.join(artifacts_path, "le_urgency.pkl"))
```

### Faire une prédiction complète

```python
import pandas as pd
from scipy.sparse import hstack

# Nouveau ticket à classifier
new_ticket = {
    'subject': 'Cannot connect to VPN',
    'body_clean': 'I cannot connect to the corporate VPN since this morning. Error message: connection timeout.',
    'body_length': 95,
    'answer_length': 0,
    'response_ratio': 0,
    'has_network': 1,
    'has_printer': 0,
    'has_security': 0,
    'has_hardware': 0,
    'has_software': 0
}

# Préparer les features
text_combined = new_ticket['subject'] + " " + new_ticket['body_clean']
num_features = ['body_length', 'answer_length', 'response_ratio',
                'has_network', 'has_printer', 'has_security', 'has_hardware', 'has_software']

# Vectoriser le texte
X_text = tfidf.transform([text_combined])
X_num = pd.DataFrame([new_ticket])[num_features]
X_full = hstack([X_text, X_num])

# Prédictions
queue_pred = model_queue.predict(X_full)
urgency_pred = model_urgency.predict(X_full)

# Décoder les résultats
queue_label = le_queue.inverse_transform(queue_pred)
urgency_label = le_urgency.inverse_transform(urgency_pred)

print(f"Queue: {queue_label[0]}")      # → Network / Infrastructure
print(f"Urgency: {urgency_label[0]}")  # → high
```

---

## Connexions

### PostgreSQL

| Paramètre | Valeur |
|-----------|--------|
| Host | `host.docker.internal` (depuis Docker) |
| Port | `5433` |
| Database | `support_tech` |
| User | `bootcamp_user` |
| Password | `bootcamp_password` |
| Table source | `tickets_tech_en_enriched` |

### MLflow

| Paramètre | Valeur |
|-----------|--------|
| Tracking URI | `file:///opt/airflow/mlruns` |
| Experiment | `tickets_classification_bootcamp` |
| UI Port | `5001` |

---

## Exécution

### Prérequis

1. PostgreSQL avec table `tickets_tech_en_enriched` (DAG 2 exécuté)
2. Docker Compose Airflow lancé
3. Volume `mlruns` monté

### Déclencher le DAG

```bash
# Via CLI
sudo docker exec support_airflow_scheduler airflow dags trigger classification_tickets_xgboost_mlflow

# Ou via UI Airflow: http://<IP>:8082
```

### Vérifier les résultats

1. **Airflow UI** : `http://<IP>:8082` → Logs du DAG
2. **MLflow UI** : `http://<IP>:5001` → Experiment `tickets_classification_bootcamp`

---

## Fichiers du projet

```
projet_mlops_support/
├── docker-compose-airflow.yaml      # Airflow + volumes
├── docker-compose-db.yaml           # PostgreSQL
├── dags/
│   ├── ingest_tickets_tech.py       # DAG 1: Ingestion
│   ├── ingest_tickets_tech.md
│   ├── prep_tickets_features.py     # DAG 2: Features
│   ├── prep_tickets_features.md
│   ├── classification_xgboost_mlflow.py  # DAG 3: Classification
│   └── classification_xgboost_mlflow.md
├── data/
│   ├── tickets_tech_en.parquet      # Données source
│   ├── evidently_quality_report.html
│   └── evidently_drift_report.html
├── mlruns/                          # MLflow artifacts
│   └── 442234482716700024/          # Experiment ID
│       └── 50ac57fb70834c5ca4addd76191d4b71/  # Run ID
│           ├── artifacts/
│           │   ├── xgboost_queue/
│           │   ├── xgboost_urgency/
│           │   ├── tfidf_vectorizer.pkl
│           │   ├── le_queue.pkl
│           │   ├── le_urgency.pkl
│           │   └── classification_report_*.txt
│           ├── metrics/
│           ├── params/
│           └── meta.yaml
└── docs/
    └── classification_mlflow_complete.md  # Ce document
```

---

## Résumé

Ce pipeline MLOps implémente une solution complète de classification de tickets de support technique:

1. **Ingestion** : Données HuggingFace → PostgreSQL
2. **Feature Engineering** : Nettoyage + features + Evidently reports
3. **Classification** : XGBoost multi-classe avec TF-IDF
4. **Tracking** : MLflow pour params, metrics, artifacts
5. **Registry** : Versioning des modèles pour production

**Performance finale** : 90% accuracy sur la catégorisation des tickets (6 classes).
