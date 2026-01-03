# Agent Support IT - MLOps

Agent de support technique automatise utilisant le Machine Learning pour classifier les demandes et generer des reponses pertinentes.

## Pipeline 100% Automatise

Ce projet implemente un pipeline MLOps **entierement automatise** :

```
Drift detecte → Retraining → Validation → DVC push → CI/CD → Deploy K3s
```

**Aucune intervention manuelle requise** entre la detection de drift et le deploiement du nouveau modele.

## Features

- Classification automatique des tickets (queue + urgence) avec XGBoost
- Recherche semantique dans la base de connaissances (RAG avec Sentence Transformers)
- Generation de reponses avec Mistral API
- Monitoring de drift avec Evidently
- Retraining automatique si drift detecte
- CI/CD avec GitHub Actions + deploiement K3s

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER REQUEST                            │
│                    "Mon VPN ne marche pas"                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API FastAPI (K3s)                          │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
      ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
      │  XGBoost    │ │  Sentence   │ │   Mistral   │
      │  Classifier │ │ Transformers│ │     API     │
      │ (queue +    │ │    (RAG)    │ │  (reponse)  │
      │  urgence)   │ │             │ │             │
      └─────────────┘ └─────────────┘ └─────────────┘
```

## Pipeline MLOps Complet

```
   ┌───────────────────────┐       ┌───────────────────────┐       ┌───────────────────────┐
   │   1. DATA PIPELINE    │       │  2. TRAINING PIPELINE │       │    3. DEPLOYMENT      │
   ├───────────────────────┤       ├───────────────────────┤       ├───────────────────────┤
   │ PostgreSQL            │       │ Airflow DAG training  │       │ Docker: build image   │
   │ - tickets_tech_en     │features│ - XGBoost (queue +   │modele │ - ghcr.io registry    │
   │ - prediction_logs     │──────>│   urgence)            │──────>│                       │
   │ - embeddings (PGVector)│      │ - Ray Tune (hyperparam)│       │ GitHub Actions: CI/CD │
   │                       │       │                       │       │                       │
   │ Airflow DAGs:         │       │ MLflow:               │       │ K3s: Deployment       │
   │ - ingest_tickets      │       │ - tracking experiments│       │ - 2 pods              │
   │ - prep_features       │       │ - model registry      │       │ - NodePort :30080     │
   │ - ingestion_rag       │       │                       │       │                       │
   │                       │       │                       │       │                       │
   │ DVC: versioning data  │       │                       │       │                       │
   │ - S3 storage          │       │                       │       │                       │
   └───────────────────────┘       └───────────────────────┘       └───────────────────────┘
   ▲                                                                          │
   │                                                                          │
   │                                                                          ▼
   │ ┌───────────────────────┐       ┌───────────────────────────────────────────────────────┐
   │ │    5. MONITORING      │       │              4. SERVING (temps reel)                  │
   │ ├───────────────────────┤       ├───────────────────────────────────────────────────────┤
   │ │ Airflow DAG:          │       │  Question utilisateur: "Mon VPN ne marche pas"       │
   │ │ - monitoring_evidently│       │                          │                           │
   │ │                       │       │                          ▼                           │
   │ │ prediction_logs:      │ logs  │  FastAPI (http://78.47.129.250:30080)                 │
   │ │ - predictions         │<──────│  - XGBoost: prediction queue + urgence               │
   │ │ - feedback            │       │  - Sentence Transformers: RAG                        │
   │ │                       │       │  - Mistral API: generation reponse                   │
   │ │ Evidently lit les logs│       │                          │                           │
   │ │                       │       │                          ▼                           │
   │ │  ┌─────┐  ┌──────┐    │       │  Feedback utilisateur → prediction_logs              │
   │ │  │ OK  │  │DRIFT │    │       └───────────────────────────────────────────────────────┘
   │ │  └──┬──┘  └──┬───┘    │
   │ │     │        │        │
   │ │  (rien)   retrain     │
   │ └──────────────┼────────┘
   │                │
   │                ▼
   │        ┌──────────────┐
   │        │ DAG retrain  │
   │        └──────────────┘
   │                │
   └────────────────┘
```

## Stack Technique

| Composant | Technologie |
|-----------|-------------|
| API | FastAPI |
| Classification | XGBoost |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| LLM | Mistral API |
| Base de donnees | PostgreSQL + PGVector |
| Orchestration | Airflow |
| Tracking ML | MLflow |
| Versioning data | DVC + S3 |
| Monitoring | Evidently |
| Hyperparametres | Ray Tune |
| Conteneurisation | Docker |
| Orchestration K8s | K3s |
| CI/CD | GitHub Actions |

## Les 6 Composants MLOps

| # | Composant | Implementation |
|---|-----------|----------------|
| 1 | Pipelines de donnees | Airflow DAGs (ingestion, features, RAG) |
| 2 | Versioning | DVC (donnees sur S3) + MLflow (modeles) |
| 3 | Monitoring | Evidently (detection drift) via Airflow |
| 4 | Retraining | DAG retraining_pipeline automatique |
| 5 | Hyperparametres | Ray Tune via Airflow |
| 6 | Deploiement | GitHub Actions CI/CD + K3s |

## Installation

### Prerequis

- Python 3.10+
- Docker & Docker Compose
- Acces AWS S3 (pour DVC)

### Lancer le projet

```bash
# 1. Cloner le repo
git clone https://github.com/IsabelleB75/support-it-agent.git
cd support-it-agent

# 2. Configurer les variables d'environnement
cp .env.example .env
# Editer .env avec vos cles

# 3. Recuperer les modeles depuis S3
pip install dvc dvc-s3
dvc pull

# 4. Lancer les services
docker-compose -f docker-compose-db.yaml up -d
docker-compose -f docker-compose-airflow.yaml up -d

# 5. Lancer l'API
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Endpoints API

| Endpoint | Methode | Description |
|----------|---------|-------------|
| `/health` | GET | Health check |
| `/predict` | POST | Envoie une question, recoit prediction + reponse |
| `/feedback` | POST | Feedback utilisateur pour retraining |

### Exemple

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"user_query": "Mon VPN ne marche pas"}'
```

## Structure du Projet

```
projet_mlops_support/
├── app.py                      # API FastAPI
├── Dockerfile                  # Image Docker
├── requirements.txt            # Dependances Python
├── dags/                       # DAGs Airflow
│   ├── ingest_tickets_tech.py
│   ├── prep_tickets_features.py
│   ├── classification_xgboost_mlflow.py
│   ├── hyperparameter_tuning_dag.py
│   ├── monitoring_evidently.py
│   ├── retraining_pipeline.py
│   └── ingestion_rag_pgvector.py
├── k8s/                        # Manifests Kubernetes
│   ├── deployment.yaml
│   └── service.yaml
├── .github/workflows/          # CI/CD
│   └── ci-cd.yaml
├── docs/                       # Documentation
└── data/                       # Donnees (via DVC)
```

## URLs des Services (Production)

| Service | URL |
|---------|-----|
| API Support Agent | http://78.47.129.250:30080 |
| API Docs (Swagger) | http://78.47.129.250:30080/docs |
| MLflow | http://78.47.129.250:5000 |
| Airflow | http://78.47.129.250:8082 |
| Evidently Reports | http://78.47.129.250:8083 |

## CI/CD Pipeline

```
git push → Tests → Build Docker → Push ghcr.io → Deploy K3s
```

Le pipeline CI/CD est declenche automatiquement :
- A chaque push sur `main`
- **Automatiquement apres retraining** (via GitHub API workflow_dispatch)

Cela permet un deploiement **100% automatique** des nouveaux modeles sans intervention manuelle.

## Licence

Projet realise dans le cadre du bootcamp Jedha - Data Engineering & MLOps.
