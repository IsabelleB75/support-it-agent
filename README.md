# Agent Support IT - MLOps

Agent de support technique automatise utilisant le Machine Learning pour classifier les tickets et generer des reponses pertinentes.

## Resultats

| Metrique | Score |
|----------|-------|
| **F1 Score Queue** | **91.2%** (6 classes) |
| **F1 Score Urgence** | **78.1%** (3 classes) |
| Inference XGBoost | ~1ms |
| Tickets d'entrainement | 17 893 |

## Pipeline 100% Automatise

```
Drift detecte → Hyperparameter Tuning → Retraining → DVC push → CI/CD → Deploy K3s
     ↑                                                                      │
     └──────────────────────── logs prediction ────────────────────────────┘
```

**Aucune intervention manuelle requise** entre la detection de drift et le deploiement du nouveau modele.

## Architecture MLOps

```
┌─────────────────────────┐              ┌─────────────────────────┐              ┌─────────────────────────┐
│  1) DATA PIPELINE       │   datasets   │  2) TRAINING PIPELINE   │   modeles   │  3) DEPLOYMENT          │
│                         │   (DVC/S3)   │                         │ (DVC+MLflow)│                         │
│  PostgreSQL:            │      →       │  XGBoost: queue+urgence │      →      │  Docker build → ghcr.io │
│   • tickets_tech_en     │              │  Ray Tune + Optuna:     │             │  GitHub Actions: CI/CD  │
│   • ticket_embeddings   │              │   hyperparametres       │             │  K3s: 2 pods NodePort   │
│  PGVector: rag_docs     │              │  MLflow: Tracking +     │             │                         │
│  DVC → S3: versioning   │              │   Model Registry        │             │  DAG: retraining        │
│                         │              │                         │             │  (deploy_model task)    │
│  DAGs: ingest + prep    │              │  DAGs: hyperparameter_  │             │   • DVC push S3         │
│        + rag            │              │  tuning + retraining    │             │   • Trigger GHA API     │
└─────────────────────────┘              └────────────▲────────────┘              └───────────┬─────────────┘
                                                      │                                       │ image Docker
                                                      │ trigger si drift                      ↓
┌─────────────────────────┐              ┌────────────┴────────────────────────────────────────────────────┐
│  5) MONITORING          │     logs     │                     4) SERVING (temps reel)                     │
│                         │  (PostgreSQL)│                                                                 │
│  Evidently: drift       │      ←       │  FastAPI: /predict, /feedback, /health                         │
│   (seuil configurable)  │              │  XGBoost: classification (~1ms)                                 │
│  Rapport HTML auto      │──────────────│  RAG (PGVector) + Mistral: reponse enrichie                    │
│  Trigger si drift ──────┼──────┘       │  Logs → prediction_logs (PostgreSQL)                           │
│                         │              │                                                                 │
│  DAG: monitoring_       │              │  Kubernetes: 2 replicas, port 30080                            │
│  evidently (6h)         │              │                                                                 │
└─────────────────────────┘              └─────────────────────────────────────────────────────────────────┘
```

## Classification des Tickets

### Les 6 Queues (Categories)

Le modele classe automatiquement les tickets dans la bonne **queue** (file d'attente) :

| Queue | Description | Tickets |
|-------|-------------|---------|
| **Software / Product** | Problemes logiciels, applications, licences | 6 564 |
| **General Technical Support** | Support technique general | 4 649 |
| **Security / Access** | Mots de passe, acces, permissions, VPN | 4 160 |
| **Network / Infrastructure** | Reseau, connexion, infrastructure | 1 196 |
| **Hardware / Device** | Materiel, ordinateurs, imprimantes | 783 |
| **Service Outages** | Pannes de service, incidents | 461 |

### Les 3 Niveaux d'Urgence

| Niveau | Description |
|--------|-------------|
| **High** | Critique, impact business immediat |
| **Medium** | Important mais non bloquant |
| **Low** | Faible priorite |

## Dataset

- **Source** : HuggingFace (Tobi-Bueck/customer-support-tickets)
- **Original** : 61 800 tickets synthetiques (EN + DE)
- **Filtre** : 17 893 tickets retenus (anglais, IT)
- **52 queues** regroupees en **6 categories**
- **5 priorites** regroupees en **3 niveaux**

## Stack Technique

| Categorie | Technologies |
|-----------|--------------|
| **Orchestration** | Airflow (DAGs, XCOM, TriggerDagRunOperator) |
| **ML Training** | XGBoost, Scikit-learn, SentenceTransformers |
| **Hyperparameter Tuning** | Ray Tune, Optuna (TPE Sampler) |
| **Experiment Tracking** | MLflow (Tracking + Model Registry) |
| **Data Versioning** | DVC + AWS S3 |
| **Monitoring** | Evidently (DataDriftPreset) |
| **Serving** | FastAPI, Uvicorn |
| **RAG** | PGVector, Mistral API |
| **CI/CD** | GitHub Actions (workflow_dispatch) |
| **Container** | Docker, GHCR |
| **Orchestration K8s** | K3s (2 replicas, NodePort) |
| **Database** | PostgreSQL + pgvector |

## DAGs Airflow

| DAG | Schedule | Description |
|-----|----------|-------------|
| `monitoring_evidently` | `0 6 * * *` (6h) | Detection drift Evidently → trigger training |
| `hyperparameter_tuning` | Triggered | Ray Tune + Optuna (25 trials) → best hyperparams |
| `retraining_pipeline` | Triggered | Train XGBoost + MLflow + DVC push + GHA trigger |

## Pipeline CI/CD (GitHub Actions)

```
┌─────────────┐      ┌─────────────────┐      ┌─────────────┐
│    test     │ ───→ │  build-and-push │ ───→ │   deploy    │
├─────────────┤      ├─────────────────┤      ├─────────────┤
│ pytest      │      │ dvc pull (S3)   │      │ kubectl     │
│             │      │ docker build    │      │ rollout     │
│             │      │ push ghcr.io    │      │ K3s cluster │
└─────────────┘      └─────────────────┘      └─────────────┘
```

Declenchement automatique :
- A chaque push sur `main`
- **Automatiquement apres retraining** (via GitHub API workflow_dispatch)

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
# Editer .env avec vos cles (AWS, GITHUB_TOKEN, etc.)

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
| `/predict` | POST | Prediction queue + urgence + reponse RAG |
| `/feedback` | POST | Feedback utilisateur pour retraining |

### Exemple

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"user_query": "Mon VPN ne marche pas"}'
```

Reponse :
```json
{
  "predicted_queue": "Security / Access",
  "predicted_urgency": "Medium",
  "confidence_queue": 0.92,
  "confidence_urgency": 0.85,
  "response": "Pour resoudre votre probleme VPN..."
}
```

## URLs des Services (Production)

| Service | URL |
|---------|-----|
| API Support Agent | http://xx.xx.xxx.xxx:30080 |
| API Docs (Swagger) | http://xx.xx.xxx.xxx:30080/docs |
| MLflow | http://xx.xx.xxx.xxx:5000 |
| Airflow | http://xx.xx.xxx.xxx:8082 |
| Evidently Reports | http://xx.xx.xxx.xxx:8083 |

## Structure du Projet

```
projet_mlops_support/
├── app.py                      # API FastAPI
├── Dockerfile                  # Image Docker
├── requirements.txt            # Dependances Python
├── dags/                       # DAGs Airflow
│   ├── 5_dags_monitoring/
│   │   └── monitoring_evidently.py
│   ├── 6_dags_retraining/
│   │   ├── hyperparameter_tuning_dag.py
│   │   └── retraining_pipeline.py
│   └── utils/
│       └── db_config.py
├── k8s/                        # Manifests Kubernetes
│   ├── deployment.yaml
│   └── service.yaml
├── .github/workflows/          # CI/CD
│   └── ci-cd.yaml
└── dvc_project/                # Repo DVC (modeles versiones)
```

## Licence

Projet realise dans le cadre du bootcamp Jedha - Data Engineering & MLOps.
