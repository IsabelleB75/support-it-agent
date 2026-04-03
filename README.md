# Agent Support IT - MLOps

Agent de support technique qui classifie les tickets et genere des reponses avec du RAG.

## Resultats

| Metrique | Score |
|----------|-------|
| F1 Queue | 0.9392 (6 classes) |
| F1 Urgence | 0.8377 (3 classes) |
| Inference | ~1ms |
| Dataset | 17 813 tickets |

## Pipeline automatise

```
DAG 1 (donnees) -> DAG 2 (training) -> DAG 3 (deploy) -> API en production
                                                              |
DAG 5 (feedbacks) <- DAG 4 (drift Evidently) <--- prediction logs
```

Si le monitoring detecte un drift, le pipeline se relance en mode incremental : les feedbacks sont fusionnes avec le dataset existant, le modele est reentraine et redeploye si les seuils sont atteints.

## Classification des Tickets

### 6 Queues

| Queue | Tickets |
|-------|---------|
| Software / Product | 6 564 |
| General Technical Support | 4 649 |
| Security / Access | 4 160 |
| Network / Infrastructure | 1 196 |
| Hardware / Device | 783 |
| Service Outages | 461 |

### 3 Niveaux d'Urgence

high, medium, low

## Dataset

- Source : HuggingFace (Tobi-Bueck/customer-support-tickets)
- 17 813 tickets en anglais apres filtrage
- 6 queues (reclassification par keywords)
- 3 niveaux d'urgence

## Stack

| Composant | Technologie |
|-----------|------------|
| Classification | XGBoost |
| Embeddings | SentenceTransformer (all-MiniLM-L6-v2) |
| RAG | PGVector + Mistral API |
| Orchestration | Airflow (5 DAGs) |
| Tuning | Ray Tune + Optuna |
| Tracking | MLflow |
| Versioning | DVC + S3 |
| Monitoring | Evidently |
| API | FastAPI |
| CI/CD | GitHub Actions |
| Deploiement | K3s (Kubernetes) |
| Frontend | Streamlit |
| Base de donnees | PostgreSQL + pgvector |

## Les 5 DAGs Airflow

| DAG | Description |
|-----|-------------|
| dag1_data_pipeline | Ingestion, feature engineering, RAG |
| dag2_training_pipeline | Tuning, training XGBoost, validation, MLflow |
| dag3_deployment_pipeline | Push DVC, trigger CI/CD, health check |
| dag4_monitoring_pipeline | Detection de drift (Evidently, cron 6h UTC) |
| dag5_retraining_trigger | Charge les feedbacks, relance le pipeline |

## CI/CD (GitHub Actions)

```
test (pytest) -> build Docker + push ghcr.io -> deploy (kubectl rollout K3s)
```

Se declenche a chaque push sur main ou automatiquement apres un retraining.

## Deploiement (K3s)

```bash
# Secrets
kubectl apply -f k8s/secrets.yaml

# Bases de donnees
kubectl apply -f k8s/postgresql-data.yaml
kubectl apply -f k8s/postgresql-airflow.yaml

# Services
kubectl apply -f k8s/mlflow.yaml
kubectl apply -f k8s/airflow.yaml
kubectl apply -f k8s/support-agent.yaml
kubectl apply -f k8s/streamlit.yaml

# Verifier
kubectl get pods
```

## Endpoints API

| Endpoint | Methode | Description |
|----------|---------|-------------|
| /predict | POST | Classification + reponse RAG |
| /feedback | POST | Correction utilisateur |
| /health | GET | Health check |

```bash
curl -X POST http://<IP>:30080/predict \
  -H "Content-Type: application/json" \
  -d '{"user_query": "My VPN is not connecting"}'
```

## URLs

| Service | Port |
|---------|------|
| API | 30080 |
| Swagger | 30080/docs |
| Streamlit | 30085 |
| Airflow | 30082 |
| MLflow | 30050 |

## Structure

```
support-it-agent/
├── app.py                          # API FastAPI
├── Dockerfile                      # Image Docker
├── requirements.txt
├── streamlit_app.py                # Interface chat
├── dags/
│   ├── dag1_data_pipeline.py
│   ├── dag2_training_pipeline.py
│   ├── dag3_deployment_pipeline.py
│   ├── dag4_monitoring_pipeline.py
│   └── dag5_retraining_trigger.py
├── k8s/                            # Manifestes Kubernetes
│   ├── support-agent.yaml
│   ├── streamlit.yaml
│   ├── postgresql-data.yaml
│   ├── postgresql-airflow.yaml
│   ├── airflow.yaml
│   ├── mlflow.yaml
│   └── secrets.yaml
└── .github/workflows/
    └── ci-cd.yaml                  # Pipeline CI/CD
```
