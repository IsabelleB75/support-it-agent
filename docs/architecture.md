# Architecture du Projet Support IT Agent

## Pipeline 100% Automatise

Ce projet implemente un pipeline MLOps **entierement automatise** :

```
Drift detecte → Retraining → Validation → DVC push → CI/CD → Deploy K3s
```

Aucune intervention manuelle requise entre la detection de drift et le deploiement du nouveau modele.

## Vue d'ensemble

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER REQUEST                            │
│                    "Mon VPN ne marche pas"                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API FastAPI (K3s)                          │
│                    http://78.47.129.250:30080                   │
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
              │               │               │
              └───────────────┼───────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         RESPONSE                                │
│  - predicted_queue: "Network / Infrastructure"                  │
│  - predicted_urgency: "high"                                    │
│  - response: "Voici les etapes pour resoudre..."               │
└─────────────────────────────────────────────────────────────────┘
```

## Composant 1: Classification (XGBoost)

### Objectif
- Classifier la demande dans une queue (Network, Hardware, Software, etc.)
- Determiner le niveau d'urgence (low, medium, high)

### Implementation
- **Modele** : XGBoost (multi-class softprob)
- **Features** : Embeddings (Sentence Transformers) + features numeriques
- **Tracking** : MLflow

```python
from xgboost import XGBClassifier
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedder.encode(texts)

model_queue = XGBClassifier(objective='multi:softprob')
model_queue.fit(X_train, y_train)
```

## Composant 2: RAG (Retrieval Augmented Generation)

### Objectif
- Rechercher les documents pertinents dans la base de connaissances
- Fournir du contexte au LLM pour generer une reponse pertinente

### Implementation
- **Embeddings** : Sentence Transformers (all-MiniLM-L6-v2)
- **Vector DB** : PostgreSQL + PGVector
- **Recherche** : Similarite cosinus

```python
# Recherche semantique
query_emb = embedder.encode(query_text)
results = db.query(
    "SELECT content FROM rag_docs ORDER BY embedding <=> %s LIMIT 5",
    [query_emb]
)
```

## Composant 3: Generation de reponse (Mistral API)

### Implementation
- **API** : Mistral AI (open-mistral-7b)
- **Prompt** : Question + Contexte RAG + Classification

```python
messages = [
    {"role": "system", "content": "Tu es un agent de support IT..."},
    {"role": "user", "content": f"Categorie:{queue}\nContexte:{rag_docs}\nQuestion:{query}"}
]
response = requests.post(MISTRAL_API_URL, json={"messages": messages})
```

## Pipeline MLOps Automatise

### Cycle complet

```
   ┌───────────────────────┐       ┌───────────────────────┐       ┌───────────────────────┐
   │   1. DATA PIPELINE    │       │  2. TRAINING PIPELINE │       │    3. DEPLOYMENT      │
   ├───────────────────────┤       ├───────────────────────┤       ├───────────────────────┤
   │ PostgreSQL            │       │ Airflow DAG training  │       │ Docker: build image   │
   │ - tickets_tech_en     │features│ - XGBoost (queue +   │modele │ - ghcr.io registry    │
   │ - prediction_logs     │──────>│   urgence)            │──────>│                       │
   │ - embeddings (PGVector)│      │ - Ray Tune (hyperparam)│       │ GitHub Actions: CI/CD │
   │                       │       │                       │       │                       │
   │ DVC: versioning data  │       │ MLflow:               │       │ K3s: Deployment       │
   │ - S3 storage          │       │ - tracking experiments│       │ - 2 pods              │
   └───────────────────────┘       └───────────────────────┘       └───────────────────────┘
   ▲                                                                          │
   │                                                                          │
   │                                                                          ▼
   │ ┌───────────────────────┐       ┌───────────────────────────────────────────────────────┐
   │ │    5. MONITORING      │       │              4. SERVING (temps reel)                  │
   │ ├───────────────────────┤       ├───────────────────────────────────────────────────────┤
   │ │ Airflow DAG:          │       │  FastAPI (http://78.47.129.250:30080)                 │
   │ │ - monitoring_evidently│       │  - XGBoost: prediction queue + urgence               │
   │ │                       │       │  - Sentence Transformers: RAG                        │
   │ │ prediction_logs:      │ logs  │  - Mistral API: generation reponse                   │
   │ │ - predictions         │<──────│                                                       │
   │ │ - feedback            │       │  Feedback utilisateur → prediction_logs              │
   │ │                       │       └───────────────────────────────────────────────────────┘
   │ │ Evidently: detection  │
   │ │ drift                 │
   │ │  ┌─────┐  ┌──────┐    │
   │ │  │ OK  │  │DRIFT │    │
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

### Automatisation complete

| Etape | Declencheur | Action |
|-------|-------------|--------|
| 1. Monitoring | Schedule (6h/jour) | Evidently detecte le drift |
| 2. Hyperparameters | Drift > 30% | Ray Tune optimise les params |
| 3. Retraining | Apres hyperparams | XGBoost entraine avec nouveaux params |
| 4. Validation | Apres training | Verifie F1 > seuils minimaux |
| 5. DVC Push | Si validation OK | Pousse modele vers S3 |
| 6. CI/CD | Apres DVC push | GitHub Actions declenche via API |
| 7. Deploy | CI/CD success | kubectl apply sur K3s |

**Zero intervention manuelle** entre la detection de drift et le deploiement.

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

## URLs des Services

| Service | URL |
|---------|-----|
| API Support Agent | http://78.47.129.250:30080 |
| API Docs (Swagger) | http://78.47.129.250:30080/docs |
| MLflow | http://78.47.129.250:5000 |
| Airflow | http://78.47.129.250:8082 |
| Evidently Reports | http://78.47.129.250:8083 |
