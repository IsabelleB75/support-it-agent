# Architecture du Projet Support Technique

## Vue d'ensemble

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER REQUEST                            │
│                    "Mon VPN ne marche pas"                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    1. CLASSIFICATION                            │
│                      Random Forest                              │
│                                                                 │
│  Input:  "Mon VPN ne marche pas"                                │
│  Output: Catégorie = "NETWORK"                                  │
│          Urgence = "HIGH"                                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       2. RAG                                    │
│              Retrieval Augmented Generation                     │
│                                                                 │
│  Recherche dans la documentation:                               │
│  → "Guide VPN entreprise.pdf"                                   │
│  → "FAQ problèmes réseau.md"                                    │
│  → Tickets similaires résolus                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    3. LLM FINE-TUNÉ                             │
│                   Mistral 7B / Llama                            │
│                                                                 │
│  Prompt:                                                        │
│  - Question: "Mon VPN ne marche pas"                            │
│  - Contexte: [docs RAG]                                         │
│  - Catégorie: NETWORK (HIGH)                                    │
│                                                                 │
│  Réponse générée:                                               │
│  "Bonjour, je comprends votre problème de VPN.                  │
│   Voici les étapes à suivre:                                    │
│   1. Vérifiez votre connexion internet                          │
│   2. Redémarrez le client VPN                                   │
│   3. ..."                                                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         RÉPONSE                                 │
│                    Envoyée au client                            │
└─────────────────────────────────────────────────────────────────┘
```

## Composant 1: Classification (Random Forest)

### Objectif
- Catégoriser la demande
- Déterminer le niveau d'urgence
- Router vers le bon traitement

### Catégories possibles
| Catégorie | Exemples |
|-----------|----------|
| ACCOUNT | Login, mot de passe, permissions |
| NETWORK | VPN, wifi, connexion |
| HARDWARE | Imprimante, écran, clavier |
| SOFTWARE | Installation, bugs, licences |
| BILLING | Factures, abonnements |

### Niveaux d'urgence
| Niveau | Critère |
|--------|---------|
| LOW | Pas bloquant |
| MEDIUM | Gênant mais contournable |
| HIGH | Bloquant pour le travail |
| CRITICAL | Impact business majeur |

### Code exemple
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# Vectorisation
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(texts)

# Classification catégorie
clf_category = RandomForestClassifier(n_estimators=100)
clf_category.fit(X_train, y_category_train)

# Classification urgence
clf_urgency = RandomForestClassifier(n_estimators=100)
clf_urgency.fit(X_train, y_urgency_train)
```

## Composant 2: RAG (Retrieval)

### Objectif
- Trouver les informations pertinentes
- Enrichir le contexte pour le LLM

### Sources
1. Documentation technique
2. FAQ
3. Tickets précédents résolus
4. Procédures internes

### Architecture RAG
```
Question ──> Embedding ──> Recherche vectorielle ──> Top 5 docs
                                                        │
                                                        ▼
                                              Contexte pour LLM
```

### Outils
- Embeddings: `sentence-transformers`
- Vector DB: ChromaDB, Pinecone, Weaviate
- Framework: LangChain, LlamaIndex

## Composant 3: LLM Fine-tuné + RLHF

### Modèle de base
- Mistral 7B (recommandé)
- Llama 2/3
- Phi-2 (plus petit)

### Fine-tuning supervisé (SFT)
```
Input:  Question client + Contexte RAG
Output: Réponse de l'agent humain (gold standard)
```

### RLHF
```
Reward Model évalue:
- Réponse résout le problème? (+3)
- Réponse polie et professionnelle? (+1)
- Réponse hors sujet? (-2)
- Réponse incorrecte/dangereuse? (-3)
```

### Pourquoi les deux?
| Étape | Ce qu'elle apporte |
|-------|-------------------|
| SFT | Le modèle apprend le FORMAT des réponses |
| RLHF | Le modèle apprend la QUALITÉ des réponses |

## Stack technique recommandé

| Composant | Outil |
|-----------|-------|
| Classification | Scikit-learn |
| Embeddings | Sentence-transformers |
| Vector DB | ChromaDB |
| LLM | Mistral 7B (via HuggingFace) |
| Fine-tuning | TRL + PEFT (LoRA) |
| RLHF | TRL PPOTrainer |
| API | FastAPI |
| Orchestration | Airflow / Prefect |
| Monitoring | MLflow |
| Deploy | Docker + Kubernetes |

## Dataset recommandé (bootcamp)

```python
from datasets import load_dataset

dataset = load_dataset(
    "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
)

# Structure
# {
#   "instruction": "I need help with my password",
#   "intent": "password_reset",
#   "category": "ACCOUNT",
#   "response": "I can help you reset..."
# }
```

## Métriques à suivre

| Métrique | Description |
|----------|-------------|
| Accuracy classification | % bonnes catégories |
| Response quality | Score du reward model |
| Resolution rate | % problèmes résolus |
| User satisfaction | Feedbacks 👍👎 |
| Latency | Temps de réponse |
