# Projet MLOps - Agent Support Technique

## Objectif

Construire un chatbot de support technique automatisé avec:
- Classification des demandes (Random Forest)
- Niveau d'urgence
- Génération de réponses (LLM fine-tuné + RLHF)
- Retraining automatisé

## Données

### Option Bootcamp (données publiques)
```python
from datasets import load_dataset
dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
```

**Contient:**
- 26,000 exemples
- 27 catégories
- Questions + Réponses
- Intentions labellisées

### Option Production (données confidentielles)
- 4 ans d'emails collaborateurs
- Réponses des agents humains
- Documentation technique

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    PIPELINE MLOPS COMPLET                       │
│                                                                 │
│  1. CLASSIFICATION ──> 2. RAG ──> 3. LLM ──> 4. RLHF            │
│         │                │          │          │                │
│    Random Forest    Recherche   Fine-tuné   Optimisé            │
│    Catégorie +      dans doc    Mistral/    avec                │
│    Urgence                      Llama       feedback            │
│                                                                 │
│  5. DEPLOY ──> 6. MONITORING ──> 7. RETRAIN AUTO                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Structure du projet

```
projet_mlops_support/
├── README.md                    # Ce fichier
├── 01_RLHF_explications.md      # Comprendre RLHF
├── 02_architecture_projet.md    # Architecture détaillée
├── 03_retraining_auto.md        # Pipeline de retraining
├── 04_pourquoi_finetuning.md    # Avantages vs API
└── code/                        # Code du projet (à venir)
```

## Ressources

- Notebook RLHF: `../RL/RL_DL/08-Code-RLHF_gpt2-sentiment.ipynb`
- Dataset: [Bitext Customer Support](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
- TRL Documentation: https://huggingface.co/docs/trl


