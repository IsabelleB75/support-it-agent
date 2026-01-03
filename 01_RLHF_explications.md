# RLHF - Reinforcement Learning from Human Feedback

## C'est quoi RLHF ?

RLHF = Reinforcement Learning from **Human** Feedback

C'est la technique qui rend ChatGPT, Claude, etc. utiles et alignés avec les préférences humaines.

## Le pipeline RLHF

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Prompt    │ --> │    LLM      │ --> │   Reward    │
│  (input)    │     │  génère     │     │   Model     │
│             │     │  réponse    │     │  (évalue)   │
└─────────────┘     └─────────────┘     └─────────────┘
                                              │
                                              ▼
                                         Score: +2.5
                                              │
                                              ▼
                                     ┌─────────────┐
                                     │     PPO     │
                                     │  optimise   │
                                     │    LLM      │
                                     └─────────────┘
```

## Les 3 composants clés

### 1. Le modèle de base (Policy)
- LLM qui génère du texte (GPT-2, Mistral, Llama, etc.)
- C'est lui qu'on entraîne

### 2. Le Reward Model (le juge)
- Évalue si la réponse est "bonne" ou "mauvaise"
- Entraîné sur des préférences humaines
- Donne un score (ex: -3 à +3)

### 3. PPO (l'optimiseur)
- Proximal Policy Optimization
- Ajuste le LLM pour maximiser les récompenses
- Garde le modèle "raisonnable" avec KL divergence

## Le "H" = Le coeur du système

```
AVANT: Des humains annotent des données
       "Cette réponse est bonne" ✅
       "Cette réponse est mauvaise" ❌

PENDANT: Le Reward Model apprend ces préférences

APRÈS: Le Reward Model peut évaluer automatiquement
       de nouvelles réponses
```

**Sans feedback humain, pas de direction pour l'optimisation !**

## Exemple concret : Sentiment (notebook)

| Composant | Dans le notebook |
|-----------|------------------|
| Policy | GPT-2 |
| Reward Model | BERT sentiment (distilbert-imdb) |
| Données humaines | 50,000 reviews IMDB annotées +/- |
| PPO | TRL PPOTrainer |

## Exemple concret : Support technique (ton projet)

| Composant | Dans ton projet |
|-----------|-----------------|
| Policy | Mistral 7B / Llama |
| Reward Model | Modèle entraîné sur qualité des réponses |
| Données humaines | Feedbacks 👍👎 + corrections agents |
| PPO | TRL PPOTrainer |

## KL Divergence - Garder le modèle raisonnable

### Problème sans KL
```
GPT-2 pourrait tricher:
"good good good amazing wonderful perfect"
→ Score très positif mais texte stupide
```

### Solution avec KL
```
récompense_finale = score_reward - β × KL_divergence

KL_divergence = différence avec le modèle original
β = coefficient de pénalité
```

Le modèle reste proche de sa version originale tout en s'améliorant.

## Pourquoi RLHF marche si bien ?

| Méthode | Limitation |
|---------|------------|
| Fine-tuning supervisé | Imite, mais peut imiter les erreurs |
| RLHF | Optimise pour le RÉSULTAT voulu |

```
Supervisé: "Réponds comme les données"
RLHF: "Réponds pour maximiser la satisfaction"
```

## Applications réelles

| Application | Le "H" (feedback humain) |
|-------------|-------------------------|
| ChatGPT | "Réponse utile et polie" |
| Claude | "Réponse honnête et safe" |
| Copilot | "Code correct et sécurisé" |
| Ton projet | "Réponse qui résout le problème" |
