# Résultats du filtre - Dataset Support Technique

## Vue d'ensemble

| Métrique | Valeur |
|----------|--------|
| **Tickets après filtre** | 17 893 |
| **Langue** | Anglais (EN) |
| **Queues** | Techniques sélectionnées |

> Volume excellent : suffisant pour classification robuste, RAG efficace et SFT/LoRA sans overfitting sur un modèle 7B/8B.

---

## Distribution par Queue (Sujets)

| Queue | Nombre | Pourcentage |
|-------|--------|-------------|
| Technical Support | 8 149 | 45.5% |
| Product Support | 5 305 | 29.7% |
| IT Support | 3 333 | 18.6% |
| Service Outages & Maintenance | 1 106 | 6.2% |
| **Total** | **17 893** | **100%** |

**Constat** : 4 classes principales, très bien équilibrées pour du support technique interne.

---

## Distribution par Urgence

| Niveau | Nombre | Pourcentage |
|--------|--------|-------------|
| High | 8 693 | 48.6% |
| Medium | 6 767 | 37.8% |
| Low | 2 433 | 13.6% |
| **Total** | **17 893** | **100%** |

**Constat** : Distribution légèrement biaisée vers high/medium (typique des datasets support). Gérable avec `stratified split` et `class_weight` dans XGBoost.

### Mapping des urgences

| Niveau final | Priorités originales |
|--------------|---------------------|
| High | critical + high |
| Medium | medium |
| Low | low + very_low |

---

## Proposition de classes finales

### Option 1 : 4 classes (Recommandée)

| Classe | Queue originale | Nb tickets | Commentaire |
|--------|-----------------|------------|-------------|
| Technical Support | Technical Support | ~8 149 | Majoritaire, coeur IT |
| Product Support | Product Support | ~5 305 | Logiciels, bugs, features |
| IT Support | IT Support | ~3 333 | Hardware, réseau, infra |
| Service Outages | Service Outages and Maintenance | ~1 106 | Pannes, maintenance |

**Avantages** :
- Simple et naturel
- Facile à expliquer en présentation
- Bonne performance ML

### Option 2 : 5-6 classes (Extension)

Ajout possible via extraction de keywords dans le body :

| Classe additionnelle | Détection | Nb estimé |
|---------------------|-----------|-----------|
| Security | keywords: security, access, permission | ~500-1k |
| Network | keywords: network, connectivity, VPN | variable |
| Hardware | keywords: hardware, device, printer | variable |

### Option 3 : 8-12 classes (Granularité maximale)

Sous-classes extraites via NLP sur le body des tickets :
- Network Issues
- Hardware Problems
- Access & Security
- Software Bugs
- Feature Requests
- System Outages
- Scheduled Maintenance
- Account Issues

---

## Recommandations pour le Bootcamp

| Aspect | Recommandation |
|--------|----------------|
| **Classification** | Commencer avec 4 classes |
| **Urgence** | 3 niveaux (high/medium/low) |
| **Split données** | Stratified split pour éviter leakage |
| **Modèle ML** | XGBoost avec `class_weight` pour le déséquilibre |
| **LLM** | Volume suffisant pour SFT/LoRA sur 7B/8B |

---

## Prochaines étapes

1. [ ] Split train/val/test stratifié
2. [ ] Entraînement classifieur (XGBoost / RandomForest)
3. [ ] Mise en place RAG
4. [ ] Fine-tuning LLM (optionnel)
5. [ ] Évaluation et métriques
