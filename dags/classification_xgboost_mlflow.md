# DAG : classification_tickets_xgboost_mlflow

## Description
Ce DAG entraîne deux modèles XGBoost pour classifier les tickets de support :
1. **refined_queue** (6 classes) : catégorie du ticket
2. **urgency_level** (3 classes) : niveau d'urgence

Tous les résultats sont loggés dans **MLflow**.

---

## Informations générales

| Paramètre | Valeur |
|-----------|--------|
| **DAG ID** | `classification_tickets_xgboost_mlflow` |
| **Owner** | `jedha_bootcamp` |
| **Schedule** | Manuel (`None`) |
| **Tags** | `classification`, `xgboost`, `mlflow`, `tickets` |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                          DAG Airflow                                  │
│              classification_tickets_xgboost_mlflow                    │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │              Task: train_xgboost_and_log_mlflow                  │ │
│  │                                                                  │ │
│  │  1. Lit tickets_tech_en_enriched depuis Postgres                │ │
│  │  2. Combine text (subject + body_clean) + features numériques   │ │
│  │  3. Split stratifié (Train 72% / Val 13% / Test 15%)            │ │
│  │  4. TF-IDF vectorization (5000 features, 1-2 ngrams)            │ │
│  │  5. Entraîne XGBoost pour refined_queue (6 classes)             │ │
│  │  6. Entraîne XGBoost pour urgency_level (3 classes)             │ │
│  │  7. Log metrics, models, artifacts dans MLflow                   │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Features utilisées

### Features textuelles (TF-IDF)
- `subject` + `body_clean` combinés
- TF-IDF avec 5000 features max
- N-grams (1,2)

### Features numériques
| Feature | Description |
|---------|-------------|
| `body_length` | Longueur du corps du ticket |
| `answer_length` | Longueur de la réponse |
| `response_ratio` | Ratio réponse/problème |
| `has_network` | Contient mots-clés réseau |
| `has_printer` | Contient mots-clés imprimante |
| `has_security` | Contient mots-clés sécurité |
| `has_hardware` | Contient mots-clés hardware |
| `has_software` | Contient mots-clés software |

---

## Targets

### refined_queue (6 classes)
| Classe | % |
|--------|---|
| Software / Product | 36.9% |
| General Technical Support | 26.1% |
| Security / Access | 23.3% |
| Network / Infrastructure | 6.7% |
| Hardware / Device | 4.4% |
| Service Outages | 2.6% |

### urgency_level (3 classes)
| Niveau | % |
|--------|---|
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

## Artifacts sauvegardés (tout dans MLflow)

| Artifact | Description |
|----------|-------------|
| `xgboost_queue` | Modèle XGBoost pour refined_queue |
| `xgboost_urgency` | Modèle XGBoost pour urgency_level |
| `tfidf_vectorizer.pkl` | Vectorizer TF-IDF (5000 features) |
| `le_queue.pkl` | LabelEncoder pour queue (6 classes) |
| `le_urgency.pkl` | LabelEncoder pour urgency (3 classes) |
| `classification_report_queue.txt` | Rapport de classification queue |
| `classification_report_urgency.txt` | Rapport de classification urgency |

**Avantages de tout centraliser dans MLflow :**
- Versioning complet : chaque run garde son modèle + vectorizer + encoders
- Reproductibilité : on peut revenir à n'importe quelle version
- Déploiement facile : un seul Run ID pour tout récupérer

---

## MLflow

- **Tracking URI** : `file:///opt/airflow/mlruns` (stockage local, évite problèmes de permission)
- **Experiment** : `tickets_classification_bootcamp`
- **Dossier local** : `./mlruns` (monté dans le container)

### Metrics loggées
- `accuracy_queue` : Accuracy sur refined_queue
- `f1_queue` : F1-score pondéré sur refined_queue
- `accuracy_urgency` : Accuracy sur urgency_level
- `f1_urgency` : F1-score pondéré sur urgency_level

### Visualiser les résultats
```bash
# Lancer l'UI MLflow pour voir les runs
mlflow ui --backend-store-uri file://./mlruns --host 0.0.0.0 --port 5000
```

---

## Exécution

### Prérequis
1. MLflow doit être lancé sur le port 5000
2. La table `tickets_tech_en_enriched` doit exister

### Déclencher
1. Aller dans l'UI Airflow : `http://<IP>:8082`
2. Activer le DAG
3. Cliquer sur "Trigger DAG"

### Voir les résultats
- **Airflow Logs** : Accuracy et F1 affichés
- **MLflow UI** : `http://<IP>:5000` → Experiment `tickets_classification_bootcamp`

---

## Résultats attendus

| Modèle | Accuracy attendue | F1 attendu |
|--------|-------------------|------------|
| refined_queue | ~70-80% | ~0.70-0.80 |
| urgency_level | ~55-65% | ~0.55-0.65 |
