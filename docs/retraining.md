# Retraining Automatisé - Pipeline MLOps

## Vue d'ensemble

```
┌─────────────────────────────────────────────────────────────────┐
│                    PIPELINE MLOPS COMPLET                       │
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │  Agent   │───>│ Feedback │───>│ Retrain  │───>│  Deploy  │  │
│  │  Répond  │    │  Humain  │    │   Auto   │    │   Auto   │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│       │                │               │               │        │
│       └────────────────┴───────────────┴───────────────┘        │
│                         BOUCLE CONTINUE                         │
└─────────────────────────────────────────────────────────────────┘
```

## Inference vs Training - Concept clé

### Définitions

| Concept | Définition | Le modèle change ? |
|---------|-----------|-------------------|
| **Training** | Le modèle apprend des patterns (ajuste ses poids) | OUI |
| **Inference** | Le modèle utilise ses patterns pour produire un résultat | NON |

### Application dans notre projet

Lors du retraining, deux étapes se succèdent :

```
Nouveaux textes (prediction_logs avec feedback)
        │
        ▼
┌─────────────────────────────────────────────┐
│  Sentence Transformers (INFERENCE)          │
│  - Convertit texte → vecteur 384 dim        │
│  - Modèle PRE-ENTRAINE (ne change pas)      │
│  - Prend ~30 min pour 17000 textes          │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│  XGBoost (TRAINING)                         │
│  - Apprend à classifier queue/urgence       │
│  - Modèle ENTRAINE par nous (change)        │
│  - Prend ~1 min                             │
└─────────────────────────────────────────────┘
        │
        ▼
Nouveau modèle déployé
```

### Pourquoi re-vectoriser à chaque retraining ?

Les nouvelles questions (via `prediction_logs`) arrivent en **texte brut**. Or XGBoost ne comprend pas le texte - il a besoin de **vecteurs numériques**.

Donc à chaque retraining :
1. On récupère le dataset enrichi (ancien + nouveaux feedbacks)
2. On re-vectorise TOUT avec Sentence Transformers (**inference**)
3. On entraîne XGBoost sur ces vecteurs (**training**)

**C'est l'inference qui prend le plus de temps** (~30 min) car on traite 17000+ textes. Mais le modèle Sentence Transformers ne change jamais - il "travaille" simplement.

## Étape 1: Collecte des feedbacks

### Sources de feedback

| Source | Type | Valeur |
|--------|------|--------|
| Bouton 👍👎 | Explicite | +1 / -1 |
| Ticket réouvert | Implicite | -1 (pas résolu) |
| Correction agent | Gold | Nouvelle donnée |
| Temps résolution | Implicite | Court = bon |

### Interface utilisateur

```
┌─────────────────────────────────────────────┐
│  Agent: "Voici comment résoudre votre       │
│          problème de VPN..."                │
│                                             │
│  Cette réponse vous a-t-elle aidé?          │
│                                             │
│      [ 👍 Oui ]    [ 👎 Non ]               │
│                                             │
└─────────────────────────────────────────────┘
```

### Stockage des feedbacks

```python
# Schema base de données
feedback_table = {
    "id": "uuid",
    "timestamp": "datetime",
    "question": "text",
    "response": "text",
    "category": "string",
    "feedback_score": "int",  # -1, 0, +1
    "agent_correction": "text",  # Si corrigé
    "used_for_training": "bool"  # Déjà utilisé?
}
```

## Étape 2: Triggers de retraining

### Conditions de déclenchement

```python
def should_retrain():
    # Condition 1: Assez de nouvelles données
    new_feedbacks = db.count(used_for_training=False)
    if new_feedbacks >= 1000:
        return True, "1000+ nouveaux feedbacks"

    # Condition 2: Performance dégradée
    recent_satisfaction = get_satisfaction_last_7_days()
    if recent_satisfaction < 0.80:
        return True, "Satisfaction < 80%"

    # Condition 3: Scheduled (hebdomadaire)
    if is_sunday_night():
        return True, "Retraining hebdomadaire"

    return False, "Pas besoin"
```

### Orchestration avec Airflow

```python
# dag_retrain.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'mlops',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'retrain_support_model',
    default_args=default_args,
    schedule_interval='0 2 * * 0',  # Dimanche 2h du matin
    catchup=False,
) as dag:

    check_trigger = PythonOperator(
        task_id='check_retrain_needed',
        python_callable=should_retrain
    )

    prepare_data = PythonOperator(
        task_id='prepare_training_data',
        python_callable=prepare_data_for_training
    )

    train_model = PythonOperator(
        task_id='train_model_rlhf',
        python_callable=run_rlhf_training
    )

    evaluate = PythonOperator(
        task_id='evaluate_new_model',
        python_callable=evaluate_model
    )

    deploy = PythonOperator(
        task_id='deploy_if_better',
        python_callable=deploy_new_model
    )

    check_trigger >> prepare_data >> train_model >> evaluate >> deploy
```

## Étape 3: Préparation des données

```python
def prepare_data_for_training():
    # Récupérer les nouveaux feedbacks
    new_data = db.query("""
        SELECT question, response, feedback_score, agent_correction
        FROM feedbacks
        WHERE used_for_training = False
        AND feedback_score != 0
    """)

    # Créer le dataset pour RLHF
    training_data = []

    for row in new_data:
        if row['agent_correction']:
            # Si corrigé par agent = gold standard
            training_data.append({
                'prompt': row['question'],
                'chosen': row['agent_correction'],  # Bonne réponse
                'rejected': row['response']  # Mauvaise réponse
            })
        else:
            # Sinon utiliser le score
            training_data.append({
                'prompt': row['question'],
                'response': row['response'],
                'reward': row['feedback_score']
            })

    # Sauvegarder
    save_dataset(training_data, 'new_training_data.json')

    # Marquer comme utilisé
    db.update("UPDATE feedbacks SET used_for_training = True WHERE ...")

    return len(training_data)
```

## Étape 4: Retraining RLHF

```python
def run_rlhf_training():
    from trl import PPOTrainer, PPOConfig

    # Charger le modèle actuel
    model = load_current_model()
    ref_model = load_reference_model()

    # Charger les nouvelles données
    dataset = load_dataset('new_training_data.json')

    # Config PPO
    config = PPOConfig(
        learning_rate=1e-5,
        batch_size=8,
        mini_batch_size=8,
    )

    # Trainer
    trainer = PPOTrainer(
        config=config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
    )

    # Training loop
    for batch in trainer.dataloader:
        # Générer réponses
        responses = trainer.generate(batch['input_ids'])

        # Calculer rewards (depuis le reward model)
        rewards = reward_model(batch['prompt'], responses)

        # Step PPO
        trainer.step(batch['input_ids'], responses, rewards)

    # Sauvegarder
    model.save_pretrained('models/new_model')

    # Log dans MLflow
    mlflow.log_artifact('models/new_model')

    return 'models/new_model'
```

## Étape 5: Évaluation

```python
def evaluate_model():
    # Charger les modèles
    new_model = load_model('models/new_model')
    old_model = load_model('models/current_model')

    # Dataset de test
    test_data = load_test_dataset()

    # Évaluer les deux
    new_scores = []
    old_scores = []

    for sample in test_data:
        # Générer réponses
        new_response = new_model.generate(sample['prompt'])
        old_response = old_model.generate(sample['prompt'])

        # Scorer avec reward model
        new_scores.append(reward_model(sample['prompt'], new_response))
        old_scores.append(reward_model(sample['prompt'], old_response))

    # Comparer
    new_avg = sum(new_scores) / len(new_scores)
    old_avg = sum(old_scores) / len(old_scores)

    # Log
    mlflow.log_metrics({
        'new_model_score': new_avg,
        'old_model_score': old_avg,
        'improvement': new_avg - old_avg
    })

    return {
        'new_score': new_avg,
        'old_score': old_avg,
        'should_deploy': new_avg > old_avg
    }
```

## Étape 6: Déploiement automatique

```python
def deploy_new_model():
    eval_results = get_evaluation_results()

    if not eval_results['should_deploy']:
        log("Nouveau modèle pas meilleur, on garde l'ancien")
        return False

    # Backup ancien modèle
    backup_model('models/current_model', 'models/backup/')

    # Remplacer
    copy_model('models/new_model', 'models/current_model')

    # Recharger l'API
    restart_api_server()

    # Notifier
    send_notification(
        f"Nouveau modèle déployé! "
        f"Score: {eval_results['new_score']:.2f} "
        f"(+{eval_results['improvement']:.2f})"
    )

    return True
```

## Outils recommandés

| Étape | Outil |
|-------|-------|
| Orchestration | Airflow / Prefect |
| Stockage données | PostgreSQL / MongoDB |
| Training | TRL + HuggingFace |
| Tracking expériences | MLflow / W&B |
| Déploiement | Docker + Kubernetes |
| CI/CD | GitHub Actions |
| Monitoring | Prometheus + Grafana |

## Exemple de workflow GitHub Actions

```yaml
# .github/workflows/retrain.yml
name: Retrain Model

on:
  schedule:
    - cron: '0 2 * * 0'  # Dimanche 2h
  workflow_dispatch:  # Manuel aussi

jobs:
  retrain:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Check if retrain needed
        run: python scripts/check_retrain.py

      - name: Prepare data
        run: python scripts/prepare_data.py

      - name: Train model
        run: python scripts/train_rlhf.py

      - name: Evaluate
        run: python scripts/evaluate.py

      - name: Deploy if better
        run: python scripts/deploy.py
```

## Résumé

```
FEEDBACK → TRIGGER → PREPARE → TRAIN → EVALUATE → DEPLOY
    ↑                                                │
    └────────────────────────────────────────────────┘
                    BOUCLE CONTINUE
```

**C'est ça le MLOps moderne : amélioration continue automatisée !**
