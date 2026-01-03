# DAG : ingest_tickets_tech_en

## Description
Ce DAG ingère les tickets de support technique filtrés (anglais uniquement) depuis un fichier Parquet vers PostgreSQL.

---

## Informations générales

| Paramètre | Valeur |
|-----------|--------|
| **DAG ID** | `ingest_tickets_tech_en` |
| **Owner** | `jedha_bootcamp` |
| **Schedule** | Manuel (`None`) |
| **Retries** | 1 |
| **Tags** | `ingestion`, `postgres`, `tickets` |

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  DAG Airflow                     │
│            ingest_tickets_tech_en                │
│                                                  │
│  ┌──────────────────────────────────────────┐   │
│  │     Task: ingest_tickets_to_postgres      │   │
│  │                                           │   │
│  │  1. Lit train_tech_en.parquet            │   │
│  │  2. Connecte à Postgres (port 5433)      │   │
│  │  3. Insère 17893 lignes                  │   │
│  └──────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

---

## Explication du code

### 1. Les imports

```python
from datetime import datetime          # Pour les dates
from airflow import DAG                 # Le conteneur principal Airflow
from airflow.operators.python import PythonOperator  # Pour exécuter du Python
import pandas as pd                     # Manipulation de données
from sqlalchemy import create_engine    # Connexion à Postgres
import logging                          # Pour afficher des logs
```

### 2. Configuration par défaut

```python
default_args = {
    'owner': 'jedha_bootcamp',    # Qui est responsable du DAG
    'depends_on_past': False,     # Ne dépend pas des exécutions précédentes
    'retries': 1,                 # Réessaie 1 fois si échec
}
```

| Paramètre | Description |
|-----------|-------------|
| `owner` | Propriétaire du DAG, visible dans l'UI Airflow |
| `depends_on_past` | Si `True`, attend que l'exécution précédente réussisse |
| `retries` | Nombre de tentatives en cas d'échec |

### 3. Définition du DAG

```python
dag = DAG(
    'ingest_tickets_tech_en',     # Nom unique du DAG
    default_args=default_args,
    description='Ingest filtered technical support tickets into Postgres',
    schedule_interval=None,       # Pas de schedule automatique (manuel)
    start_date=datetime(2025, 12, 1),  # Date de départ
    catchup=False,                # Ne rattrape pas les exécutions manquées
    tags=['ingestion', 'postgres', 'tickets'],  # Tags pour filtrer dans l'UI
)
```

| Paramètre | Description |
|-----------|-------------|
| `schedule_interval=None` | Déclenchement manuel uniquement |
| `catchup=False` | N'exécute pas les runs manqués entre `start_date` et aujourd'hui |
| `tags` | Permet de filtrer les DAGs dans l'interface Airflow |

### 4. La fonction d'ingestion

```python
def ingest_to_postgres():
    # Chemin du fichier parquet DANS le container Docker
    parquet_path = '/opt/airflow/data/train_tech_en.parquet'

    # Connexion à ta DB Postgres
    # host.docker.internal = accès à localhost depuis Docker
    engine = create_engine('postgresql://bootcamp_user:bootcamp_password@host.docker.internal:5433/support_tech')

    logging.info("Lecture du parquet...")
    df = pd.read_parquet(parquet_path)

    logging.info(f"Ingestion de {len(df)} lignes...")
    df.to_sql(
        name='tickets_tech_en',   # Nom de la table
        con=engine,               # Connexion
        if_exists='replace',      # Remplace la table si existe
        index=False,              # N'ajoute pas l'index pandas
        method='multi',           # Insert plusieurs lignes à la fois (rapide)
        chunksize=1000            # Par lots de 1000 lignes
    )
    logging.info("Ingestion terminée !")
```

| Paramètre `to_sql()` | Description |
|----------------------|-------------|
| `name` | Nom de la table cible dans Postgres |
| `con` | Connexion SQLAlchemy |
| `if_exists='replace'` | Remplace la table si elle existe (ou `'append'` pour ajouter) |
| `index=False` | N'insère pas l'index pandas comme colonne |
| `method='multi'` | Insère plusieurs lignes par requête (plus rapide) |
| `chunksize=1000` | Traite par lots de 1000 lignes (évite surcharge mémoire) |

### 5. Déclaration de la tâche

```python
ingest_task = PythonOperator(
    task_id='ingest_tickets_to_postgres',  # Nom de la tâche
    python_callable=ingest_to_postgres,    # Fonction à exécuter
    dag=dag,                               # Rattachée à quel DAG
)
```

---

## Connexions

### Source
- **Fichier** : `/opt/airflow/data/train_tech_en.parquet`
- **Volume Docker** : `./data:/opt/airflow/data`

### Destination
- **Host** : `host.docker.internal` (localhost depuis Docker)
- **Port** : `5433`
- **Database** : `support_tech`
- **Table** : `tickets_tech_en`
- **User** : `bootcamp_user`

---

## Exécution

### Déclencher manuellement
1. Aller dans l'UI Airflow : `http://<IP>:8082`
2. Activer le DAG (toggle ON)
3. Cliquer sur "Trigger DAG"

### Vérifier le résultat
```sql
SELECT COUNT(*) FROM tickets_tech_en;  -- Doit retourner 17893
```

---

## Résultat attendu

| Métrique | Valeur |
|----------|--------|
| Lignes ingérées | 17 893 |
| Temps d'exécution | ~5 secondes |
| Status | SUCCESS |
