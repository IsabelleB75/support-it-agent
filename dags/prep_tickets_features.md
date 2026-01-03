# DAG : prep_tickets_features

## Description
Ce DAG effectue le nettoyage et le feature engineering sur les tickets de support technique pour préparer les données pour le ML.

---

## Informations générales

| Paramètre | Valeur |
|-----------|--------|
| **DAG ID** | `prep_tickets_features` |
| **Owner** | `jedha_bootcamp` |
| **Schedule** | Manuel (`None`) |
| **Retries** | 1 |
| **Tags** | `preparation`, `features`, `postgres` |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DAG Airflow                               │
│                 prep_tickets_features                        │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              Task: prepare_features                     │ │
│  │                                                         │ │
│  │  1. Lit tickets_tech_en depuis Postgres                │ │
│  │  2. Nettoie body et answer (HTML, espaces, tel)        │ │
│  │  3. Calcule features (longueurs, ratios)               │ │
│  │  4. Détecte keywords (network, printer, security...)   │ │
│  │  5. Sauvegarde dans tickets_tech_en_enriched           │ │
│  │  6. Génère rapports Evidently (Quality + Drift)        │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## Explication du code

### 1. Connexion et lecture

```python
engine = create_engine('postgresql://bootcamp_user:bootcamp_password@host.docker.internal:5433/support_tech')

query = "SELECT * FROM tickets_tech_en"
df = pd.read_sql(query, engine)
```

Lit les 17 893 tickets depuis la table source.

### 2. Nettoyage du texte

```python
def clean_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r'<.*?>', '', text)               # supprime HTML
    text = re.sub(r'\s+', ' ', text)                # espaces multiples
    text = text.strip()
    text = re.sub(r'\d{3}-\d{3}-\d{4}', '<tel>', text)  # anonymise tel US
    return text

df['body_clean'] = df['body'].apply(clean_text)
df['answer_clean'] = df['answer'].apply(clean_text)
```

| Action | Regex | Description |
|--------|-------|-------------|
| Supprime HTML | `<.*?>` | Enlève toutes les balises HTML |
| Normalise espaces | `\s+` | Remplace espaces multiples par un seul |
| Anonymise téléphone | `\d{3}-\d{3}-\d{4}` | Remplace numéros US par `<tel>` |

### 3. Features textuelles

```python
df['body_length'] = df['body_clean'].str.len()
df['answer_length'] = df['answer_clean'].str.len()
df['response_ratio'] = df['answer_length'] / (df['body_length'] + 1)
```

| Feature | Description |
|---------|-------------|
| `body_length` | Nombre de caractères du problème |
| `answer_length` | Nombre de caractères de la réponse |
| `response_ratio` | Ratio réponse/problème (qualité de réponse) |

### 4. Détection de keywords

```python
keywords = {
    'network':   r'\b(network|wifi|vpn|connect|internet|lan|router)\b',
    'printer':   r'\b(printer|imprimante|print|scan|scanner)\b',
    'security':  r'\b(security|securite|password|login|access|breach|hack|malware)\b',
    'hardware':  r'\b(hardware|laptop|pc|macbook|screen|disk|ssd|cpu)\b',
    'software':  r'\b(software|app|update|bug|crash|install|version)\b',
}

for k, regex in keywords.items():
    df[f'has_{k}'] = df['body_clean'].str.contains(regex, case=False, na=False).astype(int)
```

Crée des colonnes binaires (0/1) pour chaque catégorie de mots-clés détectés.

| Colonne | Mots-clés détectés |
|---------|-------------------|
| `has_network` | network, wifi, vpn, connect, internet, lan, router |
| `has_printer` | printer, imprimante, print, scan, scanner |
| `has_security` | security, password, login, access, breach, hack, malware |
| `has_hardware` | hardware, laptop, pc, macbook, screen, disk, ssd, cpu |
| `has_software` | software, app, update, bug, crash, install, version |

### 5. Sauvegarde

```python
df.to_sql(
    name='tickets_tech_en_enriched',
    con=engine,
    if_exists='replace',
    index=False,
    method='multi',
    chunksize=1000
)
```

---

## Nouvelles colonnes créées

| Colonne | Type | Description |
|---------|------|-------------|
| `body_clean` | text | Body nettoyé |
| `answer_clean` | text | Answer nettoyé |
| `body_length` | int | Longueur du body |
| `answer_length` | int | Longueur de la réponse |
| `response_ratio` | float | Ratio réponse/problème |
| `has_network` | int (0/1) | Contient mots-clés réseau |
| `has_printer` | int (0/1) | Contient mots-clés imprimante |
| `has_security` | int (0/1) | Contient mots-clés sécurité |
| `has_hardware` | int (0/1) | Contient mots-clés hardware |
| `has_software` | int (0/1) | Contient mots-clés software |

### 6. Rapports Evidently

```python
from evidently.report import Report
from evidently.metric_preset import DataQualityPreset, DataDriftPreset

# Data Quality Report
quality_report = Report(metrics=[DataQualityPreset()])
quality_report.run(reference_data=None, current_data=df)
quality_report.save_html("/opt/airflow/data/evidently_quality_report.html")

# Data Drift Report
drift_report = Report(metrics=[DataDriftPreset()])
drift_report.run(reference_data=df.head(9000), current_data=df.tail(8893))
drift_report.save_html("/opt/airflow/data/evidently_drift_report.html")
```

| Rapport | Description | Fichier |
|---------|-------------|---------|
| **Data Quality** | Analyse qualité des données (valeurs manquantes, distributions, types) | `evidently_quality_report.html` |
| **Data Drift** | Détecte les dérives entre 2 sous-ensembles | `evidently_drift_report.html` |

---

## Connexions

### Source
- **Table** : `tickets_tech_en`
- **Lignes** : 17 893

### Destination
- **Table** : `tickets_tech_en_enriched`
- **Database** : `support_tech`
- **Port** : `5433`

---

## Exécution

### Déclencher manuellement
1. Aller dans l'UI Airflow : `http://<IP>:8082`
2. Activer le DAG (toggle ON)
3. Cliquer sur "Trigger DAG"

### Vérifier le résultat
```sql
SELECT COUNT(*) FROM tickets_tech_en_enriched;  -- Doit retourner 17893

SELECT body_length, answer_length, response_ratio,
       has_network, has_security
FROM tickets_tech_en_enriched
LIMIT 5;
```

---

## Résultat attendu

| Métrique | Valeur |
|----------|--------|
| Lignes enrichies | 17 893 |
| Nouvelles colonnes | 10 |
| Status | SUCCESS |
