# 📋 FICHE PROJET FINAL MLOps - Data Engineering Bootcamp

---

## 🎯 Informations générales

| Élément | Détail |
|---------|--------|
| **Durée** | 180 minutes |
| **Module** | 7 - Final Projects |
| **Cours** | Préparez votre projet final |
| **Objectif principal** | Développer un pipeline MLOps entièrement fonctionnel qui automatise le cycle de vie complet d'un modèle de machine learning |

---

## 📦 Livrables obligatoires

### 1. Rapport sur le dataset et le prétraitement
- Explication claire du dataset choisi
- Source des données (publique ou collectée)
- Méthode de prétraitement appliquée
- Gestion des valeurs manquantes
- Traitement des valeurs aberrantes
- Équilibrage des données si nécessaire
- Feature engineering réalisé
- Justification de chaque transformation

### 2. Notebook ou script du modèle entraîné
- Code d'entraînement complet
- Choix de l'algorithme justifié (classification, régression, clustering...)
- Framework utilisé (TensorFlow, PyTorch, Scikit-learn)
- Hyperparameter tuning documenté
- Évaluation avec métriques appropriées :
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Autres métriques selon le cas d'usage
- Modèle robuste et prêt pour la production

### 3. Pipeline MLOps complet
- **Schéma d'architecture** du pipeline (diagramme visuel)
- **Code source** pour chaque composant :
  - Déploiement du modèle
  - Configuration CI/CD
  - Système de surveillance
  - Pipeline de réentraînement
- **Vidéo de présentation** ou captures d'écran montrant le processus en action

### 4. Dépôt de code GitHub
- Code source complet et organisé
- Instructions claires pour :
  - Installation des dépendances
  - Exécution du pipeline
  - Déploiement complet
- README détaillé

### 5. Documentation API
- Guide clair et concis
- Description des endpoints
- Format des entrées (inputs)
- Format des sorties (outputs)
- Exemples d'utilisation
- Instructions d'intégration pour tiers

### 6. Présentation (slides)
- Support visuel pour présenter devant la classe ou le jury
- Structure recommandée : Introduction forte → Développement → Conclusion
- Utiliser le template Jedha : https://docs.google.com/presentation/d/1gFp4J8irQJs_5SrzTWxxQUUex8lHHTNXtEZ2P-a1VVM/edit?usp=sharing

---

## 🏗️ Les 6 composants techniques obligatoires du pipeline

### a. Déploiement du modèle
- Conteneurisation avec **Docker**
- Orchestration avec **Kubernetes**
- Exposition via **API REST**
- Déploiement évolutif (scalable)
- Capable de gérer données et requêtes en temps réel
- Options : AWS, Google Cloud, Azure ou on-premise

### b. CI/CD (Intégration Continue / Déploiement Continu)
- Outils possibles : **GitHub Actions**, GitLab CI, Jenkins
- Pipelines automatisés pour :
  - Tests automatiques
  - Validation du code
  - Déploiement automatique
- Déclenchement à chaque mise à jour du modèle

### c. Surveillance et logging (Monitoring)
- Outils recommandés : **Evidently**, Aporia
- Métriques à suivre :
  - Latence des requêtes
  - Précision du modèle en production
  - Détection de drift (dérive des données)
- Configuration d'alertes :
  - Quand drift détecté
  - Quand précision descend sous un seuil défini

### d. Réentraînement automatisé (Continuous Training)
- Pipeline de retraining automatique
- Déclencheurs :
  - Détection de drift par le monitoring
  - Nouvelles données disponibles
- Outils recommandés : **Apache Airflow**, Kubeflow
- Mise à jour transparente du modèle en production

### e. Gestion des versions et rollback
- Outils recommandés : **MLflow**, DVC
- Versioning des données
- Versioning des modèles
- Capacité de rollback vers version précédente si problème

### f. API documentée
- Framework recommandé : **FastAPI** (génère /docs automatiquement)
- Documentation claire des :
  - Endpoints disponibles
  - Paramètres d'entrée
  - Format de sortie
  - Codes d'erreur
- Faciliter l'intégration par des tiers

---

## ✅ Critères d'évaluation (grille de notation)

### 1. Préparation des données et performances du modèle
- Dataset pertinent et stimulant sélectionné
- Prétraitement correctement réalisé
- Modèle performant entraîné
- Métriques satisfaisantes

### 2. Exhaustivité du pipeline
- Pipeline couvre l'intégralité du cycle de vie :
  - Déploiement ✓
  - Surveillance ✓
  - Réentraînement ✓
- Tous les composants sont connectés

### 3. Automatisation
- Processus de déploiement automatisé
- Processus de retraining automatisé
- Système robuste face aux changements de données
- Système robuste face aux baisses de performance

### 4. Évolutivité et surveillance
- Pipeline capable de scaler (plus de données, plus d'utilisateurs)
- Monitoring proactif et efficace
- Alertes configurées et fonctionnelles

### 5. Documentation et accessibilité
- Pipeline clairement documenté
- Code lisible et commenté
- Accessible aux futurs développeurs
- Instructions reproductibles

---

## 🔧 Architecture existante (ébauche Isabelle)

### Cas d'usage choisi
**Agent de support IT automatique**
- Classification automatique des tickets (sujet + urgence)
- Génération de réponses avec LLM (Mistral/OpenAI)

### Les 5 blocs du pipeline

#### Bloc 1 : Data Pipeline
```
PostgreSQL/Neon (raw tickets) 
    → Airflow DAG (ingestion + transformation)
    → dbt Core (transformation + tests)
    → PostgreSQL/Neon (features/marts)
```

#### Bloc 2 : Training Pipeline
```
Airflow DAG training
    → Ray (training distribué + tuning)
    → MLflow Tracking (params + metrics + artefacts)
    → Validation métriques (F1 + recall urgent)
    → MLflow Registry (versioning + Production)
```

#### Bloc 3 : Deployment
```
Docker (build image)
    → GitHub Actions (CI/CD)
    → Kubernetes (deployment)
```

#### Bloc 4 : Serving (Production temps réel)
```
Utilisateur (ticket)
    → FastAPI (K8s) - Prédiction sujet + urgence
    → Knowledge Base (recherche contexte)
    → API LLM externe (génération réponse)
    → Réponse à l'utilisateur
    + Logs production (inputs + preds + latence + feedback)
```

#### Bloc 5 : Monitoring + Retraining
```
Logs production
    → Airflow DAG monitoring
    → Evidently (drift données + distribution prédictions)
    → Si drift détecté → Déclenche retraining (retour Bloc 2)
    → Si pas de drift → Continue surveillance
```

### Stack technique complète
| Composant | Technologie |
|-----------|-------------|
| Base de données | PostgreSQL / Neon |
| Transformation données | dbt Core |
| Orchestration | Apache Airflow |
| Training distribué | Ray |
| Tracking ML | MLflow |
| Registry modèles | MLflow Model Registry |
| Conteneurisation | Docker |
| CI/CD | GitHub Actions |
| Orchestration containers | Kubernetes |
| API serving | FastAPI |
| Base de connaissances | Fichiers Markdown |
| LLM externe | Mistral / OpenAI API |
| Monitoring drift | Evidently AI |

---

## 📝 Checklist des tâches restantes

### Documents à produire
- [ ] Rédiger le rapport sur le dataset et prétraitement
- [ ] Documenter le notebook d'entraînement
- [ ] Créer le schéma d'architecture propre (export image)
- [ ] Structurer le repo GitHub avec README complet
- [ ] Enregistrer la vidéo de démonstration
- [ ] Vérifier la documentation API (/docs FastAPI)
- [ ] Créer les slides de présentation

### Code à implémenter/vérifier
- [ ] DAG Airflow : ingestion + dbt transformation
- [ ] DAG Airflow : training avec Ray
- [ ] DAG Airflow : monitoring Evidently
- [ ] Service FastAPI : chargement modèle depuis MLflow Registry
- [ ] Workflow GitHub Actions : build Docker + deploy K8s
- [ ] Manifests Kubernetes : Deployment + Service + ConfigMap
- [ ] Dockerfile optimisé
- [ ] Tests unitaires et d'intégration

### Flux à tester end-to-end
- [ ] Ingestion données → Features disponibles
- [ ] Training → Modèle enregistré dans Registry
- [ ] Push code → CI/CD → Déploiement automatique
- [ ] Requête API → Prédiction + Réponse LLM
- [ ] Logs → Drift détecté → Retraining déclenché
- [ ] Rollback vers version précédente du modèle

---

## 📚 Ressources pour trouver des données

### Open Data
- https://www.data.gov/
- https://www.enigma.com/
- https://snap.stanford.edu/data/index.html (données sociales)
- https://opendata.cityofnewyork.us/
- https://mattermark.com/
- https://www.crunchbase.com/
- https://www.kaggle.com/
- https://www.quandl.com/

### APIs publiques
- https://github.com/toddmotto/public-apis

### Inspiration projets IA
- https://experiments.withgoogle.com/collection/ai

### Recherche de datasets
- https://toolbox.google.com/datasetsearch

### Articles datasets Deep Learning
- https://www.analyticsvidhya.com/blog/2018/03/comprehensive-collection-deep-learning-datasets/

---

## 🎤 Conseils pour la présentation orale

### Structure recommandée
1. **Introduction** - Commencer fort avec les meilleurs arguments
2. **Développement** - Approfondir l'argumentation
3. **Conclusion** - Reprendre les points clés

### Storytelling
- Présenter un personnage (utilisateur type)
- Décrire le problème/défi qu'il rencontre
- Montrer les obstacles surmontés
- Raconter comment ça se termine (solution)

### Gestion du stress
- Power pose 30 secondes avant
- Bien manger avant (éviter la malbouffe excessive)
- Exercice physique si possible
- Préparation, préparation, préparation !

### Expression
- Parler plus fort que d'habitude
- Varier le rythme (ni trop vite, ni trop lent)
- Éviter la voix monotone
- Ne jamais tourner le dos au public
- Ne pas lire les slides, les mémoriser

### Règles slides (Guy Kawasaki)
- Maximum 10 slides
- Maximum 20 minutes
- Police minimum 30pt
- 1 slide = 1 idée
- Contraste fort (fond clair/texte foncé ou inverse)

---

## 🛠️ Outils de conception projet

### Value Proposition Canvas
Framework pour trouver un cas d'usage business :
- Jobs to be done
- Pains (problèmes)
- Gains (bénéfices attendus)
- Pain relievers
- Gain creators
- Products & Services

### AI Model Canvas (Jedha)
Framework spécifique pour projets IA :
- Problème à résoudre
- Données nécessaires
- Modèle/algorithme
- Infrastructure
- Métriques de succès
- Risques et limitations

### Kanban (GitHub Projects)
- Visualiser les étapes du projet
- Colonnes : To Do → In Progress → Done
- Faciliter le suivi de progression

---

## ⚠️ Points d'attention importants

### MVP (Minimum Viable Product)
- Créer un produit fonctionnel, pas parfait
- Prédictions acceptables et utiles
- Utilisateurs finaux peuvent l'utiliser
- Ne pas y passer trop de temps

### Commencer petit, devenir grand
- Réduire la portée initiale du projet
- Ne pas tout implémenter d'un coup (pas K8s + Spark + PyTorch dès le départ)
- Commencer en local, déployer progressivement
- Penser microservices (services indépendants)

### Cycle itératif données/modèle
1. Collecter données (minimum nécessaire)
2. Construire modèle
3. Si résultats insuffisants → plus de données + meilleure EDA
4. Réappliquer modèle + fine-tuning
5. Répéter jusqu'à résultats acceptables
6. Ne pas s'enliser : fixer un nombre max d'itérations

### Docker
- Tout conteneuriser le plus tôt possible
- Facilite la gestion en production
- Reproductibilité garantie

---

## 📅 Récapitulatif final

**Ce projet évalue ta capacité à :**
1. Trouver et préparer un dataset pertinent
2. Entraîner un modèle performant
3. Construire une infrastructure MLOps complète
4. Automatiser le cycle de vie ML
5. Monitorer et maintenir un modèle en production
6. Documenter et présenter ton travail

**Technologies clés à maîtriser :**
- Docker + Kubernetes (déploiement)
- Airflow (orchestration)
- MLflow (tracking + registry)
- Evidently (monitoring)
- GitHub Actions (CI/CD)
- FastAPI (serving)

**Ton avantage :** Tu as déjà une architecture complète et cohérente. Il te reste à implémenter, documenter et présenter.

---

*Fiche générée le 02/01/2026 - Projet Final MLOps Bootcamp Data Engineering*