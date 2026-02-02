# 🚨 Fraud Detection System - Real-Time MLOps Pipeline

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Airflow](https://img.shields.io/badge/Airflow-3.0-green.svg)](https://airflow.apache.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Système de détection de fraude en temps réel avec pipeline MLOps complet : ingestion de données, preprocessing, prédiction automatisée et monitoring via Airflow, API REST, et dashboard interactif.

---

## 📊 Vue d'Ensemble

Ce projet implémente un **système de détection de fraude bancaire** avec :
- ✅ Pipeline ETL automatisé (Airflow)
- ✅ Prédictions en temps réel (toutes les 20 secondes)
- ✅ API REST pour inférence on-demand
- ✅ Dashboard de monitoring (Streamlit)
- ✅ Data warehouse (PostgreSQL)
- ✅ Model registry (HuggingFace Hub)
- ✅ Tracking expérimentations (MLflow - en développement)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SOURCES DE DONNÉES                           │
│  Real-Time API → Transactions toutes les 20 secondes            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AIRFLOW (Orchestration)                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  DAG ETL & Predict (20s)                                 │   │
│  │  ├─ Extract → Fetch API                                  │   │
│  │  ├─ Transform → Feature Engineering                      │   │
│  │  ├─ Load → Save to PostgreSQL                            │   │
│  │  └─ Predict → Random Forest Classifier                   │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  DAG Weekly Tests                                        │   │
│  │  └─ Model validation & data quality checks               │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                DATA WAREHOUSE (PostgreSQL)                      │
│  Tables: transactions | predictions | training_data             │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    APPLICATIONS                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐     │
│  │  Streamlit   │  │   API REST   │  │  MLflow (dev)      │     │
│  │  Dashboard   │  │  FastAPI     │  │  Experimentation   │     │
│  └──────────────┘  └──────────────┘  └────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Fonctionnalités Clés

### 1. Pipeline ETL Automatisé (Airflow)

**DAG Principal** (`dag_ETL_branch.py`) - Exécution toutes les 20 secondes :
- 📥 **Extract** : Récupération des transactions via API REST
- ⚙️ **Transform** : Feature engineering (agrégations, encodages, normalisation)
- 💾 **Load** : Stockage dans PostgreSQL (data warehouse)
- 🔮 **Predict** : Prédiction fraude/légitime (Random Forest)
- 📊 **Branch** : Routing conditionnel selon résultat

**DAG Testing** (`dag_weekly_tests.py`) - Exécution hebdomadaire :
- ✅ Data quality checks
- ✅ Model performance validation
- ✅ Pipeline health monitoring

### 2. API REST (FastAPI)

Endpoint : `POST /predict`
- Input : Transaction JSON
- Output : Probabilité de fraude + prédiction binaire
- Feature engineering intégré
- Modèle chargé depuis HuggingFace Hub

### 3. Dashboard Streamlit

Interface interactive pour :
- 📊 Visualisation des transactions en temps réel
- 📈 Statistiques de détection (taux de fraude, précision)
- 🔍 Analyse exploratoire des données
- 📉 Métriques de performance du modèle

### 4. MLflow (En développement)

- Tracking des expérimentations
- Model registry
- Retraining pipeline
- Backend : Neon PostgreSQL
- Artifacts : Cloudflare R2

---

## 🚀 Installation & Déploiement

### Prérequis

- Docker & Docker Compose
- Python 3.12+
- 4 GB RAM minimum
- Git

### 1. Cloner le Repository

```bash
git clone https://github.com/Ter0rra/fraud-detector.git
cd fraud-detector
```

### 2. Configuration

```bash
# Copier le fichier d'environnement
cp .env.example .env

# Éditer les variables
nano .env
```

**Variables requises** :
```bash
# Database
DATABASE_URL=postgresql://user:pass@host:5432/fraud_detection

# HuggingFace (model registry)
HF_MODEL_REPO=user/repo
HF_TOKEN=hf_xxxxx

# API Data Source
API_URL=<Url real time api>

# MLflow (optionnel)
MLFLOW_TRACKING_URI=https://your-mlflow-server.hf.space

# SMTP (alertes optionnelles)
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
ALERT_EMAIL_TO=alerts@example.com
```

### 3. Initialiser la Base de Données

```bash
# Exécuter le script d'initialisation
python 05_script/init_db_v2.py
```

### 4. Uploader le Modèle vers HuggingFace

```bash
# (Première fois seulement)
python 05_script/add_model.py
```

### 5. Démarrer les Services

```bash
# Lancer tous les services
docker-compose up --build -d

# Vérifier les logs
docker-compose logs -f
```

### 6. Accéder aux Applications

| Service | URL | Description |
|---------|-----|-------------|
| **Airflow UI** | http://localhost:8080 | Orchestration & monitoring |
| **Streamlit Dashboard** | http://localhost:8501 | Visualisation données |
| **API REST** | http://localhost:8000 | Endpoint prédiction |
| **MLflow** | http://localhost:5000 | Tracking (dev) |

**Credentials Airflow** : `admin` / `admin`

---

## 📂 Structure du Projet

```
fraud_detector/
├── 00_notebook/               # Notebooks expérimentation
│   └── train_model_RFC.ipynb # Training & hyperparameter search
├── 01_data/                   # Configuration data sources
│   ├── csv_path.txt          # Path training dataset
│   └── real_time_API.txt     # API endpoint
├── 02_airflow/               # Orchestration Airflow
│   ├── dags/
│   │   ├── dag_ETL_branch.py    # Pipeline principal (20s)
│   │   └── dag_weekly_tests.py  # Tests hebdomadaires
│   ├── config/airflow.cfg    # Configuration Airflow
│   ├── logs/                 # Logs d'exécution
│   └── plugins/              # Custom plugins
├── 03_applications/          # Applications déployées
│   ├── API/                  # FastAPI service
│   │   ├── app.py
│   │   ├── feature_engineering.py
│   │   └── Dockerfile
│   ├── Streamlit/            # Dashboard
│   │   ├── app.py
│   │   └── Dockerfile
│   └── MLflow/               # Tracking (dev)
│       ├── start.sh
│       └── Dockerfile
├── 04_models/                # Models legacy (local)
│   ├── fraud_model.pkl
│   └── preprocessor.pkl
├── 05_script/                # Scripts utilitaires
│   ├── init_db_v2.py         # Init database
│   └── add_model.py          # Upload model HF
├── docker-compose.yaml       # Orchestration services
└── README.md                 # Documentation
```

---

## 🎓 Exigences Projet Certification

### 1. ✅ Data Pipeline Automatisé

**Exigence** : Pipeline ETL complet avec orchestration

**Implémentation** :
- **Airflow DAG** : Extraction, transformation, chargement toutes les 20 secondes
- **Branching** : Logique conditionnelle selon fraude détectée
- **Error handling** : Retry automatique, logging détaillé
- **Monitoring** : UI Airflow + logs temps réel

### 2. ✅ Feature Engineering Reproductible

**Exigence** : Preprocessing cohérent train/production

**Implémentation** :
- **Pipeline scikit-learn** : ColumnTransformer sauvegardé
- **Versioning** : Preprocessor versionné sur HuggingFace
- **Réutilisabilité** : Même preprocessing DAG + API + retraining
- **Documentation** : Feature engineering expliqué (notebook)

### 3. ✅ Model Deployment & Serving

**Exigence** : Modèle accessible en production

**Implémentation** :
- **Model Registry** : HuggingFace Hub (versionning)
- **API REST** : FastAPI pour inférence on-demand
- **Batch Predictions** : Via Airflow DAG (temps réel)
- **Load from cloud** : Téléchargement automatique depuis HF

### 4. ✅ Data Warehouse

**Exigence** : Stockage structuré pour analytics

**Implémentation** :
- **PostgreSQL** : Data warehouse production
- **Tables** : 
  - `transactions` : Données brutes
  - `predictions` : Résultats modèle
  - `training_data` : Historique pour retraining
- **Optimisations** : Index, partitioning par date

### 5. ✅ Monitoring & Observability

**Exigence** : Suivi performance et qualité données

**Implémentation** :
- **Airflow UI** : Monitoring pipeline (succès/échecs)
- **Streamlit Dashboard** : Métriques temps réel
- **Weekly Tests** : Validation automatique modèle
- **Logs** : Traçabilité complète (Docker volumes)

### 6. ✅ MLOps Best Practices

**Exigence** : Industrialisation ML

**Implémentation** :
- **CI/CD** : Docker Compose, reproductibilité
- **Versioning** : Git + HuggingFace model registry
- **Config management** : Variables d'environnement (.env)
- **Scalability** : Architecture microservices
- **Testing** : DAG tests hebdomadaires

### 7. ✅ Documentation & Reproductibilité

**Exigence** : Projet facilement déployable

**Implémentation** :
- **README complet** : Installation pas-à-pas
- **Docker** : Environnement isolé, reproductible
- **Comments** : Code documenté
- **Architecture diagram** : Vue d'ensemble système

---

## 🛠️ Technologies Utilisées

| Composant | Technologie | Version |
|-----------|-------------|---------|
| **Orchestration** | Apache Airflow | 2.8.0 |
| **ML Framework** | scikit-learn | 1.4.0 |
| **Database** | PostgreSQL | 13 |
| **API** | FastAPI | 0.109.0 |
| **Dashboard** | Streamlit | 1.30.0 |
| **Tracking** | MLflow | 2.16.2 |
| **Model Registry** | HuggingFace Hub | - |
| **Containerization** | Docker / Docker Compose | 3.0 + |
| **Language** | Python | 3.12 |

---

## 📊 Performance du Modèle

**Algorithme** : Random Forest Classifier (100 estimators)

**Métriques de Production** :
- 🎯 **Recall** : 92% (objectif : détecter les fraudes)
- ⚖️ **Precision** : 78% (minimiser faux positifs)
- 📈 **F1-Score** : 84%
- ✅ **Accuracy** : 95%

**Features Principales** :
1. Montant de la transaction
2. Heure de la transaction
3. Agrégations client (moyenne, écart-type)
4. Fréquence transactions récentes
5. Catégorie marchand

---

## 🔄 Workflow Complet

### 1. Développement (Notebook)

```bash
# Expérimentation & training
jupyter notebook 00_notebook/train_model_RFC.ipynb
```

### 2. Déploiement Modèle

```bash
# Upload vers HuggingFace
python 05_script/add_model.py
```

### 3. Lancement Production

```bash
# Démarrer tous les services
docker-compose up -d
```

### 4. Monitoring

```bash
# Airflow : http://localhost:8080
# Streamlit : http://localhost:8501
# Logs : docker-compose logs -f
```

### 5. Prédictions

**Via DAG Airflow** : Automatique toutes les 20 secondes

**Via API** :
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "trans_num": "T123456",
    "amt": 125.50,
    "merch_lat": 40.7128,
    "merch_long": -74.0060,
    "category": "grocery_pos",
    "unix_time": 1609459200
  }'
```

---

## 🚧 Roadmap

### En Développement

- [ ] **MLflow Integration** : Connexion pipeline retraining
- [ ] **Model Retraining** : DAG automatique mensuel
- [ ] **A/B Testing** : Comparaison versions modèles
- [ ] **Alerting** : Notifications Slack/Email

### Améliorations Futures

- [ ] **GPU Support** : Deep Learning models
- [ ] **Real-time Streaming** : Kafka integration
- [ ] **Feature Store** : Feast implementation
- [ ] **Explainability** : SHAP values dans dashboard
- [ ] **Auto-scaling** : Kubernetes deployment

---

## 🤝 Contribution

Les contributions sont les bienvenues ! 

1. Fork le projet
2. Créer une branche (`git checkout -b feature/AmazingFeature`)
3. Commit (`git commit -m 'Add AmazingFeature'`)
4. Push (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

---

## 📄 License

Ce projet est sous licence MIT - voir [LICENSE](LICENSE) pour détails.

---

## 👤 Auteur

**Terorra** - Data Scientist & ML Engineer

- GitHub: [@Ter0rra](https://github.com/Ter0rra)
- HuggingFace: [@Terorra](https://huggingface.co/Terorra)

---

## 🙏 Remerciements

- **Jedha Bootcamp** : Formation MLOps
- **HuggingFace** : Model registry gratuit
- **Apache Airflow** : Orchestration puissante
- **Streamlit** : Dashboarding simplifié

---

## 📞 Support

Pour toute question ou problème :
- 🐛 [Ouvrir une Issue](https://github.com/Ter0rra/fraud-detector/issues)
- 💬 Discussion : [GitHub Discussions](https://github.com/Ter0rra/fraud-detector/discussions)

---

<div align="center">

**⭐ N'oubliez pas de star le projet si vous le trouvez utile ! ⭐**

Made with ❤️ by Terorra | Projet Certification MLOps 2024

</div>
