# 📰 Classification de Texte - Pipeline MLOps Complet

> **Auteurs** : Akram BENHAMMOU - Oussama KHOUYA  
> **Master 2** - DevOps & Machine Learning

---

## 🎯 Résumé du Projet

Ce projet implémente un **pipeline MLOps complet** pour la classification automatique d'articles de presse. Il couvre toutes les étapes depuis l'entraînement du modèle jusqu'au déploiement en production avec CI/CD automatisé.

### Objectif
Permettre à un journal en ligne de **catégoriser automatiquement** ses articles en 7 grandes catégories :
- 💻 Informatique
- ⚽ Sport
- 🔬 Science
- 🏛️ Politique
- ⛪ Religion
- 🚗 Automobile
- 🛒 Commerce

---

## 🏗️ Architecture Globale

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              PIPELINE MLOps                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   [1] DONNÉES          [2] MODÈLE           [3] API            [4] FRONTEND │
│   ─────────────        ──────────           ─────              ────────     │
│   20 Newsgroups   →    RandomForest    →    FastAPI      →     Angular     │
│   TF-IDF               + MLflow             + Docker            + Design    │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                              CI/CD (GitHub Actions)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│   cml.yaml              docker.yaml              deploy.yaml                │
│   ─────────             ────────────             ────────────               │
│   Rapport métriques  →  Build Docker image   →   Staging → Production      │
│   + Matrice confusion   Push to GHCR             + Rollback automatique    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Structure du Projet

```
Projet-MLOPS-Classification-text-CI-CD/
│
├── 📂 src/                          # Code source Python
│   ├── preprocess.py                # Nettoyage des données (NLP)
│   ├── train.py                     # Entraînement + MLflow
│   ├── predict.py                   # Script de prédiction standalone
│   └── app.py                       # API FastAPI
│
├── 📂 tests/                        # Tests automatisés
│   ├── test_preprocess.py           # Tests unitaires preprocessing
│   ├── test_train.py                # Tests des artefacts
│   └── test_api.py                  # Tests d'intégration API
│
├── 📂 frontend/                     # Application Angular
│   └── src/app/
│       ├── app.ts                   # Logique composant
│       ├── app.html                 # Template
│       └── app.css                  # Styles
│
├── 📂 .github/workflows/            # CI/CD
│   ├── cml.yaml                     # Rapport CML
│   ├── docker.yaml                  # Build & Push Docker
│   └── deploy.yaml                  # Staging/Production/Rollback
│
├── 📂 data/processed/               # Données nettoyées (CSV)
├── 📂 models/                       # Modèles entraînés (.joblib)
├── 📂 reports/                      # Métriques et visualisations
├── 📂 mlruns/                       # Logs MLflow
│
├── Dockerfile                       # Configuration Docker
├── requirements.txt                 # Dépendances Python
└── README.md                        # Ce fichier
```

---

## 🚀 Guide de Simulation Étape par Étape

### **ÉTAPE 1 : Prétraitement des Données**

```bash
# Exécuter le prétraitement
python src/preprocess.py
```

**Ce que ça fait :**
- Télécharge le dataset **20 Newsgroups** (18,000 articles)
- Nettoie le texte (minuscules, suppression ponctuation)
- Lemmatisation (réduction des mots à leur racine)
- Suppression des stop-words anglais
- Sauvegarde dans `data/processed/train.csv` et `test.csv`

---

### **ÉTAPE 2 : Entraînement du Modèle**

```bash
# Entraîner le modèle
python src/train.py
```

**Ce que ça fait :**
- Charge les données prétraitées
- Vectorise avec **TF-IDF** (Term Frequency-Inverse Document Frequency)
- Entraîne un **RandomForestClassifier** (100 arbres)
- Évalue le modèle (accuracy, precision, recall, F1)
- Sauvegarde les artefacts :
  - `models/model.joblib` - Modèle entraîné
  - `models/tfidf_vectorizer.joblib` - Vectorizer
  - `reports/metrics.json` - Métriques
  - `reports/confusion_matrix.png` - Matrice de confusion
- Enregistre tout dans **MLflow**

**Visualiser MLflow :**
```bash
mlflow ui
# Ouvrir http://localhost:5000
```

---

### **ÉTAPE 3 : Lancer les Tests**

```bash
# Exécuter tous les tests
pytest tests/ -v
```

**Tests disponibles :**
| Fichier | Nombre | Description |
|---------|--------|-------------|
| `test_preprocess.py` | 4 tests | Nettoyage, lemmatisation, stop-words |
| `test_train.py` | 3 tests | Vérification artefacts générés |
| `test_api.py` | 3 tests | Health check, prédiction API |

---

### **ÉTAPE 4 : Démarrer l'API Backend**

```bash
# Lancer l'API FastAPI
uvicorn src.app:app --reload
```

**Endpoints disponibles :**

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `GET` | `/health` | Vérifie que l'API fonctionne |
| `POST` | `/predict` | Classifie un texte |
| `POST` | `/upload` | Classifie un fichier (PDF, DOCX, TXT, MD) |

**Tester avec curl/PowerShell :**
```powershell
# Health check
Invoke-RestMethod -Uri "http://localhost:8000/health"

# Prédiction
Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -ContentType "application/json" -Body '{"text": "The basketball game was amazing"}'
```

**Documentation interactive :**
- Swagger UI : http://localhost:8000/docs
- ReDoc : http://localhost:8000/redoc

---

### **ÉTAPE 5 : Démarrer le Frontend Angular**

```bash
# Aller dans le dossier frontend
cd frontend

# Installer les dépendances (première fois seulement)
npm install

# Lancer l'application
ng serve
```

**Ouvrir :** http://localhost:4200

**Fonctionnalités :**
- ✅ Zone de texte pour saisir un article
- ✅ Import de fichiers (TXT, MD, PDF, DOCX)
- ✅ Classification en un clic
- ✅ Affichage de la catégorie avec icône

---

### **ÉTAPE 6 : Build Docker**

```bash
# Construire l'image Docker
docker build -t text-classifier .

# Lancer le conteneur
docker run -p 8000:8000 text-classifier

# Tester
curl http://localhost:8000/health
```

---

### **ÉTAPE 7 : CI/CD (GitHub Actions)**

Les workflows se déclenchent automatiquement lors d'un `git push` sur `master`.

#### Workflow 1 : `cml.yaml` - Rapport de Métriques
```bash
git add .
git commit -m "feat: nouvelle fonctionnalité"
git push origin master
# → Génère un commentaire avec les métriques sur GitHub
```

#### Workflow 2 : `docker.yaml` - Build & Push Docker
```bash
# Après le push, l'image est disponible sur :
docker pull ghcr.io/akrambenhammou-e/classification-texte-pipeline-ci-cd:latest
```

#### Workflow 3 : `deploy.yaml` - Déploiement
```
Staging → Tests d'intégration → Production (si OK)
                              → Rollback (si échec)
```

---

## 📊 Performances du Modèle

| Métrique | Valeur |
|----------|--------|
| **Accuracy** | ~64% |
| **Precision** | ~63% |
| **Recall** | ~64% |
| **F1-Score** | ~63% |

**Note** : Le modèle est entraîné sur des textes **anglais**. Les textes français ne seront pas correctement classifiés.

---

## 🛠️ Technologies Utilisées

| Catégorie | Technologies |
|-----------|--------------|
| **ML/NLP** | scikit-learn, NLTK, pandas |
| **Tracking** | MLflow |
| **API** | FastAPI, Uvicorn, Pydantic |
| **Frontend** | Angular 19, TypeScript |
| **Tests** | pytest, httpx |
| **CI/CD** | GitHub Actions, CML |
| **Container** | Docker |
| **Registry** | GitHub Container Registry |

---

## ⚡ Commandes Rapides

```bash
# Tout lancer en une fois (2 terminaux nécessaires)

# Terminal 1 - Backend
cd "c:/Users/Akram/Documents/M2-S3/devops&M/Projet-MLOPS-Classification-text-CI-CD"
uvicorn src.app:app --reload

# Terminal 2 - Frontend
cd "c:/Users/Akram/Documents/M2-S3/devops&M/Projet-MLOPS-Classification-text-CI-CD/frontend"
ng serve
```

**URLs :**
- Frontend : http://localhost:4200
- API : http://localhost:8000
- API Docs : http://localhost:8000/docs
- MLflow : http://localhost:5000 (si lancé)

---

## 📄 Licence

Projet réalisé dans le cadre du **Master 2 - DevOps & Machine Learning**  
Copyright © 2025 - Akram BENHAMMOU & Oussama KHOUYA
