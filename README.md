# ChurnML — Customer Churn Prediction

Prédiction du churn client sur le **Telecom Customers Churn Dataset** (Kaggle).  
Projet MLA — Tâches 2, 3 & 4.

---

## Structure du projet

```
project_nomProjet/
│
├── backend/                    # API Python + MLflow
│   ├── app.py                  # Serveur Flask (REST API)
│   ├── train.py                # Entraînement + MLflow (Tâche 3)
│   ├── task4_random_forest_analysis.py   # Analyse RF (Tâche 4)
│   ├── evaluate.py             # Évaluation des modèles
│   ├── data_loader.py          # Chargement du CSV
│   ├── preprocessing.py        # Nettoyage + encodage + split
│   ├── requirements.txt
│   └── data/
│       ├── raw/                <- placer le CSV ici
│       └── processed/          <- splits auto-générés (gitignored)
│
├── frontend/                   # Interface React
│   ├── package.json
│   ├── public/
│   │   └── index.html
│   └── src/
│       ├── index.js            # Point d'entrée React
│       ├── index.css           # Styles globaux (thème warm)
│       ├── App.js              # Application principale (6 onglets)
│       ├── components/
│       │   └── PredictForm.js  # Formulaire de prédiction standalone
│       └── services/
│           └── api.js          # Tous les appels REST vers le backend
│
├── mlruns/                     # Runs MLflow (auto-générés, gitignored)
├── models/                     # Modèles .pkl sauvegardés (gitignored)
├── .gitignore
└── README.md
```

---

## Installation & Lancement

### 1. Cloner le repo

```bash
git clone <votre-repo-url>
cd project_nomProjet
```

### 2. Backend Python

```bash
cd backend

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# Installer les dépendances
pip install -r requirements.txt

# Placer le dataset
# -> backend/data/raw/Telecom_Customers_Churn.csv

# Prétraitement
python preprocessing.py

# Entraînement (Tâche 3) — log dans MLflow
python train.py

# Analyse Random Forest (Tâche 4)
python task4_random_forest_analysis.py

# Lancer l'API
python app.py
# -> http://localhost:5000
```

### 3. Frontend React

```bash
cd frontend
npm install
npm start
# -> http://localhost:3000
```

### 4. MLflow UI

```bash
# Depuis la racine du projet
mlflow ui --backend-store-uri ./mlruns --port 5001
# -> http://localhost:5001
```

---

## API Endpoints

| Méthode | Endpoint              | Description                              |
|---------|-----------------------|------------------------------------------|
| GET     | `/health`             | Statut de l'API                          |
| GET     | `/models`             | Modèles disponibles et sauvegardés       |
| POST    | `/train`              | Entraîner un modèle avec hyperparamètres |
| GET     | `/results`            | Récupérer toutes les métriques           |
| POST    | `/predict`            | Prédire le churn d'un client             |
| GET     | `/feature_importance` | Importances des variables (Random Forest)|

**Exemple — Entraîner un modèle :**
```bash
curl -X POST http://localhost:5000/train \
  -H "Content-Type: application/json" \
  -d '{"model": "random_forest", "params": {"n_estimators": 200, "max_depth": 10}}'
```

**Exemple — Prédire :**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"model": "random_forest", "features": {"tenure": 6, "MonthlyCharges": 75, "TotalCharges": 450}}'
```

---

## Résultats (Tâche 3 — Test Set : 1 409 exemples)

| Modèle                | Accuracy | Recall | F1    | AUC   |
|-----------------------|----------|--------|-------|-------|
| KNN (k=7)             | 75.9%    | 52.1%  | 0.534 | 0.788 |
| SVM (rbf, C=1)        | 79.5%    | 48.1%  | 0.555 | 0.795 |
| **Random Forest ★**   | 76.7%    | 73.3%  | 0.625 | 0.840 |
| Logistic Regression   | 73.9%    | 78.3%  | 0.614 | 0.842 |

**Modèle recommandé :** Random Forest (n=100, max_depth=10, class_weight='balanced')

---

## Tâche 4 — Conclusions clés

1. **Features les plus importantes :** `tenure`, `TotalCharges`, `MonthlyCharges`
2. **Stabilité :** σ_accuracy = 0.0018 sur 10 seeds → modèle très robuste
3. **Overfitting :** max_depth ≥ 10 (variance > 0.08)
4. **Underfitting :** max_depth ≤ 5 (biais > 0.25)
5. **Paramétrage optimal :** n=100, max_depth=7, class_weight='balanced'
6. **RF > Decision Tree :** +6 points d'accuracy à profondeur=3

---

## Technologies

| Couche       | Technologie                    |
|--------------|-------------------------------|
| Machine Learning | scikit-learn 1.3+          |
| Tracking     | MLflow 2.10+                   |
| Backend API  | Flask 3.0 + Flask-CORS         |
| Frontend     | React 18 + Axios               |
| Visualisation | Matplotlib + Seaborn          |
| Sérialisation | joblib                        |
