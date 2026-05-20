# Tâche 5 — Pipeline MLOps Local

## Fichiers ajoutés

| Fichier | Description |
|---|---|
| `backend/mlflow_registry.py` | Partie 3 — Model Registry (Staging → Production) |
| `backend/mlflow_serving.py` | Partie 4 — API REST FastAPI |
| `backend/detect_drift.py` | Partie 6 — Détection drift (Evidently + KS-test) |
| `backend/mlflow_client_query.py` | Partie 2 — Requêtes MlflowClient + réponses Q1-Q16 |
| `Makefile` | Partie 5 — Orchestration pipeline |
| `pre-commit` | Partie 5 — Hook Git validation |
| `backend/requirements.txt` | Dépendances mises à jour |

---

## Ordre d'exécution

### Windows (PowerShell)

```powershell
# 0. Placer le CSV dans backend\data\raw\
# Télécharger : https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets

# 1. Installer les dépendances (Tâche 5 complète)
cd backend
pip install -r requirements.txt

# 2. Entraîner les 6 modèles (Partie 1 + 2)
python train.py

# 3. Lancer MLflow UI dans un NOUVEAU terminal
cd ..
mlflow ui --backend-store-uri .\mlruns --port 5001
# → Ouvrir http://localhost:5001

# 4. Enregistrer le meilleur modèle dans le Registry (Partie 3)
cd backend
python mlflow_registry.py

# 5. Démarrer l'API REST (Partie 4)
python mlflow_serving.py
# → API sur http://localhost:1234
# → Docs sur http://localhost:1234/docs

# 6. Tester l'API (dans un autre terminal)
python mlflow_serving.py --test

# 7. Détecter le drift (Partie 6)
python detect_drift.py

# 8. Voir les réponses aux questions de réflexion
python mlflow_client_query.py
```

### Linux/Mac

```bash
make setup    # installer dépendances
make train    # entraîner les modèles
make mlflow   # lancer MLflow UI
make registry # Model Registry
make serve    # API REST
make test     # tester l'API
make drift    # détection drift
make pipeline # tout en une commande
```

---

## Installation hook Git

```bash
# Linux/Mac
cp pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit

# Windows PowerShell
Copy-Item pre-commit .git\hooks\pre-commit
```

---

## Architecture Pipeline MLOps

```
Données brutes (CSV Kaggle)
    |
[preprocessing.py] → nettoyage, OHE, StandardScaler
    |
[train.py] → 6 modèles + MLflow Tracking (params, métriques, artefacts)
    |
[mlflow_client_query.py] → identifier le meilleur run
    |
[mlflow_registry.py] → Model Registry → Staging → Production
    |
[mlflow_serving.py] → API REST FastAPI (port 1234)
    |
[detect_drift.py] → Evidently + KS-test + MLflow monitoring
    |
drift > 30% ? → OUI → relancer train.py (boucle fermée)
              → NON → surveillance continue
```

---

## Réponses aux questions — résumé

Les réponses complètes aux 16 questions (Q1–Q16) sont dans :
```
python backend/mlflow_client_query.py
```
