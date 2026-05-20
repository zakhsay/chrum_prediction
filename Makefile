# ════════════════════════════════════════════════════════════════
# Makefile — Pipeline MLOps Tache 5
# Usage : make <cible>
# ════════════════════════════════════════════════════════════════

.PHONY: setup train registry serve drift test pipeline help

PYTHON  = python
BACKEND = backend
PORT_MLFLOW = 5001
PORT_API    = 1234

help:
	@echo ""
	@echo "  ChurnML — Pipeline MLOps (Tache 5)"
	@echo "  ===================================="
	@echo "  make setup     : installer les dependances"
	@echo "  make train     : entrainer les 6 modeles (MLflow tracking)"
	@echo "  make registry  : enregistrer le meilleur modele dans le Registry"
	@echo "  make mlflow    : lancer MLflow UI sur http://localhost:$(PORT_MLFLOW)"
	@echo "  make serve     : demarrer l'API REST sur http://localhost:$(PORT_API)"
	@echo "  make drift     : detecter le data drift (Evidently + KS-test)"
	@echo "  make test      : tester l'API REST"
	@echo "  make pipeline  : executer train + registry + drift"
	@echo ""

# ── Installation ────────────────────────────────────────────────
setup:
	pip install -r $(BACKEND)/requirements.txt
	@echo "[OK] Dependances installees"

# ── Entrainement ─────────────────────────────────────────────────
train:
	@echo "=== Entrainement des 6 modeles ==="
	cd $(BACKEND) && $(PYTHON) train.py

# ── Tache 4 ──────────────────────────────────────────────────────
task4:
	@echo "=== Tache 4 : Analyse Random Forest ==="
	cd $(BACKEND) && $(PYTHON) task4_random_forest_analysis.py

# ── Model Registry ───────────────────────────────────────────────
registry:
	@echo "=== Enregistrement dans le Model Registry ==="
	cd $(BACKEND) && $(PYTHON) mlflow_registry.py

# ── MLflow UI ────────────────────────────────────────────────────
mlflow:
	@echo "=== MLflow UI sur http://localhost:$(PORT_MLFLOW) ==="
	mlflow ui --backend-store-uri ./mlruns --port $(PORT_MLFLOW)

# ── Serving API ──────────────────────────────────────────────────
serve:
	@echo "=== API REST sur http://localhost:$(PORT_API) ==="
	cd $(BACKEND) && $(PYTHON) mlflow_serving.py

# ── Test API ─────────────────────────────────────────────────────
test:
	@echo "=== Test de l'API REST ==="
	cd $(BACKEND) && $(PYTHON) mlflow_serving.py --test

# ── Detection du drift ───────────────────────────────────────────
drift:
	@echo "=== Detection du Data Drift ==="
	cd $(BACKEND) && $(PYTHON) detect_drift.py

# ── Requetes MLflow client ───────────────────────────────────────
query:
	@echo "=== Requetes MlflowClient + Questions reflexion ==="
	cd $(BACKEND) && $(PYTHON) mlflow_client_query.py

# ── Pipeline complet ─────────────────────────────────────────────
pipeline: train registry drift
	@echo ""
	@echo "╔══════════════════════════════════════════════╗"
	@echo "║  Pipeline MLOps complet execute avec succes  ║"
	@echo "╚══════════════════════════════════════════════╝"
	@echo ""
	@echo "  Prochaines etapes :"
	@echo "  1. make mlflow    → voir les resultats"
	@echo "  2. make serve     → demarrer l'API"
	@echo "  3. make test      → tester l'API"
