"""
mlflow_serving.py — Tache 5 Partie 4
Serving du modele en Production via FastAPI.
Alternative au 'mlflow models serve' natif.

Installation :
    pip install fastapi uvicorn

Usage :
    cd backend
    python mlflow_serving.py
    → API disponible sur http://localhost:1234

Test :
    python mlflow_serving.py --test
"""

import os
import sys
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BACKEND_DIR)
MODELS_DIR  = os.path.join(PROJECT_DIR, "models")
MLRUNS_DIR  = os.path.join(PROJECT_DIR, "mlruns")


def test_with_requests():
    """
    Tache 5 Partie 4 etape 16 :
    Tester l'endpoint de prediction depuis Python (requests).
    """
    try:
        import requests
    except ImportError:
        print("[ERREUR] pip install requests")
        return

    base_url = "http://localhost:1234"

    print("\n=== Test de l'API REST de prediction ===")

    # Test /health
    try:
        r = requests.get(f"{base_url}/health", timeout=5)
        print(f"[GET /health] Status={r.status_code}  {r.json()}")
    except Exception as e:
        print(f"[ERREUR] Le serveur n'est pas demarre : {e}")
        print("  → Lancez d'abord : python mlflow_serving.py")
        return

    # Test /predict — profil client a risque eleve
    payload_high = {
        "dataframe_split": {
            "columns": ["tenure", "MonthlyCharges", "TotalCharges",
                        "contract_month", "internet_fiber"],
            "data": [[2, 85.0, 170.0, 1, 1]],
        }
    }
    r = requests.post(f"{base_url}/invocations", json=payload_high)
    print(f"\n[POST /invocations — client risque eleve]")
    print(f"  Status  : {r.status_code}")
    print(f"  Reponse : {r.json()}")

    # Test /predict/simple
    payload_simple = {
        "tenure": 6,
        "MonthlyCharges": 75.0,
        "TotalCharges": 450.0,
        "Contract": "Month-to-month",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "TechSupport": "No",
    }
    r = requests.post(f"{base_url}/predict/simple", json=payload_simple)
    print(f"\n[POST /predict/simple — client 6 mois, fibre, mensuel]")
    print(f"  Status  : {r.status_code}")
    print(f"  Reponse : {r.json()}")

    # Test /models
    r = requests.get(f"{base_url}/models")
    print(f"\n[GET /models]")
    print(f"  Modeles disponibles : {r.json()}")


def run_server():
    """Demarre le serveur FastAPI."""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        import uvicorn
        import joblib
        import numpy as np
        import pandas as pd
        import mlflow
    except ImportError as e:
        print(f"[ERREUR] Dependance manquante : {e}")
        print("  → pip install fastapi uvicorn mlflow")
        sys.exit(1)

    # Charger le meilleur modele disponible
    mlflow.set_tracking_uri(Path(MLRUNS_DIR).as_uri())

    model      = None
    scaler     = None
    model_name = "non charge"

    # Essayer de charger depuis MLflow Registry
    try:
        import mlflow.sklearn
        model = mlflow.sklearn.load_model(f"models:/churn_prediction_model/Production")
        model_name = "churn_prediction_model/Production (MLflow Registry)"
        print(f"[OK] Modele charge depuis MLflow Registry")
    except Exception:
        pass

    # Fallback : charger le .pkl local
    if model is None:
        for fname in ["Random_Forest.pkl", "XGBoost.pkl", "AdaBoost.pkl",
                      "SVM.pkl", "Logistic_Regression.pkl", "KNN.pkl"]:
            fpath = os.path.join(MODELS_DIR, fname)
            if os.path.exists(fpath):
                model      = joblib.load(fpath)
                model_name = fname
                print(f"[OK] Modele charge depuis : {fpath}")
                break

    if model is None:
        print("[ERREUR] Aucun modele trouve.")
        print("  → Lancez d'abord : python train.py")
        sys.exit(1)

    # Charger le scaler
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print(f"[OK] Scaler charge depuis : {scaler_path}")

    # ── Application FastAPI ──────────────────────────────────────
    app = FastAPI(
        title="ChurnML — API de prediction",
        description="Prediction du churn client — Tache 5 Partie 4",
        version="1.0.0",
    )
    app.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["*"], allow_headers=["*"])

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model": model_name,
            "serving": "Tache 5 - FastAPI",
        }

    @app.get("/models")
    def list_models():
        saved = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pkl")] \
                if os.path.exists(MODELS_DIR) else []
        return {"model_en_service": model_name, "modeles_disponibles": saved}

    @app.post("/invocations")
    def invocations(payload: dict):
        """
        Format MLflow natif — compatible avec 'mlflow models serve'.
        Payload attendu :
        {
          "dataframe_split": {
            "columns": ["col1", "col2", ...],
            "data": [[val1, val2, ...], ...]
          }
        }
        """
        try:
            if "dataframe_split" not in payload:
                raise HTTPException(status_code=400,
                    detail="Format attendu : {'dataframe_split': {'columns':[], 'data':[]}}")

            ds    = payload["dataframe_split"]
            df    = pd.DataFrame(ds["data"], columns=ds["columns"])
            preds = model.predict(df).tolist()
            proba = model.predict_proba(df)[:, 1].tolist() \
                    if hasattr(model, "predict_proba") else None

            return {
                "predictions": preds,
                "churn_probability": proba,
                "model_used": model_name,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/predict/simple")
    def predict_simple(customer: dict):
        """
        Prediction simplifiee — envoyer les features brutes du client.
        Exemple :
        {
          "tenure": 6,
          "MonthlyCharges": 75.0,
          "TotalCharges": 450.0,
          "Contract": "Month-to-month",
          "InternetService": "Fiber optic",
          "OnlineSecurity": "No",
          "TechSupport": "No"
        }
        """
        try:
            sys.path.insert(0, BACKEND_DIR)
            from preprocessing import load_and_clean, transform

            # Construire un DataFrame minimal
            row = {
                "tenure":          customer.get("tenure", 12),
                "MonthlyCharges":  customer.get("MonthlyCharges", 65.0),
                "TotalCharges":    customer.get("TotalCharges", 780.0),
            }
            df_input = pd.DataFrame([row])

            # Appliquer le scaler sur les colonnes numeriques
            if scaler is not None:
                df_input[["tenure", "MonthlyCharges", "TotalCharges"]] = \
                    scaler.transform(df_input[["tenure", "MonthlyCharges", "TotalCharges"]])

            # Charger les splits pour aligner les colonnes
            try:
                from preprocessing import load_splits
                X_train, _, _, _, _ = load_splits()
                for col in X_train.columns:
                    if col not in df_input.columns:
                        df_input[col] = 0
                df_input = df_input[X_train.columns]
            except Exception:
                pass

            prob  = float(model.predict_proba(df_input)[0][1]) \
                    if hasattr(model, "predict_proba") else None
            pred  = int(model.predict(df_input)[0])
            label = "Churn" if pred == 1 else "No Churn"
            risk  = ("High"   if prob and prob > 0.7 else
                     "Medium" if prob and prob > 0.4 else "Low")

            return {
                "churn_label":       label,
                "churn_probability": round(prob, 4) if prob else None,
                "risk_level":        risk,
                "model_used":        model_name,
                "input_received":    customer,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    print(f"\n{'='*55}")
    print(f"  ChurnML — API REST (Tache 5 Partie 4)")
    print(f"{'='*55}")
    print(f"  Modele : {model_name}")
    print(f"  URL    : http://localhost:1234")
    print(f"  Docs   : http://localhost:1234/docs")
    print(f"  Sante  : http://localhost:1234/health")
    print(f"{'='*55}\n")
    uvicorn.run(app, host="0.0.0.0", port=1234)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="Tester l'API (le serveur doit etre demarre)")
    args = parser.parse_args()

    if args.test:
        test_with_requests()
    else:
        run_server()
