"""
app.py  -  Flask REST API Backend
Supports: KNN, SVM, Random Forest, Logistic Regression, AdaBoost, XGBoost.

Usage:
    cd backend
    python app.py  →  http://localhost:5000
"""

import os
import sys
import traceback
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

from flask import Flask, request, jsonify
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import load_and_clean, transform, load_splits

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
)

app = Flask(__name__)
CORS(app)

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BACKEND_DIR)
MODELS_DIR  = os.path.join(PROJECT_DIR, "models")

results_store = {}

# ── Model factory ──────────────────────────────────────────────────
def build_model(model_key, params):
    """Instantiate any of the 6 supported models with given params."""

    def _int(p, d): return int(params.get(p, d))
    def _flt(p, d): return float(params.get(p, d))
    def _str(p, d): return str(params.get(p, d))

    if model_key == "knn":
        return KNeighborsClassifier(
            n_neighbors=_int("n_neighbors", 7),
            weights=_str("weights", "distance"),
            metric=_str("metric", "euclidean"),
        )
    if model_key == "svm":
        return SVC(
            C=_flt("C", 1.0),
            kernel=_str("kernel", "rbf"),
            gamma=_str("gamma", "scale"),
            probability=True, random_state=42,
        )
    if model_key == "random_forest":
        return RandomForestClassifier(
            n_estimators=_int("n_estimators", 100),
            max_depth=_int("max_depth", 10),
            min_samples_split=_int("min_samples_split", 5),
            random_state=42, class_weight="balanced",
        )
    if model_key == "logistic_regression":
        return LogisticRegression(
            C=_flt("C", 1.0),
            solver=_str("solver", "lbfgs"),
            max_iter=_int("max_iter", 1000),
            random_state=42, class_weight="balanced",
        )
    if model_key == "adaboost":
        base = DecisionTreeClassifier(
            max_depth=_int("base_max_depth", 1), random_state=42
        )
        return AdaBoostClassifier(
            estimator=base,
            n_estimators=_int("n_estimators", 100),
            learning_rate=_flt("learning_rate", 0.5),
            algorithm="SAMME",
            random_state=42,
        )
    if model_key == "xgboost":
        return XGBClassifier(
            n_estimators=_int("n_estimators", 200),
            max_depth=_int("max_depth", 5),
            learning_rate=_flt("learning_rate", 0.05),
            subsample=_flt("subsample", 0.8),
            colsample_bytree=_flt("colsample_bytree", 0.8),
            scale_pos_weight=_flt("scale_pos_weight", 2.77),
            eval_metric="logloss",
            random_state=42,
        )
    raise ValueError(f"Unknown model: {model_key}")


SUPPORTED_MODELS = ["knn", "svm", "random_forest",
                    "logistic_regression", "adaboost", "xgboost"]


# ── Routes ─────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "models_supported": SUPPORTED_MODELS})


@app.route("/models", methods=["GET"])
def list_models():
    saved = []
    if os.path.exists(MODELS_DIR):
        saved = [f.replace(".pkl", "")
                 for f in os.listdir(MODELS_DIR) if f.endswith(".pkl")]
    return jsonify({"available": SUPPORTED_MODELS, "saved": saved})


@app.route("/train", methods=["POST"])
def train():
    try:
        body      = request.get_json(force=True) or {}
        model_key = body.get("model", "random_forest")
        params    = body.get("params", {})

        if model_key not in SUPPORTED_MODELS:
            return jsonify({"error": f"Unknown model '{model_key}'. "
                            f"Choose from: {SUPPORTED_MODELS}"}), 400

        df = load_and_clean()
        X_train, X_test, y_train, y_test, _ = transform(df, save=True)

        model = build_model(model_key, params)
        model.fit(X_train, y_train)

        y_pred    = model.predict(X_test)
        y_prob    = model.predict_proba(X_test)[:, 1]
        train_acc = accuracy_score(y_train, model.predict(X_train))

        metrics = {
            "model":          model_key,
            "params":         params,
            "accuracy":       round(accuracy_score(y_test, y_pred), 4),
            "precision":      round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall":         round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1":             round(f1_score(y_test, y_pred, zero_division=0), 4),
            "roc_auc":        round(roc_auc_score(y_test, y_prob), 4),
            "train_accuracy": round(train_acc, 4),
        }

        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(model, os.path.join(MODELS_DIR, f"{model_key}.pkl"))
        results_store[model_key] = metrics

        return jsonify(metrics)

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/results", methods=["GET"])
def results():
    return jsonify(results_store)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        body      = request.get_json(force=True) or {}
        model_key = body.get("model", "random_forest")
        features  = body.get("features", {})

        model_path  = os.path.join(MODELS_DIR, f"{model_key}.pkl")
        scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")

        if not os.path.exists(model_path):
            return jsonify({"error": f"Model '{model_key}' not trained yet. "
                            f"Call POST /train first."}), 404
        if not os.path.exists(scaler_path):
            return jsonify({"error": "Scaler not found. Call POST /train first."}), 404

        model  = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        df_input = pd.DataFrame([{
            "tenure":         float(features.get("tenure", 12)),
            "MonthlyCharges": float(features.get("MonthlyCharges", 65)),
            "TotalCharges":   float(features.get("TotalCharges", 780)),
        }])
        df_input[["tenure", "MonthlyCharges", "TotalCharges"]] = scaler.transform(
            df_input[["tenure", "MonthlyCharges", "TotalCharges"]]
        )

        try:
            X_train, _, _, _, _ = load_splits()
            for col in X_train.columns:
                if col not in df_input.columns:
                    df_input[col] = 0
            df_input = df_input[X_train.columns]
        except Exception:
            pass

        prob  = float(model.predict_proba(df_input)[0][1])
        label = int(prob > 0.5)

        return jsonify({
            "churn_probability": round(prob, 4),
            "churn_label":       label,
            "churn_text":        "Yes" if label else "No",
            "model_used":        model_key,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/feature_importance", methods=["GET"])
def feature_importance():
    """
    Works for Random Forest, AdaBoost and XGBoost (all have feature_importances_).
    """
    for fname in ["random_forest.pkl", "xgboost.pkl", "adaboost.pkl"]:
        fp = os.path.join(MODELS_DIR, fname)
        if os.path.exists(fp):
            try:
                df = load_and_clean()
                X_train, _, _, _, _ = transform(df, save=False)
                model = joblib.load(fp)
                imp = dict(zip(
                    X_train.columns.tolist(),
                    [round(float(v), 6) for v in model.feature_importances_]
                ))
                return jsonify(dict(sorted(imp.items(), key=lambda x: -x[1])))
            except Exception as e:
                continue
    return jsonify({"error": "No model with feature_importances_ found."}), 404


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
