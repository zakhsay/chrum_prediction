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
import mlflow
import mlflow.sklearn
from pathlib import Path

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
    f1_score, roc_auc_score, confusion_matrix,
)

app = Flask(__name__)
CORS(app)

BACKEND_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR  = os.path.dirname(BACKEND_DIR)
MODELS_DIR   = os.path.join(PROJECT_DIR, "models")
MLRUNS_DIR   = os.path.join(PROJECT_DIR, "mlruns")
EXPERIMENT   = "churn_prediction_task3"

mlflow.set_tracking_uri(Path(MLRUNS_DIR).as_uri())
mlflow.set_experiment(EXPERIMENT)

results_store = {}

SUPPORTED_MODELS = ["knn", "svm", "random_forest",
                    "logistic_regression", "adaboost", "xgboost"]

# ── Startup caches ─────────────────────────────────────────────────
_X_TRAIN_COLS = None
_MODEL_CACHE  = {}
_SCALER       = None

try:
    _xtr = pd.read_csv(
        os.path.join(BACKEND_DIR, "data", "processed", "X_train.csv"), nrows=0
    )
    _X_TRAIN_COLS = _xtr.columns.tolist()
except Exception:
    pass


def _get_model(key):
    if key not in _MODEL_CACHE:
        path = os.path.join(MODELS_DIR, f"{key}.pkl")
        if os.path.exists(path):
            _MODEL_CACHE[key] = joblib.load(path)
    return _MODEL_CACHE.get(key)


def _get_scaler():
    global _SCALER
    if _SCALER is None:
        path = os.path.join(MODELS_DIR, "scaler.pkl")
        if os.path.exists(path):
            _SCALER = joblib.load(path)
    return _SCALER


def _init_results():
    """Evaluate all saved models on startup and populate results_store."""
    try:
        X_train, X_test, y_train, y_test, _ = load_splits()
    except Exception:
        return
    for model_key in SUPPORTED_MODELS:
        path = os.path.join(MODELS_DIR, f"{model_key}.pkl")
        if not os.path.exists(path):
            continue
        try:
            m = joblib.load(path)
            y_pred    = m.predict(X_test)
            y_prob    = m.predict_proba(X_test)[:, 1]
            train_acc = accuracy_score(y_train, m.predict(X_train))
            cm        = confusion_matrix(y_test, y_pred).tolist()
            results_store[model_key] = {
                "model":          model_key,
                "params":         {},
                "accuracy":       round(accuracy_score(y_test, y_pred), 4),
                "precision":      round(precision_score(y_test, y_pred, zero_division=0), 4),
                "recall":         round(recall_score(y_test, y_pred, zero_division=0), 4),
                "f1":             round(f1_score(y_test, y_pred, zero_division=0), 4),
                "roc_auc":        round(roc_auc_score(y_test, y_prob), 4),
                "train_accuracy": round(train_acc, 4),
                "cm":             cm,
            }
            _MODEL_CACHE[model_key] = m
            print(f"[init] {model_key}: acc={results_store[model_key]['accuracy']}")
        except Exception as e:
            print(f"[init] Failed {model_key}: {e}")

_init_results()
_get_scaler()


# ── Model factory ──────────────────────────────────────────────────
def build_model(model_key, params):
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
            C=_flt("C", 1.0), kernel=_str("kernel", "rbf"),
            gamma=_str("gamma", "scale"), probability=True, random_state=42,
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
            C=_flt("C", 1.0), solver=_str("solver", "lbfgs"),
            max_iter=_int("max_iter", 1000), random_state=42, class_weight="balanced",
        )
    if model_key == "adaboost":
        base = DecisionTreeClassifier(max_depth=_int("base_max_depth", 1), random_state=42)
        return AdaBoostClassifier(
            estimator=base, n_estimators=_int("n_estimators", 100),
            learning_rate=_flt("learning_rate", 0.5),
            algorithm="SAMME", random_state=42,
        )
    if model_key == "xgboost":
        return XGBClassifier(
            n_estimators=_int("n_estimators", 200),
            max_depth=_int("max_depth", 5),
            learning_rate=_flt("learning_rate", 0.05),
            subsample=_flt("subsample", 0.8),
            colsample_bytree=_flt("colsample_bytree", 0.8),
            scale_pos_weight=_flt("scale_pos_weight", 2.77),
            eval_metric="logloss", random_state=42,
        )
    raise ValueError(f"Unknown model: {model_key}")


# ── Routes ─────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    from flask import Response
    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>ChurnML API</title>
  <style>
    *{box-sizing:border-box;margin:0;padding:0}
    body{font-family:'Segoe UI',system-ui,sans-serif;background:#f5f2ed;color:#1a1714;padding:40px}
    h1{font-size:22px;font-weight:600;margin-bottom:4px}
    .sub{color:#7a7268;font-size:13px;margin-bottom:32px}
    .card{background:#fff;border:0.5px solid #dbd5cc;border-radius:10px;padding:24px;max-width:680px}
    table{width:100%;border-collapse:collapse;font-size:13px}
    th{text-align:left;padding:7px 10px;color:#7a7268;font-weight:500;font-size:11px;
       text-transform:uppercase;letter-spacing:.06em;border-bottom:1px solid #ede9e2}
    td{padding:9px 10px;border-bottom:0.5px solid #f5f2ed;vertical-align:top}
    .m{font-family:monospace;font-size:12px}
    .get{background:#edf4fb;color:#1a4f8a;padding:2px 7px;border-radius:4px;font-size:10px;font-weight:600}
    .post{background:#fdf0ef;color:#c0392b;padding:2px 7px;border-radius:4px;font-size:10px;font-weight:600}
    .badge{display:inline-block;padding:2px 8px;border-radius:8px;font-size:11px;font-weight:500;
           background:#edf7f1;color:#1e7e4a;border:0.5px solid #a8dfc0;margin-left:8px}
    a{color:#2980b9;text-decoration:none}
    a:hover{text-decoration:underline}
    .footer{margin-top:16px;font-size:11px;color:#afa99f}
  </style>
</head>
<body>
  <h1>ChurnML <span style="color:#c0392b">API</span></h1>
  <p class="sub">Customer Churn Prediction — Flask REST Backend &nbsp;·&nbsp; <span style="color:#27ae60">&#9679;</span> Running</p>
  <div class="card">
    <table>
      <thead><tr><th>Method</th><th>Endpoint</th><th>Description</th></tr></thead>
      <tbody>
        <tr><td><span class="get">GET</span></td><td class="m"><a href="/health">/health</a></td><td>API status &amp; supported models</td></tr>
        <tr><td><span class="get">GET</span></td><td class="m"><a href="/models">/models</a></td><td>Available &amp; saved models</td></tr>
        <tr><td><span class="get">GET</span></td><td class="m"><a href="/results">/results</a></td><td>Live metrics for all trained models</td></tr>
        <tr><td><span class="get">GET</span></td><td class="m"><a href="/stats">/stats</a></td><td>Dataset statistics (total, split, churn rate)</td></tr>
        <tr><td><span class="get">GET</span></td><td class="m"><a href="/analytics">/analytics</a></td><td>Churn rates by contract, internet, payment</td></tr>
        <tr><td><span class="get">GET</span></td><td class="m"><a href="/feature_importance">/feature_importance</a></td><td>Feature importances (RF / XGB / ADA)</td></tr>
        <tr><td><span class="get">GET</span></td><td class="m"><a href="/top_risk">/top_risk?model=random_forest&amp;n=7</a></td><td>Top N highest-risk customers from test set</td></tr>
        <tr><td><span class="get">GET</span></td><td class="m"><a href="/mlflow_runs">/mlflow_runs</a></td><td>All MLflow experiment runs</td></tr>
        <tr><td><span class="post">POST</span></td><td class="m">/train</td><td>Train a model &nbsp;<code style="font-size:11px;color:#7a7268">{"model":"random_forest","params":{}}</code></td></tr>
        <tr><td><span class="post">POST</span></td><td class="m">/predict</td><td>Predict churn probability &nbsp;<code style="font-size:11px;color:#7a7268">{"model":"random_forest","features":{...}}</code></td></tr>
      </tbody>
    </table>
    <p class="footer">
      Frontend &rarr; <a href="http://localhost:3000" target="_blank">http://localhost:3000</a> &nbsp;·&nbsp;
      MLflow UI &rarr; <a href="http://localhost:5001" target="_blank">http://localhost:5001</a>
    </p>
  </div>
</body>
</html>"""
    return Response(html, mimetype="text/html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "models_supported": SUPPORTED_MODELS})


@app.route("/models", methods=["GET"])
def list_models():
    saved = []
    if os.path.exists(MODELS_DIR):
        saved = [f.replace(".pkl", "")
                 for f in os.listdir(MODELS_DIR)
                 if f.endswith(".pkl") and f != "scaler.pkl"]
    return jsonify({"available": SUPPORTED_MODELS, "saved": saved})


@app.route("/train", methods=["POST"])
def train():
    try:
        body      = request.get_json(force=True) or {}
        model_key = body.get("model", "random_forest")
        params    = body.get("params", {})

        if model_key not in SUPPORTED_MODELS:
            return jsonify({"error": f"Unknown model '{model_key}'."}), 400

        df = load_and_clean()
        X_train, X_test, y_train, y_test, _ = transform(df, save=True)

        model = build_model(model_key, params)

        run_name = model_key.replace("_", " ").title()
        with mlflow.start_run(run_name=f"{run_name} (UI)"):
            model.fit(X_train, y_train)

            y_pred    = model.predict(X_test)
            y_prob    = model.predict_proba(X_test)[:, 1]
            train_acc = accuracy_score(y_train, model.predict(X_train))
            cm        = confusion_matrix(y_test, y_pred).tolist()

            metrics = {
                "model":          model_key,
                "params":         params,
                "accuracy":       round(accuracy_score(y_test, y_pred), 4),
                "precision":      round(precision_score(y_test, y_pred, zero_division=0), 4),
                "recall":         round(recall_score(y_test, y_pred, zero_division=0), 4),
                "f1":             round(f1_score(y_test, y_pred, zero_division=0), 4),
                "roc_auc":        round(roc_auc_score(y_test, y_prob), 4),
                "train_accuracy": round(train_acc, 4),
                "cm":             cm,
            }

            # Log params
            mlflow.log_param("model", model_key)
            for k, v in params.items():
                mlflow.log_param(k, v)

            # Log metrics
            for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "train_accuracy"):
                mlflow.log_metric(k, metrics[k])

            # Log model artifact
            mlflow.sklearn.log_model(
                model,
                artifact_path=f"model_{model_key}",
            )

        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(model, os.path.join(MODELS_DIR, f"{model_key}.pkl"))
        _MODEL_CACHE[model_key] = model
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

        model  = _get_model(model_key)
        scaler = _get_scaler()

        if model is None:
            return jsonify({"error": f"Model '{model_key}' not trained yet. Call POST /train first."}), 404
        if scaler is None:
            return jsonify({"error": "Scaler not found. Call POST /train first."}), 404

        # Build full feature row from provided values (all numeric after encoding)
        row = {k: float(v) if v is not None else 0.0 for k, v in features.items()}
        row.setdefault("tenure",         12.0)
        row.setdefault("MonthlyCharges", 65.0)
        row.setdefault("TotalCharges",   780.0)

        df_input = pd.DataFrame([row])

        # Scale the three numeric columns
        df_input[["tenure", "MonthlyCharges", "TotalCharges"]] = scaler.transform(
            df_input[["tenure", "MonthlyCharges", "TotalCharges"]]
        )

        # Align columns to training set
        if _X_TRAIN_COLS:
            for col in _X_TRAIN_COLS:
                if col not in df_input.columns:
                    df_input[col] = 0
            df_input = df_input[_X_TRAIN_COLS]

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
    cols = _X_TRAIN_COLS or []
    for fname in ["random_forest.pkl", "xgboost.pkl", "adaboost.pkl"]:
        fp = os.path.join(MODELS_DIR, fname)
        if os.path.exists(fp):
            try:
                model = joblib.load(fp)
                imp   = dict(zip(cols, [round(float(v), 6) for v in model.feature_importances_]))
                return jsonify(dict(sorted(imp.items(), key=lambda x: -x[1])))
            except Exception:
                continue
    return jsonify({"error": "No model with feature_importances_ found."}), 404


@app.route("/stats", methods=["GET"])
def stats():
    try:
        X_train, X_test, y_train, y_test, _ = load_splits()
        n_total   = len(X_train) + len(X_test)
        n_churned = int((y_train == 1).sum() + (y_test == 1).sum())
        return jsonify({
            "total":      n_total,
            "train":      len(X_train),
            "test":       len(X_test),
            "features":   len(X_train.columns),
            "churned":    n_churned,
            "churn_rate": round(n_churned / n_total * 100, 1),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/top_risk", methods=["GET"])
def top_risk():
    model_key = request.args.get("model", "random_forest")
    n         = min(int(request.args.get("n", 7)), 20)
    try:
        model = _get_model(model_key)
        if model is None:
            return jsonify({"error": "Model not found"}), 404

        X_train, X_test, y_train, y_test, _ = load_splits()
        probs   = model.predict_proba(X_test)[:, 1]
        indices = np.argsort(probs)[::-1][:n]

        rows = []
        for idx in indices:
            p = float(probs[idx])
            rows.append({
                "id":     f"CUST-{int(X_test.index[idx]):05d}",
                "prob":   round(p * 100, 1),
                "risk":   "High" if p > 0.70 else "Medium" if p > 0.40 else "Low",
                "actual": int(y_test.iloc[idx]),
            })
        return jsonify(rows)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/analytics", methods=["GET"])
def analytics():
    try:
        from data_loader import load_raw, CSV_PATH
        df = load_raw(CSV_PATH)
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].str.strip()
        df["_churn"] = df["Churn"].str.capitalize().map({"Yes": 1, "No": 0})
        df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
        df["tenure"]         = pd.to_numeric(df["tenure"],         errors="coerce")

        def rate(s): return round(float(s.mean()) * 100, 1)

        return jsonify({
            "by_contract":           df.groupby("Contract")["_churn"].apply(rate).to_dict(),
            "by_internet":           df.groupby("InternetService")["_churn"].apply(rate).to_dict(),
            "by_payment":            df.groupby("PaymentMethod")["_churn"].apply(rate).to_dict(),
            "avg_monthly":           round(float(df["MonthlyCharges"].mean()), 2),
            "avg_monthly_churner":   round(float(df[df["_churn"]==1]["MonthlyCharges"].mean()), 2),
            "avg_tenure_churner":    round(float(df[df["_churn"]==1]["tenure"].mean()), 1),
            "avg_tenure_retained":   round(float(df[df["_churn"]==0]["tenure"].mean()), 1),
            "mtm_rate":              round(float((df["Contract"]=="Month-to-month").mean()*100), 1),
            "fiber_churn_rate":      round(float((df[df["InternetService"]=="Fiber optic"]["_churn"]).mean()*100), 1),
            "dsl_churn_rate":        round(float((df[df["InternetService"]=="DSL"]["_churn"]).mean()*100), 1),
            "no_internet_churn_rate":round(float((df[df["InternetService"]=="No"]["_churn"]).mean()*100), 1),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/mlflow_runs", methods=["GET"])
def mlflow_runs_endpoint():
    import mlflow
    from pathlib import Path
    try:
        mlflow.set_tracking_uri(Path(os.path.join(PROJECT_DIR, "mlruns")).as_uri())
        client      = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()
        runs        = []
        for exp in experiments:
            exp_runs = client.search_runs(
                [exp.experiment_id], order_by=["start_time DESC"], max_results=50
            )
            for run in exp_runs:
                p     = run.data.params
                p_str = ", ".join(f"{k}={v}" for k, v in list(p.items())[:3])
                runs.append({
                    "id":         run.info.run_id[:8],
                    "model":      run.data.tags.get("mlflow.runName", "Unknown"),
                    "experiment": exp.name.replace("churn_prediction_", ""),
                    "accuracy":   run.data.metrics.get("accuracy"),
                    "f1":         run.data.metrics.get("f1"),
                    "params":     p_str,
                    "timestamp":  run.info.start_time,
                })
        runs.sort(key=lambda r: r["timestamp"] or 0, reverse=True)
        return jsonify({"runs": runs, "total": len(runs)})
    except Exception as e:
        return jsonify({"runs": [], "total": 0}), 200


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
