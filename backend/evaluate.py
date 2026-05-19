"""
evaluate.py
Loads all saved models and prints full evaluation metrics.
Generates a combined ROC curve and saves a CSV summary.

Usage:
    cd backend
    python evaluate.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, RocCurveDisplay,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import load_splits

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BACKEND_DIR)
MODELS_DIR  = os.path.join(PROJECT_DIR, "models")
PLOTS_DIR   = os.path.join(BACKEND_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

MODEL_FILES = {
    "KNN":                 "KNN.pkl",
    "SVM":                 "SVM.pkl",
    "Random Forest":       "Random_Forest.pkl",
    "Logistic Regression": "Logistic_Regression.pkl",
}


def evaluate_all():
    X_train, X_test, y_train, y_test, _ = load_splits()
    results = {}
    loaded_models = []

    print("=" * 65)
    print("  FULL MODEL EVALUATION REPORT")
    print("=" * 65)

    for name, fname in MODEL_FILES.items():
        path = os.path.join(MODELS_DIR, fname)
        if not os.path.exists(path):
            print(f"\n  [skip] {name} — not found at {path}")
            print(f"         Run train.py first.")
            continue

        model     = joblib.load(path)
        y_pred    = model.predict(X_test)
        y_prob    = model.predict_proba(X_test)[:, 1]
        train_acc = accuracy_score(y_train, model.predict(X_train))

        metrics = {
            "accuracy":  round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
            "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
            "train_acc": round(train_acc, 4),
        }
        results[name] = metrics
        loaded_models.append((name, model))

        print(f"\n  --- {name} ---")
        for k, v in metrics.items():
            print(f"    {k:15s}: {v:.4f}")
        print(classification_report(y_test, y_pred,
                                    target_names=["No Churn", "Churn"]))

    if not results:
        print("\n  No models found. Run train.py first.")
        return {}

    # Summary table
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    df = pd.DataFrame(results).T
    print(df.to_string())
    csv_path = os.path.join(PLOTS_DIR, "evaluation_summary.csv")
    df.to_csv(csv_path)
    print(f"\n  Saved: {csv_path}")

    # Combined ROC plot
    if loaded_models:
        colors = ["#2980b9", "#e8920e", "#e74c3c", "#27ae60"]
        fig, ax = plt.subplots(figsize=(8, 6))
        for (name, model), color in zip(loaded_models, colors):
            RocCurveDisplay.from_estimator(model, X_test, y_test,
                                           ax=ax, name=name, color=color)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_title("ROC Curves — All Models", fontsize=13, fontweight="bold")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        roc_path = os.path.join(PLOTS_DIR, "evaluate_roc_all.png")
        plt.savefig(roc_path, dpi=150)
        plt.close()
        print(f"  Saved: {roc_path}")

    return results


if __name__ == "__main__":
    evaluate_all()
