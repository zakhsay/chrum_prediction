"""
train.py  -  Task 3
Trains KNN, SVM, Random Forest, Logistic Regression, AdaBoost, XGBoost.
Every run is tracked with MLflow (params, metrics, model artifact, plots).

Usage:
    cd backend
    python train.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import mlflow
import mlflow.sklearn

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    RocCurveDisplay,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import load_and_clean, transform, load_splits

# ── Paths ──────────────────────────────────────────────────────────
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BACKEND_DIR)
MODELS_DIR  = os.path.join(PROJECT_DIR, "models")
PLOTS_DIR   = os.path.join(BACKEND_DIR, "plots")
MLRUNS_DIR  = os.path.join(PROJECT_DIR, "mlruns")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

EXPERIMENT_NAME = "churn_prediction_task3"


# ── Metric helper ──────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_true, y_prob), 4),
    }


# ── Plot helpers ───────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", ax=ax,
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    path = os.path.join(PLOTS_DIR, f"cm_{model_name.lower().replace(' ', '_')}.png")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return path


def plot_roc(model, X_test, y_test, model_name):
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=model_name)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_title(f"ROC Curve — {model_name}")
    path = os.path.join(PLOTS_DIR, f"roc_{model_name.lower().replace(' ', '_')}.png")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return path


# ── Generic MLflow runner ──────────────────────────────────────────
def train_and_log(model, model_name, params, X_train, X_test, y_train, y_test):
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)
        y_pred    = model.predict(X_test)
        y_prob    = model.predict_proba(X_test)[:, 1]
        train_acc = accuracy_score(y_train, model.predict(X_train))
        metrics   = compute_metrics(y_test, y_pred, y_prob)

        for k, v in params.items():
            mlflow.log_param(k, v)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        mlflow.log_metric("train_accuracy", round(train_acc, 4))
        mlflow.sklearn.log_model(
            model, artifact_path=f"model_{model_name.lower().replace(' ', '_')}"
        )
        joblib.dump(model, os.path.join(MODELS_DIR,
                                        f"{model_name.replace(' ', '_')}.pkl"))
        mlflow.log_artifact(plot_confusion_matrix(y_test, y_pred, model_name))
        mlflow.log_artifact(plot_roc(model, X_test, y_test, model_name))

        print(f"\n{'='*55}\n  {model_name}\n{'='*55}")
        print(f"  Train Acc : {train_acc:.4f}")
        for k, v in metrics.items():
            print(f"  {k:10s}: {v:.4f}")
        print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    return model, metrics


# ── Individual model runners ───────────────────────────────────────
def run_knn(X_train, X_test, y_train, y_test):
    params = {"n_neighbors": 7, "weights": "distance", "metric": "euclidean"}
    return train_and_log(KNeighborsClassifier(**params),
                         "KNN", params, X_train, X_test, y_train, y_test)


def run_svm(X_train, X_test, y_train, y_test):
    params = {"C": 1.0, "kernel": "rbf", "gamma": "scale",
              "probability": True, "random_state": 42}
    return train_and_log(SVC(**params),
                         "SVM", params, X_train, X_test, y_train, y_test)


def run_random_forest(X_train, X_test, y_train, y_test):
    params = {"n_estimators": 100, "max_depth": 10, "min_samples_split": 5,
              "min_samples_leaf": 2, "random_state": 42, "class_weight": "balanced"}
    return train_and_log(RandomForestClassifier(**params),
                         "Random Forest", params, X_train, X_test, y_train, y_test)


def run_logistic_regression(X_train, X_test, y_train, y_test):
    params = {"C": 1.0, "solver": "lbfgs", "max_iter": 1000,
              "random_state": 42, "class_weight": "balanced"}
    return train_and_log(LogisticRegression(**params),
                         "Logistic Regression", params, X_train, X_test, y_train, y_test)


# ── NEW: AdaBoost ──────────────────────────────────────────────────
def run_adaboost(X_train, X_test, y_train, y_test):
    """
    AdaBoost with a shallow Decision Tree as base estimator.
    n_estimators : number of weak learners (trees) to combine.
    learning_rate: shrinks the contribution of each estimator.
    algorithm     : SAMME supports predict_proba natively.
    """
    base = DecisionTreeClassifier(max_depth=1, random_state=42)   # stump
    params = {
        "n_estimators":  100,
        "learning_rate": 0.5,
        "random_state":  42,
    }
    model = AdaBoostClassifier(estimator=base, **params)
    return train_and_log(model, "AdaBoost", params, X_train, X_test, y_train, y_test)


# ── NEW: XGBoost ───────────────────────────────────────────────────
def run_xgboost(X_train, X_test, y_train, y_test):
    """
    XGBoost gradient-boosted trees.
    scale_pos_weight compensates the class imbalance (neg/pos ratio).
    eval_metric='logloss' avoids deprecation warnings with MLflow.
    """
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    spw = round(neg / pos, 2)

    params = {
        "n_estimators":     200,
        "max_depth":        5,
        "learning_rate":    0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": spw,      # handles class imbalance
        "eval_metric":      "logloss",
        "random_state":     42,
    }
    model = XGBClassifier(**params)
    return train_and_log(model, "XGBoost", params, X_train, X_test, y_train, y_test)


# ── Hyperparameter sweeps ──────────────────────────────────────────
def run_adaboost_sweep(X_train, X_test, y_train, y_test):
    """Sweep learning_rate and n_estimators for AdaBoost."""
    mlflow.set_experiment(EXPERIMENT_NAME)
    print("\n[SWEEP] AdaBoost — learning_rate × n_estimators")
    for lr in [0.1, 0.5, 1.0]:
        for n in [50, 100, 200]:
            with mlflow.start_run(run_name=f"ADA_lr{lr}_n{n}"):
                base = DecisionTreeClassifier(max_depth=1, random_state=42)
                m = AdaBoostClassifier(estimator=base, n_estimators=n,
                                       learning_rate=lr, random_state=42)
                m.fit(X_train, y_train)
                y_pred = m.predict(X_test)
                y_prob = m.predict_proba(X_test)[:, 1]
                metrics = compute_metrics(y_test, y_pred, y_prob)
                mlflow.log_param("learning_rate", lr)
                mlflow.log_param("n_estimators",  n)
                for mk, mv in metrics.items():
                    mlflow.log_metric(mk, mv)
                print(f"  lr={lr} n={n:3d} -> acc={metrics['accuracy']:.4f} "
                      f"f1={metrics['f1']:.4f}")


def run_xgboost_sweep(X_train, X_test, y_train, y_test):
    """Sweep max_depth and learning_rate for XGBoost."""
    mlflow.set_experiment(EXPERIMENT_NAME)
    neg = int((y_train == 0).sum()); pos = int((y_train == 1).sum())
    spw = round(neg / pos, 2)
    print("\n[SWEEP] XGBoost — max_depth × learning_rate")
    for max_d in [3, 5, 7]:
        for lr in [0.01, 0.05, 0.1]:
            with mlflow.start_run(run_name=f"XGB_d{max_d}_lr{lr}"):
                m = XGBClassifier(n_estimators=200, max_depth=max_d,
                                   learning_rate=lr, subsample=0.8,
                                   colsample_bytree=0.8, scale_pos_weight=spw,
                                   eval_metric="logloss",
                                   random_state=42)
                m.fit(X_train, y_train)
                y_pred = m.predict(X_test)
                y_prob = m.predict_proba(X_test)[:, 1]
                metrics = compute_metrics(y_test, y_pred, y_prob)
                mlflow.log_param("max_depth",     max_d)
                mlflow.log_param("learning_rate", lr)
                for mk, mv in metrics.items():
                    mlflow.log_metric(mk, mv)
                print(f"  d={max_d} lr={lr} -> acc={metrics['accuracy']:.4f} "
                      f"f1={metrics['f1']:.4f}")


# ── Comparison chart ───────────────────────────────────────────────
def plot_comparison(results_dict):
    models  = list(results_dict.keys())
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    x       = np.arange(len(models))
    width   = 0.14
    colors  = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = [results_dict[m][metric] for m in models]
        bars = ax.bar(x + i * width, vals, width, label=metric.upper(),
                      color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xlabel("Model"); ax.set_ylabel("Score")
    ax.set_title("Task 3 — All 6 Models Comparison", fontsize=13, fontweight="bold")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(models, fontsize=9, rotation=10)
    ax.legend(fontsize=9); ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=0.3)
    path = os.path.join(PLOTS_DIR, "task3_model_comparison.png")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f"\n[plot] Saved: {path}")
    return path


def plot_all_roc(trained_models, X_test, y_test):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2196F3", "#FF9800", "#e74c3c", "#27ae60", "#9b59b6", "#1abc9c"]
    for (name, model), color in zip(trained_models, colors):
        RocCurveDisplay.from_estimator(model, X_test, y_test,
                                       ax=ax, name=name, color=color)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_title("ROC Curves — All 6 Models (Task 3)", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)
    path = os.path.join(PLOTS_DIR, "task3_roc_all.png")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f"[plot] Saved: {path}")


# ── Main ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Windows-compatible: use pathlib to build a proper file URI
    from pathlib import Path
    mlflow.set_tracking_uri(Path(MLRUNS_DIR).as_uri())

    print("=== Preprocessing ===")
    df = load_and_clean()
    X_train, X_test, y_train, y_test, scaler = transform(df, save=True)

    print("\n=== Task 3: Training all 6 models ===")
    all_results    = {}
    trained_models = []

    for runner, key in [
        (run_knn,                "KNN"),
        (run_svm,                "SVM"),
        (run_random_forest,      "Random Forest"),
        (run_logistic_regression,"Logistic Regression"),
        (run_adaboost,           "AdaBoost"),
        (run_xgboost,            "XGBoost"),
    ]:
        model, metrics = runner(X_train, X_test, y_train, y_test)
        all_results[key]    = metrics
        trained_models.append((key, model))

    print("\n=== Hyperparameter Sweeps ===")
    run_adaboost_sweep(X_train, X_test, y_train, y_test)
    run_xgboost_sweep(X_train,  X_test, y_train, y_test)

    # Plots
    plot_comparison(all_results)
    plot_all_roc(trained_models, X_test, y_test)

    # Summary
    print("\n=== FINAL COMPARISON — 6 MODELS ===")
    df_results = pd.DataFrame(all_results).T
    print(df_results.to_string())
    df_results.to_csv(os.path.join(PLOTS_DIR, "task3_comparison_table.csv"))
    print(f"\n[done] mlflow ui --backend-store-uri ../mlruns")