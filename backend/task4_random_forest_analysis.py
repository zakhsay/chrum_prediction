"""
task4_random_forest_analysis.py  -  Task 4
Deep analysis of Random Forest on the Churn dataset.
 
Questions answered:
  1. Feature importance (feature_importances_)
  2. Prediction stability across random seeds
  3. Error analysis — False Negatives & False Positives
  4. Bias-Variance tradeoff (n_estimators x max_depth grid)
  5. Comparison with Decision Tree
 
Usage:
    cd backend
    python task4_random_forest_analysis.py
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
 
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
 
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import load_and_clean, transform
 
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BACKEND_DIR)
MODELS_DIR  = os.path.join(PROJECT_DIR, "models")
PLOTS_DIR   = os.path.join(BACKEND_DIR, "plots")
MLRUNS_DIR  = os.path.join(PROJECT_DIR, "mlruns")
 
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
 
from pathlib import Path
mlflow.set_tracking_uri(Path(MLRUNS_DIR).as_uri())
mlflow.set_experiment("churn_task4_random_forest")
 
# ── Load data ──────────────────────────────────────────────────────
print("=== Task 4: Loading data ===")
df = load_and_clean()
X_train, X_test, y_train, y_test, scaler = transform(df, save=False)
 
 
# ══════════════════════════════════════════════════════════════════
# 1. FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════
print("\n=== 1. Feature Importance ===")
with mlflow.start_run(run_name="RF_feature_importance"):
    rf = RandomForestClassifier(n_estimators=200, max_depth=12,
                                min_samples_leaf=2, random_state=42,
                                class_weight="balanced")
    rf.fit(X_train, y_train)
 
    importances = pd.Series(rf.feature_importances_,
                            index=X_train.columns).sort_values(ascending=False)
    top10 = importances.head(10)
 
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#c0392b" if i < 3 else "#2980b9" for i in range(len(top10))]
    top10[::-1].plot(kind="barh", ax=ax, color=colors[::-1], edgecolor="none")
    ax.set_title("Top 10 Feature Importances — Random Forest (Task 4)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.axvline(importances.mean(), color="gray", linestyle="--", lw=1,
               label=f"Mean ({importances.mean():.4f})")
    for i, (val, lbl) in enumerate(zip(top10.values[::-1], top10.index[::-1])):
        ax.text(val + 0.001, i, f"{val:.4f}", va="center", fontsize=9)
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "task4_feature_importance.png")
    plt.savefig(path, dpi=150); plt.close()
    mlflow.log_artifact(path)
    mlflow.log_metric("train_accuracy", accuracy_score(y_train, rf.predict(X_train)))
    mlflow.log_metric("test_accuracy",  accuracy_score(y_test,  rf.predict(X_test)))
    mlflow.log_metric("f1_score",       f1_score(y_test, rf.predict(X_test)))
    joblib.dump(rf, os.path.join(MODELS_DIR, "rf_best.pkl"))
 
print(f"Top 3 features: {list(top10.head(3).index)}")
 
 
# ══════════════════════════════════════════════════════════════════
# 2. PREDICTION STABILITY
# ══════════════════════════════════════════════════════════════════
print("\n=== 2. Prediction Stability ===")
seeds    = [0, 7, 13, 21, 42, 77, 99, 123, 200, 314]
acc_vals = []
f1_vals  = []
 
for seed in seeds:
    m = RandomForestClassifier(n_estimators=100, max_depth=10,
                               random_state=seed, class_weight="balanced")
    m.fit(X_train, y_train)
    p = m.predict(X_test)
    acc_vals.append(accuracy_score(y_test, p))
    f1_vals.append(f1_score(y_test, p))
 
with mlflow.start_run(run_name="RF_stability"):
    mlflow.log_metric("acc_mean", np.mean(acc_vals))
    mlflow.log_metric("acc_std",  np.std(acc_vals))
    mlflow.log_metric("f1_mean",  np.mean(f1_vals))
    mlflow.log_metric("f1_std",   np.std(f1_vals))
 
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, vals, color, title in zip(axes,
    [acc_vals, f1_vals], ["#2980b9", "#27ae60"],
    ["Accuracy across random seeds", "F1 Score across random seeds"]):
    ax.plot(seeds, vals, "o-", color=color, lw=2)
    ax.axhline(np.mean(vals), color="#c0392b", ls="--",
               label=f"Mean={np.mean(vals):.4f}")
    ax.fill_between(seeds,
                    [np.mean(vals) - np.std(vals)] * len(seeds),
                    [np.mean(vals) + np.std(vals)] * len(seeds),
                    alpha=0.15, color="#c0392b",
                    label=f"+-std ({np.std(vals):.4f})")
    ax.set_title(title); ax.set_xlabel("random_state")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
 
plt.suptitle("Random Forest — Prediction Stability (Task 4)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
path = os.path.join(PLOTS_DIR, "task4_stability.png")
plt.savefig(path, dpi=150); plt.close()
print(f"  Accuracy: mean={np.mean(acc_vals):.4f}  std={np.std(acc_vals):.4f}")
print(f"  F1 Score: mean={np.mean(f1_vals):.4f}  std={np.std(f1_vals):.4f}")
 
 
# ══════════════════════════════════════════════════════════════════
# 3. ERROR ANALYSIS
# ══════════════════════════════════════════════════════════════════
print("\n=== 3. Error Analysis ===")
rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)[:, 1]
X_tr    = X_test.reset_index(drop=True)
y_tr    = y_test.reset_index(drop=True)
 
errors   = X_tr[rf_pred != y_tr.values].copy()
errors["true_label"] = y_tr[rf_pred != y_tr.values].values
errors["predicted"]  = rf_pred[rf_pred != y_tr.values]
errors["churn_prob"] = rf_prob[rf_pred != y_tr.values]
 
fn = errors[errors["true_label"] == 1]   # False Negatives
fp = errors[errors["true_label"] == 0]   # False Positives
 
print(f"  Total errors     : {len(errors)} / {len(y_tr)} ({len(errors)/len(y_tr)*100:.1f}%)")
print(f"  False Negatives  : {len(fn)}")
print(f"  False Positives  : {len(fp)}")
if len(fn) > 0:
    print("\n  Sample False Negatives (churners missed by the model):")
    cols = [c for c in ["tenure", "MonthlyCharges", "TotalCharges"] if c in fn.columns]
    print(fn[cols + ["churn_prob"]].head(3).to_string())
 
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, data, color, title in zip(axes, [fn, fp],
    ["#e8920e", "#9b59b6"],
    ["False Negatives — P(Churn)", "False Positives — P(Churn)"]):
    if len(data) > 0:
        ax.hist(data["churn_prob"], bins=20, color=color,
                edgecolor="white", alpha=0.85)
    ax.axvline(0.5, color="#c0392b", ls="--", label="Threshold 0.5")
    ax.set_title(title); ax.set_xlabel("Predicted P(Churn)")
    ax.set_ylabel("Count"); ax.legend(); ax.grid(alpha=0.3)
 
plt.suptitle("Error Analysis — Random Forest (Task 4)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
path = os.path.join(PLOTS_DIR, "task4_error_analysis.png")
plt.savefig(path, dpi=150); plt.close()
 
 
# ══════════════════════════════════════════════════════════════════
# 4. BIAS-VARIANCE TRADEOFF
# ══════════════════════════════════════════════════════════════════
print("\n=== 4. Bias-Variance Analysis ===")
param_grid = [
    (10, 2),  (10, 5),  (10, 10),  (10, None),
    (50, 2),  (50, 5),  (50, 10),  (50, None),
    (100, 2), (100, 5), (100, 10), (100, None),
    (200, 2), (200, 5), (200, 10), (200, None),
]
bv_rows = []
for n_est, max_d in param_grid:
    with mlflow.start_run(run_name=f"RF_bv_n{n_est}_d{max_d}"):
        m = RandomForestClassifier(n_estimators=n_est, max_depth=max_d,
                                   random_state=42, class_weight="balanced")
        m.fit(X_train, y_train)
        tr_acc = accuracy_score(y_train, m.predict(X_train))
        te_acc = accuracy_score(y_test,  m.predict(X_test))
        bias   = round(1 - tr_acc, 4)
        var    = round(tr_acc - te_acc, 4)
        status = ("Overfitting"  if var  > 0.08 else
                  "Underfitting" if bias > 0.15 else "Balanced")
        mlflow.log_params({"n_estimators": n_est, "max_depth": str(max_d)})
        mlflow.log_metrics({"train_accuracy": round(tr_acc, 4),
                             "test_accuracy": round(te_acc, 4),
                             "bias": bias, "variance": var})
        bv_rows.append({"n_estimators": n_est, "max_depth": str(max_d),
                         "Train Acc": round(tr_acc, 4),
                         "Test Acc": round(te_acc, 4),
                         "Bias": bias, "Variance": var, "Status": status})
        print(f"  n={n_est:3d} d={str(max_d):4s} "
              f"train={tr_acc:.4f} test={te_acc:.4f} "
              f"bias={bias:.4f} var={var:.4f} -> {status}")
 
bv_df = pd.DataFrame(bv_rows)
bv_df.to_csv(os.path.join(PLOTS_DIR, "task4_bias_variance_table.csv"), index=False)
print(f"\n  Saved bias-variance table.")
 
# Heatmaps
depths = ["2", "5", "10", "None"]
n_ests = [10, 50, 100, 200]
 
def make_grid(col):
    g = np.zeros((len(n_ests), len(depths)))
    for i, n in enumerate(n_ests):
        for j, d in enumerate(depths):
            row = bv_df[(bv_df["n_estimators"] == n) & (bv_df["max_depth"] == d)]
            if not row.empty:
                g[i, j] = row[col].values[0]
    return g
 
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, col, title, cmap in zip(axes,
    ["Train Acc", "Test Acc"],
    ["Train Accuracy", "Test Accuracy"],
    ["Reds", "Blues"]):
    g  = make_grid(col)
    im = ax.imshow(g, cmap=cmap, aspect="auto", vmin=0.6, vmax=1.0)
    ax.set_xticks(range(len(depths))); ax.set_xticklabels(depths)
    ax.set_yticks(range(len(n_ests))); ax.set_yticklabels(n_ests)
    ax.set_xlabel("max_depth"); ax.set_ylabel("n_estimators")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    for ii in range(len(n_ests)):
        for jj in range(len(depths)):
            ax.text(jj, ii, f"{g[ii,jj]:.3f}", ha="center", va="center",
                    fontsize=8, color="white" if g[ii, jj] > 0.87 else "black")
 
plt.suptitle("Bias-Variance Heatmaps (Task 4)", fontsize=12, fontweight="bold")
plt.tight_layout()
path = os.path.join(PLOTS_DIR, "task4_bias_variance_heatmap.png")
plt.savefig(path, dpi=150); plt.close()
 
 
# ══════════════════════════════════════════════════════════════════
# 5. RANDOM FOREST vs DECISION TREE
# ══════════════════════════════════════════════════════════════════
print("\n=== 5. Random Forest vs Decision Tree ===")
depths_cmp   = [3, 5, 10, None]
depth_labels = [f"d={d}" for d in depths_cmp]
dt_accs, dt_f1s, rf_accs, rf_f1s = [], [], [], []
 
for d in depths_cmp:
    dt = DecisionTreeClassifier(max_depth=d, random_state=42, class_weight="balanced")
    dt.fit(X_train, y_train); p = dt.predict(X_test)
    dt_accs.append(accuracy_score(y_test, p)); dt_f1s.append(f1_score(y_test, p))
 
    rm = RandomForestClassifier(n_estimators=100, max_depth=d,
                                random_state=42, class_weight="balanced")
    rm.fit(X_train, y_train); p2 = rm.predict(X_test)
    rf_accs.append(accuracy_score(y_test, p2)); rf_f1s.append(f1_score(y_test, p2))
    print(f"  {str(d):4s}: DT acc={dt_accs[-1]:.4f} f1={dt_f1s[-1]:.4f} | "
          f"RF acc={rf_accs[-1]:.4f} f1={rf_f1s[-1]:.4f}")
 
x = np.arange(len(depth_labels)); w = 0.35
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, dt_v, rf_v, ylabel in zip(axes, [dt_accs, dt_f1s], [rf_accs, rf_f1s],
                                   ["Test Accuracy", "F1 Score"]):
    ax.bar(x - w/2, dt_v, w, label="Decision Tree", color="#e8920e", alpha=0.85)
    ax.bar(x + w/2, rf_v, w, label="Random Forest", color="#2980b9", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(depth_labels)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel}: DT vs RF")
    ax.legend(); ax.grid(axis="y", alpha=0.3); ax.set_ylim(0.5, 1.0)
    for i in range(len(depth_labels)):
        ax.text(i - w/2, dt_v[i] + 0.005, f"{dt_v[i]:.3f}", ha="center", fontsize=8)
        ax.text(i + w/2, rf_v[i] + 0.005, f"{rf_v[i]:.3f}", ha="center", fontsize=8)
 
plt.suptitle("Decision Tree vs Random Forest (Task 4)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
path = os.path.join(PLOTS_DIR, "task4_dt_vs_rf.png")
plt.savefig(path, dpi=150); plt.close()
 
 
# ── Final summary ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TASK 4 COMPLETE")
print("=" * 60)
plots = [
    "task4_feature_importance.png",
    "task4_stability.png",
    "task4_error_analysis.png",
    "task4_bias_variance_heatmap.png",
    "task4_bias_variance_table.csv",
    "task4_dt_vs_rf.png",
]
for p in plots:
    print(f"  Saved: backend/plots/{p}")
 
print(f"\n  Best feature  : {importances.index[0]} ({importances.iloc[0]:.4f})")
print(f"  Stability std : {np.std(acc_vals):.4f}"
      f"  -> {'Stable' if np.std(acc_vals) < 0.005 else 'Variable'}")
print(f"  Errors        : {len(errors)} / {len(y_tr)} "
      f"(FN={len(fn)}, FP={len(fp)})")