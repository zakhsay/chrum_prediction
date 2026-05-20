"""
detect_drift.py — Tache 5 Partie 6
Detection du Data Drift avec Evidently + KS-test.
Declenchement automatique du re-entrainement si drift detecte.

Installation :
    pip install evidently scipy

Usage :
    cd backend
    python detect_drift.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import mlflow

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import load_and_clean, transform

# ── Chemins ─────────────────────────────────────────────────────
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BACKEND_DIR)
MLRUNS_DIR  = os.path.join(PROJECT_DIR, "mlruns")
PLOTS_DIR   = os.path.join(BACKEND_DIR, "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)

# URI compatible Windows ET Linux
mlflow.set_tracking_uri(Path(MLRUNS_DIR).as_uri())

# ── Seuils ───────────────────────────────────────────────────────
SEUIL_DRIFT = 0.30   # > 30% features driftees → re-entrainement
SEUIL_WARN  = 0.15   # > 15% → alerte surveillance

EXPERIMENT_DRIFT = "monitoring_drift"


# ═══════════════════════════════════════════════════════════════
# SIMULATION DU DRIFT
# ═══════════════════════════════════════════════════════════════
def simulate_drift(X_test: pd.DataFrame, drift_factor: float = 1.6) -> pd.DataFrame:
    """
    Simule un data drift sur X_prod en decalant les features numeriques.
    Tache 5 §6.2 : X_prod[col] = X_prod[col] * drift_factor + bruit
    """
    X_prod    = X_test.copy()
    num_cols  = X_prod.select_dtypes(include=np.number).columns
    # Appliquer le drift sur les 3 premieres features numeriques
    for col in num_cols[:3]:
        X_prod[col] = (
            X_prod[col] * drift_factor
            + np.random.normal(0, 0.5, len(X_prod))
        )
    print(f"\n[drift] Simulation appliquee (facteur {drift_factor}) sur {min(3,len(num_cols))} features")
    for col in num_cols[:3]:
        print(f"  {col:30s}: ref_mean={X_test[col].mean():.4f}  "
              f"prod_mean={X_prod[col].mean():.4f}")
    return X_prod


# ═══════════════════════════════════════════════════════════════
# KS-TEST FEATURE PAR FEATURE
# ═══════════════════════════════════════════════════════════════
def run_ks_test(X_ref: pd.DataFrame, X_prod: pd.DataFrame,
                client_mlflow) -> pd.DataFrame:
    """
    Tache 5 §6.4 — Test de Kolmogorov-Smirnov feature par feature.
    p-value < 0.05 → feature consideree comme driftee.
    """
    from scipy import stats

    results = []
    num_cols = X_ref.select_dtypes(include="number").columns

    for col in num_cols:
        stat, pvalue = stats.ks_2samp(X_ref[col], X_prod[col])
        drifted = pvalue < 0.05
        results.append({
            "feature":  col,
            "ks_stat":  round(stat,   4),
            "p_value":  round(pvalue, 4),
            "drifted":  drifted,
        })
        try:
            mlflow.log_metric(f"ks_pvalue_{col[:20]}", pvalue)
        except Exception:
            pass

    df_drift = pd.DataFrame(results)
    n_drifted = df_drift["drifted"].sum()
    n_total   = len(df_drift)
    ks_share  = n_drifted / n_total if n_total > 0 else 0

    # Sauvegarder le CSV
    csv_path = os.path.join(PLOTS_DIR, "ks_drift_results.csv")
    df_drift.to_csv(csv_path, index=False)
    mlflow.log_artifact(csv_path)

    print(f"\n{'='*60}")
    print(f"  KS-TEST — RESULTATS PAR FEATURE")
    print(f"{'='*60}")
    print(f"  {'Feature':<30} {'KS stat':>8} {'p-value':>8} {'Drifted':>8}")
    print(f"  {'-'*56}")
    for _, row in df_drift.iterrows():
        flag = " ← DRIFT" if row["drifted"] else ""
        print(f"  {row['feature']:<30} {row['ks_stat']:>8.4f} "
              f"{row['p_value']:>8.4f} {str(row['drifted']):>8}{flag}")
    print(f"\n  Colonnes driftees : {n_drifted}/{n_total} ({ks_share:.1%})")

    return df_drift, ks_share


# ═══════════════════════════════════════════════════════════════
# RAPPORT EVIDENTLY
# ═══════════════════════════════════════════════════════════════
def run_evidently(X_ref: pd.DataFrame, X_prod: pd.DataFrame) -> float:
    """
    Tache 5 §6.3 — Rapport HTML Evidently + extraction drift_share.
    """
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset, DataQualityPreset
        from evidently.metrics import DatasetDriftMetric
    except ImportError:
        print("\n[WARN] Evidently non installe.")
        print("  → pip install evidently")
        print("  → KS-test utilise a la place.")
        return -1.0

    # Rapport HTML complet
    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    report.run(reference_data=X_ref, current_data=X_prod)
    html_path = os.path.join(PLOTS_DIR, "drift_report.html")
    report.save_html(html_path)
    mlflow.log_artifact(html_path)
    print(f"\n[OK] Rapport Evidently HTML sauvegarde : {html_path}")

    # Extraction du score numerique
    score_report = Report(metrics=[DatasetDriftMetric()])
    score_report.run(reference_data=X_ref, current_data=X_prod)
    result = score_report.as_dict()

    drift_res    = result["metrics"][0]["result"]
    drift_share  = drift_res.get("drift_share",              0)
    n_drifted    = drift_res.get("number_of_drifted_columns", 0)
    n_total      = drift_res.get("number_of_columns",         0)
    dataset_drift= drift_res.get("dataset_drift",             False)

    mlflow.log_metric("drift_share",     drift_share)
    mlflow.log_metric("drifted_columns", n_drifted)
    mlflow.log_metric("total_columns",   n_total)
    mlflow.log_metric("dataset_drifted", int(dataset_drift))

    print(f"  Drift share  : {drift_share:.2%}")
    print(f"  Colonnes     : {n_drifted}/{n_total}")
    print(f"  Dataset drift: {dataset_drift}")
    return drift_share


# ═══════════════════════════════════════════════════════════════
# LOGIQUE DE RE-ENTRAINEMENT
# ═══════════════════════════════════════════════════════════════
def trigger_retrain(drift_share: float):
    """
    Tache 5 §6.5 — Declenchement automatique du re-entrainement.
    """
    import subprocess

    print(f"\n{'='*55}")
    print(f"  LOGIQUE DE RE-ENTRAINEMENT")
    print(f"  Drift share = {drift_share:.2%}")
    print(f"  Seuil critique = {SEUIL_DRIFT:.0%}  |  Seuil alerte = {SEUIL_WARN:.0%}")
    print(f"{'='*55}")

    if drift_share > SEUIL_DRIFT:
        print(f"\n  [CRITIQUE] drift {drift_share:.2%} > {SEUIL_DRIFT:.0%}")
        print(f"  → Declenchement du re-entrainement...")
        mlflow.log_metric("retrain_triggered", 1)
        train_script = os.path.join(BACKEND_DIR, "train.py")
        try:
            subprocess.run([sys.executable, train_script], check=True)
            print("  [OK] Re-entrainement termine avec succes.")
        except subprocess.CalledProcessError as e:
            print(f"  [ERREUR] Re-entrainement echoue : {e}")

    elif drift_share > SEUIL_WARN:
        print(f"\n  [AVERTISSEMENT] drift {drift_share:.2%} > {SEUIL_WARN:.0%}")
        print(f"  → Surveillance renforcee — re-entrainement non declenche.")
        mlflow.log_metric("retrain_triggered", 0)

    else:
        print(f"\n  [OK] drift {drift_share:.2%} — modele stable.")
        print(f"  → Aucune action requise.")
        mlflow.log_metric("retrain_triggered", 0)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n=== Tache 5 — Partie 6 : Detection du Data Drift ===")

    # Chargement et preparation des donnees
    print("\n[1] Chargement des donnees...")
    df = load_and_clean()
    X_train, X_test, y_train, y_test, _ = transform(df, save=False)

    # Simulation du drift sur X_prod
    print("\n[2] Simulation du drift...")
    X_prod = simulate_drift(X_test, drift_factor=1.6)

    mlflow.set_experiment(EXPERIMENT_DRIFT)
    with mlflow.start_run(run_name="drift_check_v1"):

        # Logging contexte
        mlflow.log_param("drift_factor",    1.6)
        mlflow.log_param("seuil_drift",     SEUIL_DRIFT)
        mlflow.log_param("seuil_warn",      SEUIL_WARN)
        mlflow.log_param("n_ref_samples",   len(X_train))
        mlflow.log_param("n_prod_samples",  len(X_prod))
        mlflow.log_param("n_features",      X_train.shape[1])

        # Rapport Evidently
        print("\n[3] Rapport Evidently...")
        drift_share_ev = run_evidently(X_train, X_prod)

        # KS-test
        print("\n[4] KS-test par feature...")
        df_ks, drift_share_ks = run_ks_test(X_train, X_prod, None)

        # Utiliser Evidently si disponible, sinon KS-test
        drift_share = drift_share_ev if drift_share_ev >= 0 else drift_share_ks
        mlflow.log_metric("drift_share_final", drift_share)

        # Logique de re-entrainement
        trigger_retrain(drift_share)

    print(f"\n[DONE] Rapport complet :")
    print(f"  Plots : {PLOTS_DIR}")
    print(f"  MLflow UI : mlflow ui --backend-store-uri .\\mlruns --port 5001")
