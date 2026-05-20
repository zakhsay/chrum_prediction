"""
mlflow_registry.py — Tache 5 Partie 3
Gestion du Model Registry MLflow :
  - Recherche du meilleur run
  - Enregistrement dans le Registry
  - Cycle de vie : Staging → Production
  - Validation par seuil

Usage :
    cd backend
    python mlflow_registry.py
"""

import os
import sys
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

# ── Chemins ─────────────────────────────────────────────────────
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BACKEND_DIR)
MLRUNS_DIR  = os.path.join(PROJECT_DIR, "mlruns")

# URI compatible Windows ET Linux
mlflow.set_tracking_uri(Path(MLRUNS_DIR).as_uri())

EXPERIMENT_NAME  = "churn_prediction_task3"
REGISTRY_NAME    = "churn_prediction_model"
SEUIL_PRODUCTION = 0.80   # accuracy minimale pour aller en Production
SEUIL_STAGING    = 0.70   # accuracy minimale pour aller en Staging


def get_best_run():
    """Trouve le meilleur run PRINCIPAL (pas de sweep) par accuracy."""
    client = MlflowClient()
    try:
        exp = client.get_experiment_by_name(EXPERIMENT_NAME)
        if exp is None:
            print(f"[ERREUR] Experience \'{EXPERIMENT_NAME}\' introuvable.")
            print("  Lancez d abord : python train.py")
            sys.exit(1)

        all_runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["metrics.accuracy DESC"],
            max_results=200,
        )
        if not all_runs:
            print("[ERREUR] Aucun run trouve. Lancez d abord : python train.py")
            sys.exit(1)

        MAIN_NAMES = {"KNN", "SVM", "Random Forest",
                      "Logistic Regression", "AdaBoost", "XGBoost"}
        main_runs = [r for r in all_runs if r.info.run_name in MAIN_NAMES]
        if not main_runs:
            main_runs = all_runs

        best = main_runs[0]
        print(f"\n{'='*55}")
        print(f"  MEILLEUR RUN")
        print(f"{'='*55}")
        print(f"  Run ID   : {best.info.run_id}")
        print(f"  Nom      : {best.info.run_name}")
        print(f"  Accuracy : {best.data.metrics.get('accuracy', 0):.4f}")
        print(f"  F1       : {best.data.metrics.get('f1', 0):.4f}")
        print(f"  AUC      : {best.data.metrics.get('roc_auc', 0):.4f}")
        print(f"  Params   : {best.data.params}")
        return best, client

    except Exception as e:
        print(f"[ERREUR] {e}")
        sys.exit(1)


def find_best_run_with_model(client):
    """
    Cherche le meilleur run qui possede un artefact modele.
    Les runs de sweep (ADA_lr..., XGB_d...) n'ont pas de modele sauvegarde.
    On prend le meilleur run parmi les 6 modeles principaux.
    """
    try:
        exp  = client.get_experiment_by_name(EXPERIMENT_NAME)
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["metrics.accuracy DESC"],
            max_results=100,
        )
        # Noms des runs principaux (pas les sweeps)
        main_names = {"KNN", "SVM", "Random Forest",
                      "Logistic Regression", "AdaBoost", "XGBoost"}

        for r in runs:
            if r.info.run_name in main_names:
                # Verifier qu'il a des artefacts
                artifacts = client.list_artifacts(r.info.run_id)
                if artifacts:
                    print(f"\n[INFO] Run retenu pour le Registry :")
                    print(f"  Nom      : {r.info.run_name}")
                    print(f"  Run ID   : {r.info.run_id}")
                    print(f"  Accuracy : {r.data.metrics.get('accuracy', 0):.4f}")
                    return r
        return None
    except Exception as e:
        print(f"[WARN] {e}")
        return None


def register_model(client, best_run):
    """Enregistre le modele dans le Registry."""
    # D'abord essayer de trouver un run avec modele
    run_with_model = find_best_run_with_model(client)
    if run_with_model:
        run_id = run_with_model.info.run_id
    else:
        run_id = best_run.info.run_id

    # Lister tous les artefacts du run pour trouver le bon chemin
    try:
        artifacts = client.list_artifacts(run_id)
        model_artifact = None
        for a in artifacts:
            # Chercher un repertoire qui ressemble a un modele MLflow
            if a.is_dir and ("model" in a.path.lower()):
                model_artifact = a.path
                break
        if not model_artifact and artifacts:
            # Prendre le premier repertoire disponible
            for a in artifacts:
                if a.is_dir:
                    model_artifact = a.path
                    break

        if model_artifact:
            model_uri = f"runs:/{run_id}/{model_artifact}"
        else:
            # Fallback : essayer les noms standards
            model_uri = f"runs:/{run_id}/model_random_forest"

        registered = mlflow.register_model(
            model_uri=model_uri,
            name=REGISTRY_NAME,
        )
        print(f"\n[OK] Modele enregistre : '{REGISTRY_NAME}' v{registered.version}")
        print(f"     URI : {model_uri}")
        return registered

    except Exception as e:
        print(f"\n[WARN] Registry MLflow : {e}")
        print("\n[ALTERNATIVE] Enregistrement depuis le fichier .pkl local...")
        try:
            import joblib, tempfile
            # Chercher le meilleur modele .pkl
            for fname in ["Random_Forest.pkl","SVM.pkl","AdaBoost.pkl",
                          "XGBoost.pkl","Logistic_Regression.pkl","KNN.pkl"]:
                pkl_path = os.path.join(
                    os.path.dirname(BACKEND_DIR), "models", fname
                )
                if os.path.exists(pkl_path):
                    model = joblib.load(pkl_path)
                    with mlflow.start_run(run_name=f"registry_from_{fname}"):
                        mlflow.sklearn.log_model(model, artifact_path="model")
                        run_id2 = mlflow.active_run().info.run_id
                    registered = mlflow.register_model(
                        model_uri=f"runs:/{run_id2}/model",
                        name=REGISTRY_NAME,
                    )
                    print(f"[OK] Modele '{fname}' enregistre : v{registered.version}")
                    return registered
        except Exception as e2:
            print(f"[ERREUR] {e2}")
        sys.exit(1)


def add_metadata(client, registered):
    """Ajoute description et tags au modele."""
    try:
        client.update_registered_model(
            name=REGISTRY_NAME,
            description=(
                "Modele de prediction du churn client — Telecom Dataset\n"
                "Algorithmes : KNN, SVM, Random Forest, LR, AdaBoost, XGBoost\n"
                "Dataset : 7 043 clients, 30 features (apres OHE)"
            ),
        )
        client.set_model_version_tag(
            name=REGISTRY_NAME,
            version=str(registered.version),
            key="validated_by",
            value="equipe_data_mla",
        )
        client.set_model_version_tag(
            name=REGISTRY_NAME,
            version=str(registered.version),
            key="project",
            value="churn_prediction_task5",
        )
        print("[OK] Metadonnees ajoutees (description + tags)")
    except Exception as e:
        print(f"[WARN] Metadonnees non ajoutees : {e}")


def promote_model(client, registered, accuracy):
    """Fait transiter le modele vers Staging puis Production."""
    version = str(registered.version)

    if accuracy >= SEUIL_STAGING:
        try:
            client.transition_model_version_stage(
                name=REGISTRY_NAME,
                version=version,
                stage="Staging",
                archive_existing_versions=False,
            )
            print(f"\n[OK] Modele v{version} → Staging (accuracy={accuracy:.4f})")
        except Exception as e:
            print(f"[WARN] Staging : {e}")
    else:
        print(f"\n[INFO] Accuracy {accuracy:.4f} < seuil Staging {SEUIL_STAGING}")
        return

    if accuracy >= SEUIL_PRODUCTION:
        try:
            client.transition_model_version_stage(
                name=REGISTRY_NAME,
                version=version,
                stage="Production",
                archive_existing_versions=True,
            )
            print(f"[OK] Modele v{version} → Production (accuracy={accuracy:.4f})")
            print(f"\n[SERVING] Pour servir le modele en local :")
            print(f"  mlflow models serve -m 'models:/{REGISTRY_NAME}/Production' --port 1234 --no-conda")
        except Exception as e:
            print(f"[WARN] Production : {e}")
    else:
        print(f"[INFO] Accuracy {accuracy:.4f} < seuil Production {SEUIL_PRODUCTION}")
        print(f"  → Modele reste en Staging pour validation supplementaire.")


def list_all_runs():
    """Affiche un tableau de tous les runs de l'experience."""
    client = MlflowClient()
    try:
        exp  = client.get_experiment_by_name(EXPERIMENT_NAME)
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["metrics.accuracy DESC"],
        )
        print(f"\n{'='*80}")
        print(f"  TOUS LES RUNS — {EXPERIMENT_NAME}")
        print(f"{'='*80}")
        print(f"  {'Nom':<25} {'Accuracy':>10} {'F1':>8} {'AUC':>8} {'Train Acc':>10}")
        print(f"  {'-'*63}")
        for r in runs:
            m = r.data.metrics
            print(f"  {r.info.run_name:<25} "
                  f"{m.get('accuracy',0):>10.4f} "
                  f"{m.get('f1',0):>8.4f} "
                  f"{m.get('roc_auc',0):>8.4f} "
                  f"{m.get('train_accuracy',0):>10.4f}")
    except Exception as e:
        print(f"[WARN] Impossible de lister les runs : {e}")


if __name__ == "__main__":
    print("\n=== Tache 5 — Partie 3 : Model Registry ===")
    list_all_runs()
    best_run, client = get_best_run()
    registered = register_model(client, best_run)
    add_metadata(client, registered)
    accuracy = best_run.data.metrics.get("accuracy", 0)
    promote_model(client, registered, accuracy)
    print(f"\n[DONE] Registry termine. Ouvrir MLflow UI pour verifier :")
    print(f"  mlflow ui --backend-store-uri .\\mlruns --port 5001")
