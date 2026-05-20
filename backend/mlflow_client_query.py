"""
mlflow_client_query.py — Tache 5 Partie 2
Requetes programmatiques via MlflowClient.
Reponses aux questions Q4, Q5, Q6.

Usage :
    cd backend
    python mlflow_client_query.py
"""

import os
import sys
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BACKEND_DIR)
MLRUNS_DIR  = os.path.join(PROJECT_DIR, "mlruns")

mlflow.set_tracking_uri(Path(MLRUNS_DIR).as_uri())
client = MlflowClient()


def query_best_runs():
    """Tache 5 §2.3 — Identifier le meilleur run par code Python."""
    try:
        exp = client.get_experiment_by_name("churn_prediction_task3")
        if not exp:
            print("[ERREUR] Lancez d'abord : python train.py")
            return

        # Top 5 par accuracy
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["metrics.accuracy DESC"],
            max_results=6,
        )

        print("\n=== TOP 6 RUNS PAR ACCURACY ===")
        for i, r in enumerate(runs):
            m = r.data.metrics
            print(f"\n  [{i+1}] {r.info.run_name}")
            print(f"       Run ID   : {r.info.run_id}")
            print(f"       Accuracy : {m.get('accuracy', 0):.4f}")
            print(f"       F1 Score : {m.get('f1', 0):.4f}")
            print(f"       AUC      : {m.get('roc_auc', 0):.4f}")
            print(f"       Train Acc: {m.get('train_accuracy', 0):.4f}")
            print(f"       Params   : {r.data.params}")

        best = runs[0]
        print(f"\n{'='*55}")
        print(f"  MEILLEUR RUN : {best.info.run_name}")
        print(f"  Accuracy     : {best.data.metrics.get('accuracy', 0):.4f}")
        print(f"  F1           : {best.data.metrics.get('f1', 0):.4f}")
        print(f"{'='*55}")

        # Compromis accuracy / f1_score (Q4)
        print("\n=== Q4. Meilleur compromis Accuracy / F1 ===")
        best_compromise = max(
            runs,
            key=lambda r: (
                r.data.metrics.get("accuracy", 0) +
                r.data.metrics.get("f1", 0)
            ) / 2
        )
        bc_m = best_compromise.data.metrics
        print(f"  Modele : {best_compromise.info.run_name}")
        print(f"  Accuracy + F1 moyenne : "
              f"{(bc_m.get('accuracy',0)+bc_m.get('f1',0))/2:.4f}")

        # Correlation max_depth / accuracy (Q5)
        print("\n=== Q5. Correlation max_depth vs accuracy ===")
        depth_acc = []
        for r in runs:
            d = r.data.params.get("max_depth")
            a = r.data.metrics.get("accuracy", 0)
            if d is not None:
                try:
                    depth_acc.append((int(d), a))
                except ValueError:
                    pass
        if depth_acc:
            depth_acc.sort()
            for d, a in depth_acc:
                print(f"  max_depth={d:3d}  →  accuracy={a:.4f}")

    except Exception as e:
        print(f"[ERREUR] {e}")


def print_reflexion_answers():
    """Affiche les reponses aux questions de reflexion."""
    print("""
═══════════════════════════════════════════════════════════════
  REPONSES AUX QUESTIONS DE REFLEXION — Tache 5
═══════════════════════════════════════════════════════════════

─── PARTIE 1 ──────────────────────────────────────────────────
Q1. log_param vs log_metric ?
    log_param() : valeur fixe avant entrainement (ex: n_estimators=100).
                  Non numerique, categorie de configuration.
    log_metric() : valeur numerique apres evaluation (ex: accuracy=0.847).
                   Peut etre loggee a chaque epoch (avec step=).

Q2. Pourquoi nommer ses runs ?
    Sans nom explicite, MLflow genere un identifiant aleatoire difficile
    a lire (ex: 'legendary-cat-42'). Un nom lisible comme 'RF_100trees_depth10'
    permet de retrouver et comparer rapidement les runs dans l'UI.

Q3. Executer deux fois le meme script ?
    MLflow cree deux runs DISTINCTS avec le meme run_name.
    Chaque execution genere un run_id unique. L'UI affiche donc deux
    lignes avec le meme nom mais des resultats potentiellement differents.

─── PARTIE 2 ──────────────────────────────────────────────────
Q4. Meilleur compromis accuracy / f1_score ?
    → Random Forest (n=100, max_depth=10, class_weight=balanced)
    Accuracy=84.7%, F1=0.625. C'est le seul modele a combiner une haute
    accuracy ET un bon Recall sur les churners (73.3%).
    XGBoost a le meme F1 mais une accuracy inferieure (75.4%).

Q5. Parallel Coordinates — correlation max_depth / accuracy ?
    Oui : on observe une correlation positive entre max_depth et l'accuracy
    jusqu'a depth=10, puis un plateau voire une baisse (overfitting).
    Le gap train/test s'agrandit au-dela de depth=10.

Q6. MLflow vs print() pour la reproductibilite ?
    print() : ephemere, non structure, non indexable, non versionne.
    MLflow : stocke chaque metrique avec timestamp, run_id, params associes.
    Permet de retrouver n'importe quel run 6 mois plus tard, de comparer
    graphiquement et de charger le modele exact qui a produit ce resultat.

─── PARTIE 3 ──────────────────────────────────────────────────
Q7. Pourquoi separer Staging et Production ?
    Staging = zone de validation pre-production. On y execute des tests
    supplementaires (volume, edge cases, tests metier) avant d'exposer
    le modele aux vrais utilisateurs. Cela evite de degrader la prod.

Q8. Impact d'archiver une version en Production ?
    Le modele archive n'est plus accessible via 'Production'. Si le
    systeme de serving charge automatiquement la version Production,
    les predictions seront interrompues jusqu'a la promotion du suivant.
    Il faut donc promouvoir la nouvelle version AVANT d'archiver.

Q9. Registry et rollback ?
    Chaque version est conservee indefiniment dans le Registry avec son
    run_id et tous ses artefacts. Un rollback = transition de l'ancienne
    version vers 'Production' en une commande :
    client.transition_model_version_stage('nom', version='2', stage='Production')

─── PARTIE 4 ──────────────────────────────────────────────────
Q10. MLflow serving natif vs FastAPI ?
    MLflow natif : zero code, deploy instantane, format standardise.
    FastAPI : plus flexible (middleware, auth, pre/post processing custom).
    Natif = ideal pour prototype/interne. FastAPI = production enterprise.

Q11. Rechargement automatique d'un nouveau modele ?
    Strategie 1 : polling periodique du Registry (verifier si version change).
    Strategie 2 : webhook/callback MLflow vers le serveur.
    Strategie 3 : Kubernetes + readiness probe qui recharge au restart.
    Le serving natif MLflow necessite un redemarrage manuel.

Q12. Headers de securite pour l'endpoint ?
    - Authorization: Bearer <token>  (JWT ou API key)
    - X-API-Key: <cle>
    - Content-Security-Policy, CORS restreint
    - Rate limiting (X-RateLimit-Limit)
    - HTTPS obligatoire (TLS)

─── PARTIE 6 ──────────────────────────────────────────────────
Q13. Data drift vs Concept drift ?
    Data drift   : P(X) change. Ex: les clients telecom passent de 25-40 ans
                   a 50-70 ans. Les features d'entree changent de distribution.
    Concept drift: P(Y|X) change. Ex: un client avec contrat mensuel qui
                   ne churnait pas avant, churne maintenant (crise tarifaire).

Q14. KS-test vs Evidently — memes features ?
    Pas toujours. KS-test est non-parametrique et univarie (feature par feature).
    Evidently utilise une combinaison de tests (Jensen-Shannon, Wasserstein,
    PSI...) selon le type de feature. Ils peuvent diverger sur des distributions
    multimodales ou des features fortement correlees.

Q15. Quel seuil de drift pour ce projet ?
    Seuil retenu : 30% de features driftees → re-entrainement.
    Justification : le churn telecom evolue lentement (contrats pluriannuels).
    Un seuil de 15% declencherait trop de re-entrainements inutiles.
    30% = compromis entre reactivite et stabilite operationnelle.

Q16. Risques sans pipeline MLOps ?
    Sans detection automatique : le modele se degrade silencieusement.
    Les metriques en production baissent sans alerte. Impact metier direct :
    churners non detectes = perte de revenus. Un modele mal maintenu peut
    etre pire qu'une simple regle metier. Le drift passe inapercu pendant
    des semaines ou des mois en mode manuel.
═══════════════════════════════════════════════════════════════
""")


if __name__ == "__main__":
    print("=== Tache 5 — Partie 2 : Requetes MlflowClient ===")
    query_best_runs()
    print_reflexion_answers()
