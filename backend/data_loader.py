"""
data_loader.py
Charge le dataset Telecom Customers Churn.

IMPORTANT — Placer le vrai CSV ici AVANT de lancer train.py :
    C:\chrum_project\chrum_prediction\backend\data\raw\Telecom_Customers_Churn.csv

Telecharger sur Kaggle :
    https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets
"""

import os
import sys
import pandas as pd

BACKEND_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR = os.path.join(BACKEND_DIR, "data", "raw")
CSV_FILENAME = "Telecom_customer_churn.csv"
CSV_PATH     = os.path.join(DATA_RAW_DIR, CSV_FILENAME)


def load_raw(filepath=CSV_PATH):
    """Charge le vrai CSV. Arrete si manquant — PAS de donnees synthetiques."""
    if not os.path.exists(filepath):
        print("\n" + "="*65)
        print("  ERREUR : fichier CSV introuvable !")
        print("="*65)
        print(f"\n  Chemin attendu :")
        print(f"  {filepath}")
        print()
        print("  SOLUTION EN 2 ETAPES :")
        print()
        print("  1. Telecharger le dataset Kaggle :")
        print("     https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets")
        print()
        print("  2. Copier le fichier dans :")
        print(f"     {DATA_RAW_DIR}")
        print()
        print("  Le fichier doit s'appeler exactement :")
        print(f"     {CSV_FILENAME}")
        print("="*65 + "\n")
        sys.exit(1)

    df = pd.read_csv(filepath)
    n_rows = len(df)

    if n_rows < 5000:
        print(f"\n[AVERTISSEMENT] Seulement {n_rows} lignes chargees.")
        print(f"  Le vrai dataset en a 7 043.")
        print(f"  Verifiez que vous avez telecharge le bon fichier Kaggle.\n")

    print(f"[data_loader] Dataset charge : {n_rows:,} lignes x {len(df.columns)} colonnes")
    return df


if __name__ == "__main__":
    df = load_raw()
    print(df.head())
    print(f"\nColonnes : {list(df.columns)}")
