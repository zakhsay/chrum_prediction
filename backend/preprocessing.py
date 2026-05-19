"""
preprocessing.py
Data cleaning, encoding, scaling, and train/test splitting
for the Telecom Customer Churn dataset.

Project structure expected:
    project_nomProjet/
        backend/
            preprocessing.py   <- this file
            data/raw/          <- raw CSV
            data/processed/    <- splits saved here
        models/                <- scaler.pkl saved here
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

from data_loader import load_raw, CSV_PATH

# ── Paths ──────────────────────────────────────────────────────────
BACKEND_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR   = os.path.dirname(BACKEND_DIR)
DATA_PROC_DIR = os.path.join(BACKEND_DIR, "data", "processed")
MODELS_DIR    = os.path.join(PROJECT_DIR, "models")


def load_and_clean(filepath: str = CSV_PATH) -> pd.DataFrame:
    """Load raw CSV and apply all cleaning steps."""
    df = load_raw(filepath)
    df_clean = df.copy()

    # Remove duplicates
    before = len(df_clean)
    df_clean.drop_duplicates(inplace=True)
    print(f"[clean] Removed {before - len(df_clean)} duplicates")

    # Strip whitespace from all string columns
    for col in df_clean.select_dtypes(include=["object"]).columns:
        df_clean[col] = df_clean[col].str.strip()

    # Fix TotalCharges: coerce non-numeric to NaN, fill with median
    df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce")
    n_missing = df_clean["TotalCharges"].isnull().sum()
    df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(
        df_clean["TotalCharges"].median()
    )
    print(f"[clean] Filled {n_missing} missing TotalCharges with median")

    # Ensure tenure is integer
    df_clean["tenure"] = df_clean["tenure"].astype(int)

    print(f"[clean] Final shape: {df_clean.shape}")
    return df_clean


def transform(df_clean: pd.DataFrame,
              test_size: float = 0.2,
              random_state: int = 42,
              save: bool = True):
    """
    Encode categorical variables, standardize numerics,
    and split into train/test sets.

    Returns: X_train, X_test, y_train, y_test, scaler
    """
    df = df_clean.copy()

    # Encode target
    df["Churn"] = df["Churn"].str.strip().str.capitalize()
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    df = df.dropna(subset=["Churn"])
    df["Churn"] = df["Churn"].astype(int)
    print(f"[transform] Churn distribution:\n{df['Churn'].value_counts()}")

    # One-hot encode categorical columns (drop customerID)
    categorical_cols = (
        df.select_dtypes(include=["object"])
        .columns.drop("customerID", errors="ignore")
    )
    df_transformed = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Standardize numeric columns
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    scaler = StandardScaler()
    df_transformed[numeric_cols] = scaler.fit_transform(df_transformed[numeric_cols])

    # Features / target split
    X = df_transformed.drop(["Churn", "customerID"], axis=1, errors="ignore")
    y = df_transformed["Churn"]

    # Train / test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    print(f"[transform] X_train: {X_train.shape}  |  X_test: {X_test.shape}")

    if save:
        os.makedirs(DATA_PROC_DIR, exist_ok=True)
        os.makedirs(MODELS_DIR, exist_ok=True)
        X_train.to_csv(os.path.join(DATA_PROC_DIR, "X_train.csv"), index=False)
        X_test.to_csv(os.path.join(DATA_PROC_DIR,  "X_test.csv"),  index=False)
        y_train.to_csv(os.path.join(DATA_PROC_DIR, "y_train.csv"), index=False)
        y_test.to_csv(os.path.join(DATA_PROC_DIR,  "y_test.csv"),  index=False)
        joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
        print(f"[transform] Splits saved  -> {DATA_PROC_DIR}")
        print(f"[transform] Scaler saved  -> {MODELS_DIR}/scaler.pkl")

    return X_train, X_test, y_train, y_test, scaler


def load_splits():
    """Load pre-saved train/test splits and scaler from disk."""
    X_train = pd.read_csv(os.path.join(DATA_PROC_DIR, "X_train.csv"))
    X_test  = pd.read_csv(os.path.join(DATA_PROC_DIR, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(DATA_PROC_DIR, "y_train.csv")).squeeze()
    y_test  = pd.read_csv(os.path.join(DATA_PROC_DIR, "y_test.csv")).squeeze()
    scaler  = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    df = load_and_clean()
    X_train, X_test, y_train, y_test, scaler = transform(df)
    print("\nPreprocessing complete.")
