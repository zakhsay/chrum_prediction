"""
data_loader.py
Locates and loads the raw Telecom Churn CSV dataset.

Place the dataset at:
    backend/data/raw/Telecom_Customers_Churn.csv
"""

import os
import pandas as pd

BACKEND_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR = os.path.join(BACKEND_DIR, "data", "raw")
CSV_FILENAME = "Telecom_customer_churn.csv"
CSV_PATH     = os.path.join(DATA_RAW_DIR, CSV_FILENAME)


def load_raw(filepath: str = CSV_PATH) -> pd.DataFrame:
    """
    Load the raw CSV and return a DataFrame.
    If the file is missing, generates a synthetic dataset.
    """
    if not os.path.exists(filepath):
        print(f"[data_loader] CSV not found at {filepath}. Generating synthetic data...")
        import numpy as np
        
        np.random.seed(42)
        n_samples = 1000
        
        df = pd.DataFrame({
            "customerID": [f"CUST_{i:04d}" for i in range(n_samples)],
            "gender": np.random.choice(["Male", "Female"], size=n_samples),
            "SeniorCitizen": np.random.choice([0, 1], size=n_samples),
            "Partner": np.random.choice(["Yes", "No"], size=n_samples),
            "Dependents": np.random.choice(["Yes", "No"], size=n_samples),
            "tenure": np.random.randint(0, 73, size=n_samples),
            "PhoneService": np.random.choice(["Yes", "No"], size=n_samples),
            "MultipleLines": np.random.choice(["No phone service", "No", "Yes"], size=n_samples),
            "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], size=n_samples),
            "OnlineSecurity": np.random.choice(["No", "Yes", "No internet service"], size=n_samples),
            "OnlineBackup": np.random.choice(["No", "Yes", "No internet service"], size=n_samples),
            "DeviceProtection": np.random.choice(["No", "Yes", "No internet service"], size=n_samples),
            "TechSupport": np.random.choice(["No", "Yes", "No internet service"], size=n_samples),
            "StreamingTV": np.random.choice(["No", "Yes", "No internet service"], size=n_samples),
            "StreamingMovies": np.random.choice(["No", "Yes", "No internet service"], size=n_samples),
            "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], size=n_samples),
            "PaperlessBilling": np.random.choice(["Yes", "No"], size=n_samples),
            "PaymentMethod": np.random.choice([
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
            ], size=n_samples),
            "MonthlyCharges": np.random.uniform(18.0, 118.0, size=n_samples),
            "Churn": np.random.choice(["Yes", "No"], size=n_samples, p=[0.26, 0.74])
        })
        
        # Calculate TotalCharges based on tenure and MonthlyCharges to make it somewhat realistic
        df["TotalCharges"] = df["tenure"] * df["MonthlyCharges"]
        df["TotalCharges"] = df["TotalCharges"].astype(str) # preprocessing expects string/object
        
        print(f"[data_loader] Generated synthetic dataset with {len(df):,} rows x {len(df.columns)} columns")
        df.to_csv(filepath, index=False)
        return df
        
    df = pd.read_csv(filepath)
    print(f"[data_loader] Loaded {len(df):,} rows x {len(df.columns)} columns")
    return df


if __name__ == "__main__":
    df = load_raw()
    print(df.head())
    print("\nDtypes:\n", df.dtypes)
