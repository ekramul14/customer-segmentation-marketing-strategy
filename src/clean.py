import os
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

NUMERIC = [
 "BALANCE","BALANCE_FREQUENCY","PURCHASES","ONEOFF_PURCHASES",
 "INSTALLMENTS_PURCHASES","CASH_ADVANCE","PURCHASES_FREQUENCY",
 "ONEOFF_PURCHASES_FREQUENCY","PURCHASES_INSTALLMENTS_FREQUENCY",
 "CASH_ADVANCE_FREQUENCY","CASH_ADVANCE_TRX","PURCHASES_TRX",
 "CREDIT_LIMIT","PAYMENTS","MINIMUM_PAYMENTS","PRC_FULL_PAYMENT","TENURE"
]

def load_raw():
    return pd.read_csv("data/processed/live_raw.csv")

def clean(df: pd.DataFrame):
    X = df[NUMERIC].copy()
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)
    os.makedirs("models", exist_ok=True)
    joblib.dump(imputer, "models/imputer.joblib")
    joblib.dump(scaler, "models/scaler.joblib")
    return pd.DataFrame(X_scaled, columns=NUMERIC)

if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)
    df = load_raw()
    X = clean(df)
    X.to_csv("data/processed/features.csv", index=False)
    print(f"[clean] features.shape={X.shape}")
