import pandas as pd, joblib, numpy as np

NUMERIC = [
 "BALANCE","BALANCE_FREQUENCY","PURCHASES","ONEOFF_PURCHASES",
 "INSTALLMENTS_PURCHASES","CASH_ADVANCE","PURCHASES_FREQUENCY",
 "ONEOFF_PURCHASES_FREQUENCY","PURCHASES_INSTALLMENTS_FREQUENCY",
 "CASH_ADVANCE_FREQUENCY","CASH_ADVANCE_TRX","PURCHASES_TRX",
 "CREDIT_LIMIT","PAYMENTS","MINIMUM_PAYMENTS","PRC_FULL_PAYMENT","TENURE"
]

raw = pd.read_csv("data/processed/live_raw.csv")
X_scaled = pd.read_csv("data/processed/features.csv")

imputer = joblib.load("models/imputer.joblib")
scaler = joblib.load("models/scaler.joblib")
kmeans = joblib.load("models/kmeans.joblib")

labels = kmeans.predict(X_scaled.values)
assignments = pd.DataFrame({"CUST_ID": raw.get("CUST_ID", pd.Series(range(len(raw)))), "cluster": labels})
assignments.to_csv("data/processed/cluster_assignments.csv", index=False)

# cluster profiles in original (imputed) scale
orig = imputer.transform(raw[NUMERIC])
orig = pd.DataFrame(orig, columns=NUMERIC)
profiles = pd.concat([pd.Series(labels, name="cluster"), orig], axis=1).groupby("cluster").mean().round(2)
profiles.to_csv("data/processed/cluster_profiles.csv")

print(f"[infer] wrote assignments for {len(assignments)} rows and profiles for {profiles.shape[0]} clusters")
