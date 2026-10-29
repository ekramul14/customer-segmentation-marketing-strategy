import pandas as pd, joblib, numpy as np

from segments import name_segments, SEGMENT_FILE

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

# data-driven segment names (cluster IDs are arbitrary, so never hard-code names by ID)
sizes = pd.Series(labels).value_counts().sort_index()
purchase_share = orig.groupby(labels)["PURCHASES"].sum() / orig["PURCHASES"].sum() * 100
segments = name_segments(profiles, sizes, purchase_share, orig.mean())
segments.to_csv(SEGMENT_FILE, index=False)

print(f"[infer] wrote assignments for {len(assignments)} rows and profiles for {profiles.shape[0]} clusters")
print(segments[["cluster", "name", "pct_customers", "pct_purchases"]].to_string(index=False))
