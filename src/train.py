import os
import numpy as np
import pandas as pd
import joblib

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

DEFAULT_K = int(os.getenv("K_CLUSTERS", 6))
AUTO_SELECT = os.getenv("AUTO_SELECT_K", "true").lower() in {"1","true","yes"}

X = pd.read_csv("data/processed/features.csv").values

def choose_best_k(X, k_min=3, k_max=10, random_state=42):
    best_k, best_score = None, -1
    metrics = []
    for k in range(k_min, k_max+1):
        km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels)
        metrics.append((k, score))
        if score > best_score:
            best_k, best_score = k, score
    pd.DataFrame(metrics, columns=["k","silhouette"]).to_csv("data/processed/k_metrics.csv", index=False)
    return best_k

if AUTO_SELECT:
    K = choose_best_k(X)
    print(f"[train] Auto-selected K={K}")
else:
    K = DEFAULT_K
    print(f"[train] Using fixed K={K}")

kmeans = KMeans(n_clusters=K, n_init="auto", random_state=42)
kmeans.fit(X)
joblib.dump(kmeans, "models/kmeans.joblib")

pca = PCA(n_components=2, random_state=42)
pca.fit(X)
joblib.dump(pca, "models/pca.joblib")

print("[train] Saved models/kmeans.joblib and models/pca.joblib")
