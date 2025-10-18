from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
import io
import os

NUMERIC = [
 "BALANCE","BALANCE_FREQUENCY","PURCHASES","ONEOFF_PURCHASES",
 "INSTALLMENTS_PURCHASES","CASH_ADVANCE","PURCHASES_FREQUENCY",
 "ONEOFF_PURCHASES_FREQUENCY","PURCHASES_INSTALLMENTS_FREQUENCY",
 "CASH_ADVANCE_FREQUENCY","CASH_ADVANCE_TRX","PURCHASES_TRX",
 "CREDIT_LIMIT","PAYMENTS","MINIMUM_PAYMENTS","PRC_FULL_PAYMENT","TENURE"
]

# Cluster business descriptions
CLUSTER_DESCRIPTIONS = {
    0: {"name": "Transactors", "description": "Low balance, low cash advance usage, moderate full payment rate. Low-risk customers."},
    1: {"name": "Revolvers", "description": "Moderate balance, low purchases, very low full payment rate. Using credit card as a loan."},
    2: {"name": "High-Value Customers", "description": "High balance, high purchases, moderate full payment rate. Active purchasers."},
    3: {"name": "Low Tenure Customers", "description": "Very low balance and activity, low tenure. New or inactive customers."},
    4: {"name": "VIP/Premium Customers", "description": "Very high balance and purchases, high full payment rate. Top-tier customers."},
    5: {"name": "Convenience Users", "description": "Low balance, highest full payment rate. Pay in full every month."},
    6: {"name": "Cash Advance Seekers", "description": "High balance and very high cash advances. High-risk segment."},
}

app = FastAPI(
    title="Customer Segmentation API",
    description="End-to-end customer segmentation system with K-Means clustering for credit card customers",
    version="2.0.0",
    docs_url="/",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
imputer = joblib.load("models/imputer.joblib")
scaler = joblib.load("models/scaler.joblib")
kmeans = joblib.load("models/kmeans.joblib")

# Load cluster profiles if available
profiles_df = None
if os.path.exists("data/processed/cluster_profiles.csv"):
    profiles_df = pd.read_csv("data/processed/cluster_profiles.csv")

class Customer(BaseModel):
    BALANCE: float | None = Field(None, description="Account balance")
    BALANCE_FREQUENCY: float | None = Field(None, description="How frequently balance is updated")
    PURCHASES: float | None = Field(None, description="Total purchases amount")
    ONEOFF_PURCHASES: float | None = Field(None, description="One-off purchases amount")
    INSTALLMENTS_PURCHASES: float | None = Field(None, description="Installment purchases amount")
    CASH_ADVANCE: float | None = Field(None, description="Cash advance amount")
    PURCHASES_FREQUENCY: float | None = Field(None, description="Purchase frequency")
    ONEOFF_PURCHASES_FREQUENCY: float | None = Field(None, description="One-off purchase frequency")
    PURCHASES_INSTALLMENTS_FREQUENCY: float | None = Field(None, description="Installment purchase frequency")
    CASH_ADVANCE_FREQUENCY: float | None = Field(None, description="Cash advance frequency")
    CASH_ADVANCE_TRX: float | None = Field(None, description="Number of cash advance transactions")
    PURCHASES_TRX: float | None = Field(None, description="Number of purchase transactions")
    CREDIT_LIMIT: float | None = Field(None, description="Credit limit")
    PAYMENTS: float | None = Field(None, description="Payment amount")
    MINIMUM_PAYMENTS: float | None = Field(None, description="Minimum payment amount")
    PRC_FULL_PAYMENT: float | None = Field(None, description="Percentage of full payment")
    TENURE: float | None = Field(None, description="Tenure of customer account")

class PredictionResponse(BaseModel):
    cluster: int
    cluster_name: str
    description: str
    confidence: str = "high"

class ClusterInfo(BaseModel):
    cluster_id: int
    name: str
    description: str
    profile: dict | None = None

@app.get("/health", tags=["System"])
def health():
    """Health check endpoint"""
    return {"status": "healthy", "models_loaded": True}

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(c: Customer):
    """
    Predict customer segment for a single customer.

    Provide customer features and get their cluster assignment with business interpretation.
    """
    try:
        arr = np.array([[getattr(c, f) for f in NUMERIC]], dtype=float)
        arr = imputer.transform(arr)
        arr = scaler.transform(arr)
        cluster = int(kmeans.predict(arr)[0])

        # Get cluster description
        cluster_info = CLUSTER_DESCRIPTIONS.get(cluster, {"name": f"Cluster {cluster}", "description": "Custom segment"})

        return PredictionResponse(
            cluster=cluster,
            cluster_name=cluster_info["name"],
            description=cluster_info["description"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(file: UploadFile = File(..., description="CSV file with customer data")):
    """
    Batch prediction for multiple customers.

    Upload a CSV file with customer features and get cluster assignments for all customers.
    The CSV must contain the required feature columns (BALANCE, PURCHASES, etc.).
    Returns a CSV file with original data plus cluster assignments.
    """
    try:
        # Read uploaded file
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        # Validate columns
        missing_cols = set(NUMERIC) - set(df.columns)
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing_cols}"
            )

        # Prepare features
        X = df[NUMERIC].values
        X_imputed = imputer.transform(X)
        X_scaled = scaler.transform(X_imputed)

        # Predict
        clusters = kmeans.predict(X_scaled)

        # Add predictions to dataframe
        df['cluster'] = clusters
        df['cluster_name'] = [CLUSTER_DESCRIPTIONS.get(c, {"name": f"Cluster {c}"})["name"] for c in clusters]
        df['cluster_description'] = [CLUSTER_DESCRIPTIONS.get(c, {"name": "", "description": "Custom segment"})["description"] for c in clusters]

        # Convert to CSV
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)

        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=predictions_{file.filename}"}
        )

    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="Empty CSV file")
    except pd.errors.ParserError:
        raise HTTPException(status_code=400, detail="Invalid CSV format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/clusters", response_model=list[ClusterInfo], tags=["Clusters"])
def get_clusters():
    """
    Get information about all customer segments/clusters.

    Returns business descriptions and average feature profiles for each cluster.
    """
    clusters = []
    for cluster_id, info in CLUSTER_DESCRIPTIONS.items():
        profile = None
        if profiles_df is not None and cluster_id in profiles_df['cluster'].values:
            profile_row = profiles_df[profiles_df['cluster'] == cluster_id].iloc[0]
            profile = profile_row.to_dict()

        clusters.append(ClusterInfo(
            cluster_id=cluster_id,
            name=info["name"],
            description=info["description"],
            profile=profile
        ))

    return clusters

@app.get("/clusters/{cluster_id}", response_model=ClusterInfo, tags=["Clusters"])
def get_cluster(cluster_id: int):
    """
    Get detailed information about a specific cluster.

    Returns business description and average feature values for the cluster.
    """
    if cluster_id not in CLUSTER_DESCRIPTIONS:
        raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} not found")

    info = CLUSTER_DESCRIPTIONS[cluster_id]
    profile = None

    if profiles_df is not None and cluster_id in profiles_df['cluster'].values:
        profile_row = profiles_df[profiles_df['cluster'] == cluster_id].iloc[0]
        profile = profile_row.to_dict()

    return ClusterInfo(
        cluster_id=cluster_id,
        name=info["name"],
        description=info["description"],
        profile=profile
    )
