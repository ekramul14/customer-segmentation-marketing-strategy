# 📊 Customer Segmentation: End-to-End ML System

> **Production-ready customer segmentation system** with interactive dashboard, REST API, and automated ML pipeline for credit card customer analysis.

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.119-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.50-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🚀 [Try Live Demo →](https://customer-segmentation-marketing-strategy-jvzek2gvt6hbuvwbcfyc8.streamlit.app/)

---

## 🎯 Project Overview

This project demonstrates a **complete, production-ready machine learning system** that segments credit card customers into actionable business groups using K-Means clustering. Unlike typical notebook-only projects, this showcases:

✅ **Interactive Web Dashboard** - Multi-page Streamlit app with 3D visualizations
✅ **Production REST API** - FastAPI with batch prediction and auto-generated docs
✅ **Model Explainability** - SHAP values showing feature importance
✅ **End-to-End Pipeline** - Data ingestion → preprocessing → training → inference
✅ **Automated Retraining** - GitHub Actions for nightly model updates
✅ **Real-world Deployment** - Docker-ready, cloud-deployable architecture

---

## 🚀 Quick Start

### 🌐 Try it Online (No Setup Required!)

**👉 [Launch Live Demo](https://customer-segmentation-marketing-strategy-jvzek2gvt6hbuvwbcfyc8.streamlit.app/)** - Explore the full interactive dashboard instantly!

### 1️⃣ Clone & Setup (Local Development)

```bash
# Clone the repository
git clone https://github.com/ekramul14/customer-segmentation-marketing-strategy.git
cd customer-segmentation-marketing-strategy

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ Run the ML Pipeline

```bash
# Run the complete pipeline (only needed once or when updating data)
python src/ingest.py    # Ingest data
python src/clean.py     # Clean & preprocess
python src/train.py     # Train K-Means model
python src/infer.py     # Generate predictions
```

### 3️⃣ Launch the Applications

**Option A: Interactive Dashboard**
```bash
streamlit run src/dashboard_app.py
```
→ Opens at `http://localhost:8501`

**Option B: REST API**
```bash
uvicorn src.api:app --reload --port 8000
```
→ API docs at `http://localhost:8000`

---

## 📊 Customer Segments

Our model identifies **7 distinct customer segments**:

| Segment | Description | Business Strategy |
|---------|-------------|-------------------|
| **💼 VIP/Premium** | High balance & purchases, excellent payment rate | Premium rewards, exclusive offers |
| **💳 Transactors** | Low balance, minimal cash advances, reliable | Standard rewards, cashback programs |
| **🔄 Revolvers** | Moderate balance, low payment rate | Interest rate promotions, payment plans |
| **💰 High-Value** | High spending, moderate payment rate | Targeted upsells, spending incentives |
| **✅ Convenience Users** | Pay in full monthly, frequent transactions | Fee waivers, convenience benefits |
| **💵 Cash Advance Seekers** | High cash advances, risky profile | Risk mitigation, fee optimization |
| **👤 Low Tenure** | New/inactive customers | Onboarding campaigns, engagement offers |

---

## 🎨 Features

### 📱 Interactive Dashboard (5 Pages)

1. **🏠 Home** - Project overview, segment distribution, quick start guide
2. **📊 Data Exploration** - Interactive 2D/3D cluster visualizations, feature distributions
3. **📤 Batch Prediction** - Upload CSV → Get predictions → Download results (CSV/Excel)
4. **🎯 Single Prediction** - Manual input with AI explainability (SHAP values)
5. **🔍 Cluster Analysis** - Side-by-side comparison with radar charts

**Key Capabilities:**
- 3D rotatable cluster visualizations (Plotly)
- Drag-and-drop file upload
- One-click CSV/Excel export
- SHAP feature importance explanations
- Professional UI with custom styling

### 🔌 REST API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Interactive API documentation (Swagger UI) |
| `/health` | GET | Health check endpoint |
| `/predict` | POST | Single customer prediction with cluster name |
| `/predict/batch` | POST | Batch prediction from uploaded CSV |
| `/clusters` | GET | List all segments with descriptions |
| `/clusters/{id}` | GET | Detailed info for specific cluster |

**API Features:**
- Auto-generated interactive docs
- Batch processing via file upload
- Business-friendly cluster descriptions
- Comprehensive error handling
- CORS-enabled for frontend integration

---

## 📁 Project Structure

```
├── data/
│   ├── seed/                  # Original dataset
│   └── processed/             # Cleaned data, cluster assignments
├── models/                    # Trained models (.joblib files)
│   ├── imputer.joblib
│   ├── scaler.joblib
│   ├── kmeans.joblib
│   └── pca.joblib
├── src/
│   ├── ingest.py             # Data ingestion from Google Sheets/CSV
│   ├── clean.py              # Data cleaning & preprocessing
│   ├── train.py              # K-Means training with auto K selection
│   ├── infer.py              # Batch inference & profiling
│   ├── api.py                # FastAPI REST API
│   └── dashboard_app.py      # Streamlit dashboard (700+ lines)
├── .github/workflows/
│   └── nightly.yml           # Automated retraining pipeline
├── requirements.txt          # Python dependencies
├── .env.example             # Environment configuration template
└── README.md                # This file
```

---

## 🛠️ Technology Stack

**Machine Learning & Data Science**
- `scikit-learn` - K-Means clustering, PCA, preprocessing
- `pandas` & `numpy` - Data manipulation
- `shap` - Model explainability

**Web & API**
- `FastAPI` - REST API with automatic documentation
- `Streamlit` - Interactive dashboard
- `Plotly` - Interactive 3D visualizations

**Deployment & DevOps**
- `uvicorn` - ASGI server
- `GitHub Actions` - Automated workflows
- `Docker` (optional) - Containerization

---

## 📈 Dataset

**Source:** [Kaggle - Credit Card Customer Data](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata)

**Features (17 total):**
- **Balance & Payments:** `BALANCE`, `PAYMENTS`, `MINIMUM_PAYMENTS`, `CREDIT_LIMIT`
- **Purchase Behavior:** `PURCHASES`, `ONEOFF_PURCHASES`, `INSTALLMENTS_PURCHASES`, `PURCHASES_TRX`
- **Cash Advances:** `CASH_ADVANCE`, `CASH_ADVANCE_TRX`, `CASH_ADVANCE_FREQUENCY`
- **Activity Metrics:** `PURCHASES_FREQUENCY`, `PRC_FULL_PAYMENT`, `BALANCE_FREQUENCY`, `TENURE`

**Size:** ~8,950 credit card customers

---

## 🔄 ML Pipeline Details

### 1. Data Ingestion (`ingest.py`)
- Supports live Google Sheets or local CSV
- Automatic data validation
- Metadata tracking (source, timestamp)

### 2. Data Cleaning (`clean.py`)
- Median imputation for missing values
- StandardScaler normalization
- Model serialization for inference

### 3. Model Training (`train.py`)
- Auto-selects optimal K using silhouette score (K=3-10)
- K-Means clustering with configurable parameters
- PCA for dimensionality reduction
- Model versioning with joblib

### 4. Inference & Profiling (`infer.py`)
- Batch prediction for all customers
- Cluster profile generation (mean features)
- Results saved as CSV for dashboard

### 5. Automated Retraining
- GitHub Actions workflow runs nightly at 6 AM UTC
- Full pipeline execution
- Auto-commits updated artifacts


---

## 📊 How to Use

### For Data Scientists/Analysts:

1. **Explore the Dashboard:**
   - Navigate through 5 interactive pages
   - Visualize clusters in 2D/3D space
   - Compare segments with radar charts

2. **Test Predictions:**
   - Upload your own CSV in "Batch Prediction"
   - Or manually input features in "Single Prediction"
   - See SHAP explanations for why customers are assigned to clusters

### For Developers:

1. **Integrate the API:**
```python
import requests

# Single prediction
response = requests.post("http://localhost:8000/predict", json={
    "BALANCE": 1000.0,
    "PURCHASES": 500.0,
    # ... other features
})
print(response.json())
# {"cluster": 2, "cluster_name": "High-Value Customers", "description": "..."}

# Batch prediction
files = {"file": open("customers.csv", "rb")}
response = requests.post("http://localhost:8000/predict/batch", files=files)
```

2. **Get Cluster Info:**
```python
# List all clusters
clusters = requests.get("http://localhost:8000/clusters").json()

# Get specific cluster
cluster_2 = requests.get("http://localhost:8000/clusters/2").json()
```

---

## 🧪 Testing

### Test the API
```bash
# Start the API
uvicorn src.api:app --reload --port 8000

# Visit http://localhost:8000 for interactive docs
# Try the endpoints in Swagger UI
```

### Test the Dashboard
```bash
# Start the dashboard
streamlit run src/dashboard_app.py

# Visit http://localhost:8501
# Navigate through all 5 pages
```

---

## 🎓 What This Project Demonstrates

### Technical Skills:
✅ **End-to-End ML Pipeline** - Not just modeling, but complete system design
✅ **API Development** - RESTful design with FastAPI
✅ **Interactive UIs** - Multi-page Streamlit dashboard
✅ **Model Explainability** - SHAP integration for interpretability
✅ **Production Thinking** - Error handling, validation, documentation
✅ **DevOps/MLOps** - Automated workflows, CI/CD concepts

### Business Value:
✅ **Actionable Insights** - 7 distinct customer segments with marketing strategies
✅ **Scalability** - Batch processing for thousands of customers
✅ **User Experience** - Non-technical users can upload data and get results
✅ **Real-world Ready** - Can be deployed and used immediately

---

## 📸 Screenshots

### Dashboard Home
![Dashboard Home](docs/screenshots/home.png)

### 3D Cluster Visualization
![3D Clusters](docs/screenshots/3d_clusters.png)

### Batch Prediction
![Batch Prediction](docs/screenshots/batch_prediction.png)

### API Documentation
![API Docs](docs/screenshots/api_docs.png)

---

## 🔧 Configuration

Create a `.env` file (use `.env.example` as template):

```bash
# Optional: Live data from Google Sheets
SHEET_URL=https://docs.google.com/spreadsheets/.../export?format=csv

# Model configuration
K_CLUSTERS=7              # Number of clusters (or auto-select)
AUTO_SELECT_K=true        # Use silhouette score to find optimal K
```

---

## 🤝 Contributing

Contributions are welcome! Areas for enhancement:

- [ ] Add Dockerfile for containerization
- [ ] Implement A/B testing for model versions
- [ ] Add more visualization types (dendrograms, elbow plots)
- [ ] Create React frontend for API
- [ ] Add user authentication
- [ ] Implement model monitoring dashboard

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Ekramul Haque**

[![Portfolio](https://img.shields.io/badge/Portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://mdtowsif1101.wixsite.com/my-site-1)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ekramulhaque110/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ekramul14)

---

## 🙏 Acknowledgements

- Dataset: [Kaggle Credit Card Dataset](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata)
- Clustering Algorithm: K-Means (scikit-learn)
- Explainability: SHAP library
- Visualizations: Plotly Express

---

## ⭐ Star this repo if you found it helpful!

**Looking for a Data Scientist who can build complete, deployable ML systems? Let's connect!**
