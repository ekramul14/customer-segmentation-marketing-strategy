# 📊 Customer Segmentation: End-to-End ML System

> Customer segmentation system for credit card customers: an automated pipeline, an interactive dashboard, and a REST API.

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🚀 [Try Live Demo →](https://customer-segmentation-marketing-strategy-jvzek2gvt6hbuvwbcfyc8.streamlit.app/)

---

## 🎯 Project Overview

This project groups 8,950 credit card customers into segments a marketing team can act on, using K-Means clustering. It goes beyond a notebook:

- **Automated pipeline:** ingest → clean → train → infer, as separate scripts
- **Data-driven segment names:** each cluster is named from its own average profile, so names stay correct when the model is retrained
- **Interactive dashboard:** 5-page Streamlit app with 2D/3D cluster views, batch and single predictions, and segment comparisons
- **REST API:** FastAPI service with single and batch prediction
- **Explainability:** SHAP shows which features make a customer match their segment
- **Scheduled retraining:** GitHub Actions workflow that reruns the pipeline daily
- **Tests:** pytest checks that segment names match the data and that the API serves them

---

## 📈 Results

**Choosing the number of segments.** The pipeline tries K = 3 to 10 and keeps the K with the best silhouette score:

| K | 3 | 4 | 5 | 6 | **7** | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|
| Silhouette | 0.207 | 0.166 | 0.193 | 0.192 | **0.215** | 0.204 | 0.204 | 0.212 |

**The 7 segments** (from `data/processed/segment_names.csv`):

| Segment | Customers | Share of customers | Share of purchases | What defines them | Suggested approach |
|---|---|---|---|---|---|
| Active Revolvers | 2,417 | 27.0% | 38.8% | Buy often but rarely pay in full (6%) | Installment plans and promotional rates |
| Big Spenders | 236 | 2.6% | 26.3% | Average purchases of $10,002, 10x the overall average | Premium rewards, credit-limit reviews, retention offers |
| Full Payers | 1,113 | 12.4% | 16.0% | Pay the full balance 80% of the time, low balances | Cashback and category rewards |
| Low Activity | 3,479 | 38.9% | 9.9% | Low purchase frequency, average purchases of $255 | Re-engagement incentives, watch for dormancy |
| Cash-Advance Heavy | 991 | 11.1% | 5.7% | Average cash advance of $4,912, 5x average, high balances | Risk monitoring, installment or balance-transfer offers |
| New Customers | 659 | 7.4% | 2.9% | Average tenure 7.3 months vs 11.5 overall | Onboarding and first-purchase incentives |
| High Minimum Payments | 55 | 0.6% | 0.5% | Average minimum payment of $22,811, 27x average | Review individually |

**Key findings**

- The two highest-spending segments (Active Revolvers and Big Spenders) are **30% of customers but 65% of purchases**.
- Big Spenders are just **2.6% of customers but 26% of purchases**.
- Cash-Advance Heavy customers are **11% of customers but 56% of all cash-advance volume**.

---

## 🚀 Quick Start

### Try it online
**👉 [Launch Live Demo](https://customer-segmentation-marketing-strategy-jvzek2gvt6hbuvwbcfyc8.streamlit.app/)** (the free hosting may take a moment to wake up).

### Run locally

```bash
git clone https://github.com/ekramul14/customer-segmentation-marketing-strategy.git
cd customer-segmentation-marketing-strategy
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run the pipeline (only needed when the data changes):

```bash
python src/ingest.py    # Load data (Google Sheet or the seed CSV)
python src/clean.py     # Impute and scale
python src/train.py     # Pick K by silhouette score and train K-Means
python src/infer.py     # Assign segments, build profiles, and name segments
python -m pytest -q     # Check the results
```

Launch the apps:

```bash
streamlit run src/dashboard_app.py          # Dashboard at http://localhost:8501
uvicorn src.api:app --reload --port 8000    # API docs at http://localhost:8000
```

---

## 🎨 Features

### 📱 Dashboard (5 pages)
1. **Home:** segment sizes, descriptions, and suggested approaches
2. **Data Exploration:** interactive 2D/3D cluster views and feature distributions
3. **Batch Prediction:** upload a CSV, get a segment for every customer, download CSV or Excel
4. **Single Prediction:** enter one customer's values and see their segment with a SHAP explanation
5. **Cluster Analysis:** compare two segments side by side with radar charts

### 🔌 API endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Interactive API documentation (Swagger UI) |
| `/health` | GET | Health check |
| `/predict` | POST | Segment, description, and suggested approach for one customer |
| `/predict/batch` | POST | Segments for every row of an uploaded CSV |
| `/clusters` | GET | All segments with shares and average profiles |
| `/clusters/{id}` | GET | One segment's details |

---

## 🔄 Pipeline Details

1. **Ingest (`ingest.py`):** reads a Google Sheet if `SHEET_URL` is set, otherwise the seed CSV, and aligns the columns to the expected schema.
2. **Clean (`clean.py`):** median imputation for missing values (313 missing `MINIMUM_PAYMENTS`, 1 missing `CREDIT_LIMIT`), then standard scaling.
3. **Train (`train.py`):** tries K = 3 to 10, keeps the best silhouette score, trains K-Means (`random_state=42`), and fits a 2-component PCA for visualization.
4. **Infer (`infer.py`):** assigns every customer to a segment, builds each segment's average profile, and names each segment from that profile (`segments.py`).
5. **Scheduled retraining (`.github/workflows/nightly.yml`):** runs the full pipeline and the tests daily at 06:00 UTC, then commits updated artifacts. GitHub pauses scheduled workflows after 60 days without repository activity; re-enable it from the **Actions** tab if needed.

**Why names are data-driven:** K-Means cluster numbers are arbitrary and can change when the model is retrained. `segments.py` names each cluster by what its profile shows (highest purchases, heaviest cash-advance use, highest full-payment rate, shortest tenure, and so on), so a label always matches the customers behind it.

---

## 🛠️ Technology Stack

- **Machine learning:** scikit-learn (K-Means, PCA, preprocessing), pandas, NumPy, SHAP
- **Apps:** FastAPI, Uvicorn, Streamlit, Plotly
- **Automation and testing:** GitHub Actions, pytest

## 📈 Dataset

**Source:** [Kaggle: Credit Card Dataset for Clustering](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata). 8,950 customers, 17 behavioral features (balances, purchases, cash advances, payments, credit limit, tenure).

## 📁 Project Structure

```
├── data/
│   ├── seed/                  # Original dataset
│   └── processed/             # Features, assignments, profiles, segment names, K metrics
├── models/                    # Saved imputer, scaler, K-Means, PCA (.joblib)
├── src/
│   ├── ingest.py  clean.py  train.py  infer.py
│   ├── segments.py            # Data-driven segment names
│   ├── api.py                 # FastAPI service
│   └── dashboard_app.py       # Streamlit dashboard
├── tests/test_segments.py     # pytest checks
├── .github/workflows/nightly.yml
└── requirements.txt
```

## 🔭 Possible next steps

- [ ] Add a Dockerfile for containerized deployment
- [ ] Track segment stability between retrains
- [ ] Add a monitoring page for data drift

## 📝 License

MIT License. See [LICENSE](LICENSE).

## 👤 Author

**Ekramul Haque** · [Portfolio](https://mdtowsif1101.wixsite.com/my-site-1) · [LinkedIn](https://www.linkedin.com/in/ekramulhaque110/) · [GitHub](https://github.com/ekramul14)
