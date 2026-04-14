# P2M – Drift Monitoring and Automated Retraining

## Overview

This project implements an end‑to‑end MLOps prototype for monitoring a machine learning model over time.  
It detects changes in input data, tracks model performance degradation, estimates business impact, and can trigger retraining with automatic comparison of model versions.

## Main Features

- Data quality and schema checks on reference vs current datasets.
- Data drift detection:
  - PSI and KS tests for numerical features.
  - Chi‑square tests for categorical features.
- Model / concept drift detection via metric drop (classification or regression).
- Simple business cost estimation:
  - Cost of performance degradation.
  - Cost of retraining.
- Automated retraining (scikit‑learn) and experiment tracking with MLflow.
- Streamlit dashboard to visualize drift, performance, costs, and old vs new models.

## How It Works

1. Load two CSV files: **Reference** (historical data) and **Current** (new data).
2. The app:
   - Checks data quality and schema alignment.
   - Computes drift statistics per feature and a global drift score.
   - Trains a base model on Reference and evaluates it on Reference and Current.
   - Estimates drift cost vs retrain cost and suggests whether to retrain.
3. If retraining is launched:
   - A new model is trained and logged in MLflow.
   - Metrics on Current are compared between old and new models.

## Run Locally

```bash
git clone <REPO_URL>
cd p2m

python -m venv .venv
.\.venv\Scripts\activate      # on Windows

python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app will be available at http://localhost:8501.
