# 🎯 Customer Segmentation ML Project

> End-to-end ML pipeline for customer segmentation using KMeans clustering, MLflow tracking, and a Streamlit dashboard.

---

## 📐 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA PIPELINE                                 │
│  CSV Data → Validate → Preprocess → Feature Engineering → Scale │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                  ML TRAINING (train.py)                          │
│  Elbow Method → Best K Selection → KMeans Fit → PCA Viz         │
│                         │                                        │
│              MLflow Experiment Tracking                          │
│      (params • metrics • artifacts • model registry)            │
└─────────────────────┬───────────────────────────────────────────┘
                      │
        ┌─────────────┴─────────────┐
        │                           │
┌───────▼──────┐          ┌─────────▼────────┐
│ evaluate.py  │          │   predict.py      │
│  Silhouette  │          │  Load Model       │
│  DBI Score   │          │  New Customer In  │
│  CHI Score   │          │  Segment Out      │
└──────────────┘          └─────────┬─────────┘
                                    │
                          ┌─────────▼────────┐
                          │   app.py          │
                          │  Streamlit UI     │
                          │  Upload CSV       │
                          │  Visualize        │
                          │  Download         │
                          └──────────────────┘
```

---

## 📊 Dataset Description

| Property | Value |
|---|---|
| Source | E-Commerce Customer Churn Dataset |
| Rows | 50,000 customers |
| Features | 25 columns |
| Target | Unsupervised Segmentation |

**Key Features Used:**
- Demographics: Age, Gender, Country
- Behavior: Login_Frequency, Session_Duration_Avg, Pages_Per_Session
- Purchase: Total_Purchases, Average_Order_Value, Days_Since_Last_Purchase
- Engagement: Email_Open_Rate, Social_Media_Engagement_Score
- Value: Lifetime_Value, Credit_Balance

---

## 🗂️ Project Structure

```
final-project/
├── README.md
├── final_report.md
├── requirements.txt
├── .gitignore
├── Dockerfile
├── src/
│   ├── data_download.py   # Data pipeline
│   ├── train.py           # Model training + MLflow
│   ├── evaluate.py        # Evaluation metrics
│   └── predict.py         # Prediction module
├── app/
│   └── app.py             # Streamlit dashboard
├── notebooks/
│   └── exploration.ipynb  # EDA notebook
├── configs/
│   └── config.yaml        # All configuration
├── models/                # Saved model artifacts
├── data/                  # Raw + processed data
├── artifacts/             # Plots + metrics JSON
├── mlruns/                # MLflow tracking
└── screenshots/           # Project screenshots
```

---

## 🚀 Installation Guide

### 1. Clone & Setup

```bash
git clone https://github.com/your-org/customer-segmentation.git
cd customer-segmentation/final-project
pip install -r requirements.txt
```

### 2. Add Dataset

Place `ecommerce_customer_churn_dataset.csv` in `data/`.

### 3. Run Pipeline

```bash
# Step 1: Data pipeline only
python src/data_download.py

# Step 2: Train model (auto-selects best K)
MLFLOW_ALLOW_FILE_STORE=true python src/train.py

# Step 3: Evaluate
MLFLOW_ALLOW_FILE_STORE=true python src/evaluate.py

# Step 4: Predict single customer
python src/predict.py

# Step 5: Launch dashboard
streamlit run app/app.py
```

---

## 🐳 Docker Instructions

### Build & Run

```bash
# Build image
docker build -t customer-segmentation:latest .

# Run container (train first, then serve)
docker run -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/artifacts:/app/artifacts \
  -e MLFLOW_ALLOW_FILE_STORE=true \
  customer-segmentation:latest
```

### Access App

Open browser → `http://localhost:8501`

---

## 📈 MLflow Usage

```bash
# View MLflow UI
MLFLOW_ALLOW_FILE_STORE=true mlflow ui --backend-store-uri mlruns

# Open: http://localhost:5000
```

Tracks:
- **Parameters:** algorithm, best_k, n_init, random_seed
- **Metrics:** silhouette_score, davies_bouldin_index, calinski_harabasz_score
- **Artifacts:** model pkl, PCA plot, cluster profiles CSV, metrics JSON

---

## 📊 Results

| Metric | Value |
|---|---|
| Best K | 2 |
| Silhouette Score | 0.2451 |
| Davies-Bouldin Index | 1.6053 |
| Calinski-Harabasz Score | 16,094.5 |
| Training Samples | 49,980 |
| Features Used | 25 |

### Customer Segments Identified

| Segment | Description |
|---|---|
| 👑 VIP Customers | Highest LTV, frequent buyers |
| ⭐ Loyal Customers | Regular, consistent buyers |
| 🚀 Potential Customers | Growing engagement |
| 🌱 New Customers | Recently acquired |
| ⚠️ At-Risk Customers | Low activity, may churn |

---

## 🔮 Future Improvements

- [ ] Add DBSCAN / Gaussian Mixture Models
- [ ] Implement real-time prediction API (FastAPI)
- [ ] Add customer lifetime value prediction (regression)
- [ ] Integrate with CRM systems
- [ ] Add A/B testing framework for campaign targeting
- [ ] Automated retraining pipeline (Airflow/Prefect)
- [ ] Migrate MLflow to PostgreSQL + S3 backend
