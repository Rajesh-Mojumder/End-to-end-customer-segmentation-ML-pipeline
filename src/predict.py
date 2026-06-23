"""
predict.py
----------
Load saved model, accept new customer data, predict cluster,
and return a business segment label.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Union

import joblib
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("predict")

# ── Segment metadata for downstream use ──────────────────────────────────────
SEGMENT_DESCRIPTIONS = {
    "VIP Customers": {
        "description": "Highest lifetime value; premium spenders with frequent, recent purchases.",
        "strategy": "Exclusive loyalty rewards, early-access offers, dedicated account manager.",
        "icon": "👑",
    },
    "Loyal Customers": {
        "description": "Regular buyers with strong engagement and consistent spending.",
        "strategy": "Loyalty programs, personalised recommendations, birthday discounts.",
        "icon": "⭐",
    },
    "Potential Customers": {
        "description": "Moderate spenders showing growth signals; not yet fully committed.",
        "strategy": "Targeted upsell campaigns, product education, limited-time offers.",
        "icon": "🚀",
    },
    "New Customers": {
        "description": "Recently acquired; low tenure and purchase history.",
        "strategy": "Welcome series, onboarding guides, first-purchase incentives.",
        "icon": "🌱",
    },
    "At-Risk Customers": {
        "description": "Low recent activity; at risk of churning.",
        "strategy": "Win-back campaigns, personalised outreach, reactivation discounts.",
        "icon": "⚠️",
    },
}


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_artifacts(
    model_path: str = "models/trained_model.pkl",
    scaler_path: str = "models/scaler.pkl",
) -> tuple:
    """Load model bundle and scaler."""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run train.py first.")
    if not Path(scaler_path).exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}. Run train.py first.")

    bundle = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return bundle["model"], scaler, bundle["label_map"], bundle["feature_cols"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply same feature engineering as training pipeline."""
    df = df.copy()
    df["Total_Spending"] = df.get("Lifetime_Value", df.get("Total_Spending", 0))
    membership = df.get("Membership_Years", pd.Series(np.ones(len(df))))
    df["Purchase_Frequency"] = df.get("Total_Purchases", 0) / membership.clip(lower=0.1)
    df["RFM_Recency"] = 1 / (df.get("Days_Since_Last_Purchase", 30) + 1)
    df["RFM_Frequency"] = df.get("Total_Purchases", 0)
    df["RFM_Monetary"] = df.get("Lifetime_Value", 0)
    df["Customer_Age_Group"] = pd.cut(
        df.get("Age", pd.Series([35] * len(df))),
        bins=[0, 30, 50, 200], labels=[0, 1, 2],
    ).astype(float)
    return df


def preprocess_input(
    df: pd.DataFrame,
    feature_cols: list,
    scaler,
) -> np.ndarray:
    """
    Apply preprocessing and scaling to new data.

    Parameters
    ----------
    df : pd.DataFrame   raw customer dataframe (one or many rows)
    feature_cols : list  ordered feature list used at training
    scaler       : fitted StandardScaler

    Returns
    -------
    np.ndarray  scaled feature matrix
    """
    # Encode gender
    if "Gender" in df.columns:
        df["Gender_Encoded"] = (df["Gender"] == "Male").astype(int)

    # Engineer features
    df = engineer_features(df)

    # Fill missing numerics with 0 (safe default for prediction)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    df[feature_cols] = df[feature_cols].fillna(0)

    X = df[feature_cols].values
    return scaler.transform(X)


def predict(
    input_data: Union[pd.DataFrame, dict, str],
    config_path: str = "configs/config.yaml",
) -> pd.DataFrame:
    """
    Predict customer segments for new data.

    Parameters
    ----------
    input_data : DataFrame, dict (single customer), or path to CSV
    config_path : str

    Returns
    -------
    pd.DataFrame with added columns: Cluster, Segment, Segment_Description, Strategy, Icon
    """
    cfg = load_config(config_path)
    models_dir = cfg["paths"]["models_dir"]

    model, scaler, label_map, feature_cols = load_artifacts(
        model_path=os.path.join(models_dir, "trained_model.pkl"),
        scaler_path=os.path.join(models_dir, "scaler.pkl"),
    )

    # Normalise input
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    elif isinstance(input_data, str):
        df = pd.read_csv(input_data)
    else:
        df = input_data.copy()

    logger.info(f"Predicting for {len(df):,} customer(s) …")

    X_scaled = preprocess_input(df, feature_cols, scaler)
    clusters = model.predict(X_scaled)

    df = df.copy()
    df["Cluster"] = clusters
    df["Segment"] = df["Cluster"].map(label_map)
    df["Segment_Description"] = df["Segment"].map(
        lambda s: SEGMENT_DESCRIPTIONS.get(s, {}).get("description", "Unknown")
    )
    df["Marketing_Strategy"] = df["Segment"].map(
        lambda s: SEGMENT_DESCRIPTIONS.get(s, {}).get("strategy", "Unknown")
    )
    df["Icon"] = df["Segment"].map(
        lambda s: SEGMENT_DESCRIPTIONS.get(s, {}).get("icon", "❓")
    )

    logger.info("Prediction complete. Segment distribution:")
    for seg, cnt in df["Segment"].value_counts().items():
        logger.info(f"  {seg}: {cnt:,} customers")

    return df


# ── CLI demo ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Single customer demo
    sample_customer = {
        "Age": 35,
        "Gender": "Female",
        "Membership_Years": 3.5,
        "Login_Frequency": 15,
        "Session_Duration_Avg": 30,
        "Pages_Per_Session": 9,
        "Cart_Abandonment_Rate": 50,
        "Wishlist_Items": 5,
        "Total_Purchases": 20,
        "Average_Order_Value": 120,
        "Days_Since_Last_Purchase": 10,
        "Discount_Usage_Rate": 40,
        "Returns_Rate": 5,
        "Email_Open_Rate": 25,
        "Customer_Service_Calls": 4,
        "Product_Reviews_Written": 3,
        "Social_Media_Engagement_Score": 30,
        "Mobile_App_Usage": 20,
        "Payment_Method_Diversity": 2,
        "Lifetime_Value": 1500,
        "Credit_Balance": 2000,
    }

    result = predict(sample_customer)
    seg = result["Segment"].iloc[0]
    icon = result["Icon"].iloc[0]
    desc = result["Segment_Description"].iloc[0]
    strat = result["Marketing_Strategy"].iloc[0]

    print(f"\n{'='*60}")
    print(f"PREDICTION RESULT")
    print(f"{'='*60}")
    print(f"Cluster   : {result['Cluster'].iloc[0]}")
    print(f"Segment   : {icon} {seg}")
    print(f"Description: {desc}")
    print(f"Strategy  : {strat}")
    print(f"{'='*60}\n")
