"""
data_download.py
----------------
Data ingestion, validation, preprocessing, and feature engineering
for the Customer Segmentation ML project.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler
import joblib

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("data_download")


# ── Config helpers ────────────────────────────────────────────────────────────
def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ── Ingestion ─────────────────────────────────────────────────────────────────
def ingest_data(raw_path: str) -> pd.DataFrame:
    """
    Load raw CSV data from disk.

    Parameters
    ----------
    raw_path : str
        Path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Raw dataframe.
    """
    logger.info(f"Ingesting data from {raw_path}")
    if not Path(raw_path).exists():
        raise FileNotFoundError(f"Dataset not found at {raw_path}")
    df = pd.read_csv(raw_path)
    logger.info(f"Loaded {len(df):,} rows × {df.shape[1]} columns")
    return df


# ── Validation ────────────────────────────────────────────────────────────────
def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean obvious data quality issues.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Validated dataframe.
    """
    logger.info("Validating data …")
    original_len = len(df)

    # Drop exact duplicates
    df = df.drop_duplicates()
    logger.info(f"Dropped {original_len - len(df):,} duplicate rows")

    # Age sanity check (5 – 100)
    if "Age" in df.columns:
        df = df[df["Age"].between(5, 100) | df["Age"].isna()]

    # Total_Purchases must be non-negative
    if "Total_Purchases" in df.columns:
        df.loc[df["Total_Purchases"] < 0, "Total_Purchases"] = np.nan

    logger.info(f"Validated dataset: {len(df):,} rows remain")
    return df


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values and encode categoricals.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Preprocessed dataframe.
    """
    logger.info("Preprocessing data …")

    # Numerical columns: impute with median
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.debug(f"Imputed '{col}' with median={median_val:.4f}")

    # Gender encoding
    if "Gender" in df.columns:
        df["Gender_Encoded"] = (df["Gender"] == "Male").astype(int)

    # Signup_Quarter ordinal encoding
    if "Signup_Quarter" in df.columns:
        quarter_map = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
        df["Signup_Quarter_Encoded"] = df["Signup_Quarter"].map(quarter_map).fillna(2)

    logger.info("Preprocessing complete")
    return df


# ── Feature Engineering ───────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features for segmentation.

    Features created
    ----------------
    Total_Spending       : Lifetime_Value (proxy for total spend)
    Purchase_Frequency   : Total_Purchases / max(Membership_Years, 0.1)
    RFM_Recency          : inverse of Days_Since_Last_Purchase
    RFM_Frequency        : Total_Purchases (normalised internally)
    RFM_Monetary         : Lifetime_Value
    Customer_Age_Group   : binned age (0=Young, 1=Middle, 2=Senior)

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    logger.info("Engineering features …")

    # Total Spending proxy
    df["Total_Spending"] = df["Lifetime_Value"]

    # Purchase Frequency (purchases per membership year)
    df["Purchase_Frequency"] = df["Total_Purchases"] / df["Membership_Years"].clip(lower=0.1)

    # RFM features
    df["RFM_Recency"] = 1 / (df["Days_Since_Last_Purchase"] + 1)
    df["RFM_Frequency"] = df["Total_Purchases"]
    df["RFM_Monetary"] = df["Lifetime_Value"]

    # Customer age group
    df["Customer_Age_Group"] = pd.cut(
        df["Age"],
        bins=[0, 30, 50, 200],
        labels=[0, 1, 2],
    ).astype(float)

    logger.info("Feature engineering complete")
    return df


# ── Scaling ───────────────────────────────────────────────────────────────────
def scale_features(
    df: pd.DataFrame,
    feature_cols: list,
    scaler_path: str = "models/scaler.pkl",
) -> Tuple[np.ndarray, StandardScaler]:
    """
    Standard-scale selected feature columns and persist the scaler.

    Parameters
    ----------
    df : pd.DataFrame
    feature_cols : list
        Column names to include.
    scaler_path : str
        Where to save the fitted scaler.

    Returns
    -------
    X_scaled : np.ndarray
    scaler : StandardScaler
    """
    logger.info(f"Scaling {len(feature_cols)} features …")
    X = df[feature_cols].copy()
    # Final safety: fill any remaining NaN with column median
    X = X.fillna(X.median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler saved to {scaler_path}")
    return X_scaled, scaler


# ── Pipeline entry point ──────────────────────────────────────────────────────
def run_pipeline(config_path: str = "configs/config.yaml") -> Tuple[pd.DataFrame, np.ndarray, list]:
    """
    Execute the full data pipeline.

    Returns
    -------
    df : pd.DataFrame        cleaned + engineered dataframe
    X_scaled : np.ndarray   scaled feature matrix
    feature_cols : list     ordered list of feature names used
    """
    cfg = load_config(config_path)
    seed = cfg["project"]["random_seed"]
    np.random.seed(seed)

    raw_path = cfg["paths"]["data_raw"]
    processed_path = cfg["paths"]["data_processed"]
    scaler_path = os.path.join(cfg["paths"]["models_dir"], "scaler.pkl")

    df = ingest_data(raw_path)
    df = validate_data(df)
    df = preprocess_data(df)
    df = engineer_features(df)

    # Build feature list from config + engineered features
    base_features = cfg["features"]["numerical"]
    engineered = cfg["features"]["engineered"]
    all_features = [f for f in base_features + engineered if f in df.columns]

    X_scaled, _ = scale_features(df, all_features, scaler_path)

    # Save processed data
    Path(processed_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
    logger.info(f"Processed data saved to {processed_path}")

    return df, X_scaled, all_features


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df, X_scaled, features = run_pipeline()
    logger.info(f"Pipeline done. Feature matrix shape: {X_scaled.shape}")
