"""
evaluate.py
-----------
Load the trained model and compute comprehensive clustering evaluation metrics.
"""

import json
import logging
import os
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_download import run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("evaluate")


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_model(model_path: str) -> dict:
    """Load model bundle (model + label_map + feature_cols)."""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run train.py first.")
    bundle = joblib.load(model_path)
    logger.info(f"Model loaded from {model_path}")
    return bundle


def compute_metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    """
    Compute three standard clustering quality metrics.

    Parameters
    ----------
    X : np.ndarray   scaled feature matrix
    labels : np.ndarray  cluster assignments

    Returns
    -------
    dict with silhouette_score, davies_bouldin_index, calinski_harabasz_score
    """
    sil = silhouette_score(X, labels, sample_size=min(5000, len(X)), random_state=42)
    dbi = davies_bouldin_score(X, labels)
    chi = calinski_harabasz_score(X, labels)
    n_clusters = len(np.unique(labels))

    metrics = {
        "n_clusters": int(n_clusters),
        "n_samples": int(len(X)),
        "silhouette_score": round(float(sil), 6),
        "davies_bouldin_index": round(float(dbi), 6),
        "calinski_harabasz_score": round(float(chi), 2),
    }
    return metrics


def plot_silhouette(X: np.ndarray, labels: np.ndarray, save_path: str) -> None:
    """Plot per-cluster silhouette analysis."""
    sil_vals = silhouette_samples(X, labels)
    n_clusters = len(np.unique(labels))
    avg_score = sil_vals.mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    y_lower = 10
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

    for i, (cluster, color) in enumerate(zip(sorted(np.unique(labels)), colors)):
        cluster_sil = np.sort(sil_vals[labels == cluster])
        size = len(cluster_sil)
        y_upper = y_lower + size
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil, alpha=0.7, color=color)
        ax.text(-0.05, y_lower + size / 2, str(cluster), fontsize=9)
        y_lower = y_upper + 10

    ax.axvline(x=avg_score, color="red", linestyle="--", label=f"Avg={avg_score:.3f}")
    ax.set_title(f"Silhouette Plot (K={n_clusters})", fontsize=13, fontweight="bold")
    ax.set_xlabel("Silhouette Coefficient")
    ax.set_ylabel("Cluster")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Silhouette plot saved to {save_path}")


def plot_cluster_distribution(df: pd.DataFrame, save_path: str) -> None:
    """Bar chart of cluster / segment sizes."""
    counts = df["Segment"].value_counts()
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(counts.index, counts.values, color=plt.cm.tab10.colors[:len(counts)])
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"{val:,}\n({val/len(df)*100:.1f}%)", ha="center", va="bottom", fontsize=9)
    ax.set_title("Customer Segment Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("Segment")
    ax.set_ylabel("Number of Customers")
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Cluster distribution plot saved to {save_path}")


def plot_feature_importance(df: pd.DataFrame, feature_cols: list, save_path: str) -> None:
    """Heatmap of mean feature values per cluster (top 15 features)."""
    top_features = feature_cols[:15]
    available = [f for f in top_features if f in df.columns]
    cluster_means = df.groupby("Cluster")[available].mean()

    # Normalise rows to [0,1] for visual comparison
    cluster_means_norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min() + 1e-9)

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(
        cluster_means_norm.T,
        annot=True, fmt=".2f", cmap="YlOrRd",
        linewidths=0.5, ax=ax, cbar_kws={"label": "Normalised Mean"},
    )
    ax.set_title("Cluster Feature Heatmap (Normalised)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Feature heatmap saved to {save_path}")


def evaluate(config_path: str = "configs/config.yaml") -> dict:
    """Run full evaluation pipeline."""
    cfg = load_config(config_path)
    artifacts_dir = cfg["paths"]["artifacts_dir"]
    models_dir = cfg["paths"]["models_dir"]
    Path(artifacts_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    df, X_scaled, _ = run_pipeline(config_path)

    # Load model
    model_path = os.path.join(models_dir, "trained_model.pkl")
    bundle = load_model(model_path)
    model = bundle["model"]
    label_map = bundle["label_map"]
    feature_cols = bundle["feature_cols"]

    labels = model.predict(X_scaled)
    df["Cluster"] = labels
    df["Segment"] = df["Cluster"].map(label_map)

    # Metrics
    metrics = compute_metrics(X_scaled, labels)
    logger.info("Evaluation Metrics:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v}")

    # Save metrics
    metrics_path = os.path.join(artifacts_dir, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")

    # Plots
    plot_silhouette(X_scaled, labels, os.path.join(artifacts_dir, "silhouette_plot.png"))
    plot_cluster_distribution(df, os.path.join(artifacts_dir, "cluster_distribution.png"))
    plot_feature_importance(df, feature_cols, os.path.join(artifacts_dir, "feature_heatmap.png"))

    return metrics


if __name__ == "__main__":
    evaluate()
