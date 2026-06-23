"""
train.py
--------
KMeans clustering with automatic K selection via Silhouette Score.
Tracks parameters, metrics, and artifacts with MLflow.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_download import run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train")


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── Elbow + Silhouette search ─────────────────────────────────────────────────
def find_best_k(
    X: np.ndarray,
    k_min: int = 2,
    k_max: int = 10,
    random_seed: int = 42,
    n_init: int = 10,
    max_iter: int = 300,
) -> Tuple[int, Dict[int, float], Dict[int, float]]:
    """
    Evaluate KMeans for k in [k_min, k_max] and return best k.

    Returns
    -------
    best_k : int
    inertias : dict  {k: inertia}
    sil_scores : dict  {k: silhouette_score}
    """
    inertias: Dict[int, float] = {}
    sil_scores: Dict[int, float] = {}

    logger.info(f"Searching K from {k_min} to {k_max} …")
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=n_init, max_iter=max_iter, random_state=random_seed)
        labels = km.fit_predict(X)
        inertias[k] = km.inertia_
        sil_scores[k] = silhouette_score(X, labels, sample_size=min(5000, len(X)), random_state=random_seed)
        logger.info(f"  K={k:2d} | inertia={inertias[k]:,.0f} | silhouette={sil_scores[k]:.4f}")

    best_k = max(sil_scores, key=sil_scores.get)
    logger.info(f"Best K = {best_k} (silhouette={sil_scores[best_k]:.4f})")
    return best_k, inertias, sil_scores


# ── Plot helpers ──────────────────────────────────────────────────────────────
def plot_elbow_silhouette(
    inertias: Dict[int, float],
    sil_scores: Dict[int, float],
    best_k: int,
    save_path: str,
) -> None:
    ks = sorted(inertias.keys())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("KMeans Model Selection", fontsize=14, fontweight="bold")

    # Elbow
    axes[0].plot(ks, [inertias[k] for k in ks], "b-o", linewidth=2)
    axes[0].axvline(x=best_k, color="red", linestyle="--", label=f"Best K={best_k}")
    axes[0].set_title("Elbow Method (Inertia)")
    axes[0].set_xlabel("Number of Clusters (K)")
    axes[0].set_ylabel("Inertia")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Silhouette
    axes[1].bar(ks, [sil_scores[k] for k in ks], color=["red" if k == best_k else "steelblue" for k in ks])
    axes[1].set_title("Silhouette Score by K")
    axes[1].set_xlabel("Number of Clusters (K)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Elbow/Silhouette plot saved to {save_path}")


def plot_pca_clusters(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    n_components: int = 2,
    save_path: str = "artifacts/pca_visualization.png",
) -> None:
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=labels, cmap="tab10", alpha=0.5, s=10,
    )
    plt.colorbar(scatter, ax=ax, label="Cluster")
    ax.set_title(
        f"PCA Cluster Visualization\n"
        f"(PC1={explained[0]:.1%}, PC2={explained[1]:.1%} variance explained)",
        fontsize=12,
    )
    ax.set_xlabel(f"PC1 ({explained[0]:.1%})")
    ax.set_ylabel(f"PC2 ({explained[1]:.1%})")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"PCA visualization saved to {save_path}")


# ── Cluster profiles ──────────────────────────────────────────────────────────
SEGMENT_NAMES = [
    "VIP Customers",
    "Loyal Customers",
    "Potential Customers",
    "New Customers",
    "At-Risk Customers",
]

def assign_segment_labels(df: pd.DataFrame, labels: np.ndarray, n_clusters: int) -> pd.DataFrame:
    """Assign human-readable segment names based on Lifetime_Value rank."""
    df = df.copy()
    df["Cluster"] = labels

    cluster_value = df.groupby("Cluster")["Lifetime_Value"].mean().sort_values(ascending=False)
    label_map = {
        cluster: SEGMENT_NAMES[i % len(SEGMENT_NAMES)]
        for i, cluster in enumerate(cluster_value.index)
    }
    df["Segment"] = df["Cluster"].map(label_map)
    return df, label_map


def build_cluster_profiles(df: pd.DataFrame, feature_cols: list, save_path: str) -> pd.DataFrame:
    """Compute per-cluster mean statistics and save CSV."""
    profile_cols = feature_cols + ["Segment"]
    profile = (
        df[profile_cols + ["Cluster"]]
        .groupby(["Cluster", "Segment"])
        .mean()
        .reset_index()
    )
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    profile.to_csv(save_path, index=False)
    logger.info(f"Cluster profiles saved to {save_path}")
    return profile


# ── Main training function ────────────────────────────────────────────────────
def train(config_path: str = "configs/config.yaml") -> None:
    cfg = load_config(config_path)
    seed = cfg["project"]["random_seed"]
    np.random.seed(seed)

    artifacts_dir = cfg["paths"]["artifacts_dir"]
    models_dir = cfg["paths"]["models_dir"]
    Path(artifacts_dir).mkdir(parents=True, exist_ok=True)
    Path(models_dir).mkdir(parents=True, exist_ok=True)

    # ── Data pipeline ─────────────────────────────────────────────────────────
    logger.info("Running data pipeline …")
    df, X_scaled, feature_cols = run_pipeline(config_path)

    # ── MLflow setup ──────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=cfg["mlflow"]["run_name"]):

        # ── K search ─────────────────────────────────────────────────────────
        k_min = cfg["model"]["k_min"]
        k_max = cfg["model"]["k_max"]
        n_init = cfg["model"]["n_init"]
        max_iter = cfg["model"]["max_iter"]

        best_k, inertias, sil_scores = find_best_k(
            X_scaled, k_min=k_min, k_max=k_max,
            random_seed=seed, n_init=n_init, max_iter=max_iter,
        )

        # ── Log params ───────────────────────────────────────────────────────
        mlflow.log_params({
            "algorithm": cfg["model"]["algorithm"],
            "k_min": k_min,
            "k_max": k_max,
            "best_k": best_k,
            "n_init": n_init,
            "max_iter": max_iter,
            "random_seed": seed,
            "n_features": len(feature_cols),
            "n_samples": len(df),
        })

        # ── Train final model ─────────────────────────────────────────────────
        logger.info(f"Training final KMeans with K={best_k} …")
        final_model = KMeans(
            n_clusters=best_k, n_init=n_init,
            max_iter=max_iter, random_state=seed,
        )
        labels = final_model.fit_predict(X_scaled)

        # ── Evaluation metrics ────────────────────────────────────────────────
        sil = silhouette_score(X_scaled, labels, sample_size=min(5000, len(X_scaled)), random_state=seed)
        dbi = davies_bouldin_score(X_scaled, labels)
        chi = calinski_harabasz_score(X_scaled, labels)

        metrics = {
            "silhouette_score": round(sil, 6),
            "davies_bouldin_index": round(dbi, 6),
            "calinski_harabasz_score": round(chi, 2),
            "inertia": round(final_model.inertia_, 2),
            "best_k": best_k,
        }
        mlflow.log_metrics(metrics)
        logger.info(f"Metrics: {metrics}")

        # ── Save metrics JSON ─────────────────────────────────────────────────
        metrics_path = os.path.join(artifacts_dir, "evaluation_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        mlflow.log_artifact(metrics_path)

        # ── Plots ─────────────────────────────────────────────────────────────
        elbow_path = os.path.join(artifacts_dir, "elbow_silhouette.png")
        plot_elbow_silhouette(inertias, sil_scores, best_k, elbow_path)
        mlflow.log_artifact(elbow_path)

        pca_path = os.path.join(artifacts_dir, "pca_visualization.png")
        plot_pca_clusters(X_scaled, labels, save_path=pca_path)
        mlflow.log_artifact(pca_path)

        # ── Segment assignment + profiles ─────────────────────────────────────
        df, label_map = assign_segment_labels(df, labels, best_k)
        logger.info(f"Segment mapping: {label_map}")

        profiles_path = os.path.join(artifacts_dir, "cluster_profiles.csv")
        build_cluster_profiles(df, feature_cols, profiles_path)
        mlflow.log_artifact(profiles_path)

        # ── Save model ────────────────────────────────────────────────────────
        model_path = os.path.join(models_dir, "trained_model.pkl")
        joblib.dump({"model": final_model, "label_map": label_map, "feature_cols": feature_cols}, model_path)
        mlflow.sklearn.log_model(final_model, artifact_path="kmeans_model")
        logger.info(f"Model saved to {model_path}")

        # Save segment mapping
        mapping_path = os.path.join(artifacts_dir, "segment_mapping.json")
        with open(mapping_path, "w") as f:
            json.dump({str(k): v for k, v in label_map.items()}, f, indent=2)
        mlflow.log_artifact(mapping_path)

        logger.info("Training complete! ✓")
        logger.info(f"  Best K        : {best_k}")
        logger.info(f"  Silhouette    : {sil:.4f}")
        logger.info(f"  Davies-Bouldin: {dbi:.4f}")
        logger.info(f"  Calinski-H    : {chi:.2f}")

        # Save processed df with cluster labels
        labeled_path = os.path.join("data", "labeled_data.csv")
        df.to_csv(labeled_path, index=False)
        logger.info(f"Labeled dataset saved to {labeled_path}")


if __name__ == "__main__":
    train()
