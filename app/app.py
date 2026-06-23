"""
app.py
------
Streamlit application for Customer Segmentation.
Upload a CSV, get cluster assignments, visualize segments, download results.
"""

import io
import json
import os
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.predict import SEGMENT_DESCRIPTIONS, predict

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem; border-radius: 0.75rem; color: white;
    text-align: center; margin-bottom: 0.5rem;
}
.segment-badge {
    display: inline-block; padding: 0.3rem 0.8rem;
    border-radius: 9999px; font-weight: 600; font-size: 0.85rem;
}
h1 { color: #1f2937; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/group.png", width=80)
    st.title("⚙️ Controls")
    st.markdown("---")
    uploaded_file = st.file_uploader(
        "📤 Upload Customer CSV",
        type=["csv"],
        help="Upload a CSV with customer features.",
    )
    st.markdown("---")
    st.markdown("### 📖 Segment Guide")
    for seg, info in SEGMENT_DESCRIPTIONS.items():
        st.markdown(f"**{info['icon']} {seg}**")
        st.caption(info["description"])
        st.markdown("")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🎯 Customer Segmentation Dashboard")
st.markdown(
    "Upload your customer data to automatically assign ML-powered segments "
    "and uncover actionable marketing insights."
)
st.markdown("---")

# ── Load model (cached) ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model …")
def load_model_bundle():
    model_path = "models/trained_model.pkl"
    scaler_path = "models/scaler.pkl"
    if not Path(model_path).exists():
        return None, None, None, None
    bundle = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return bundle["model"], scaler, bundle["label_map"], bundle["feature_cols"]


model, scaler, label_map, feature_cols = load_model_bundle()

if model is None:
    st.error("⚠️ No trained model found. Please run `python src/train.py` first.")
    st.stop()

# ── Main content ──────────────────────────────────────────────────────────────
if uploaded_file is None:
    # Landing / demo mode
    st.info("👈 Upload a CSV file in the sidebar to get started.")

    # Load sample data for demo
    sample_path = "data/ecommerce_customer_churn_dataset.csv"
    if Path(sample_path).exists():
        st.markdown("### 🔍 Demo: Using built-in dataset")
        demo_df = pd.read_csv(sample_path, nrows=500)
        result_df = predict(demo_df)
        uploaded_file = None  # flag as demo
        use_demo = True
    else:
        st.stop()
else:
    demo_df = pd.read_csv(uploaded_file)
    result_df = predict(demo_df)
    use_demo = False

# ─────────────────────────────────────────────────────────────────────────────
# KPIs
# ─────────────────────────────────────────────────────────────────────────────
n_customers = len(result_df)
n_segments = result_df["Segment"].nunique()
top_segment = result_df["Segment"].value_counts().idxmax()
avg_ltv = result_df["Lifetime_Value"].mean() if "Lifetime_Value" in result_df.columns else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("👥 Total Customers", f"{n_customers:,}")
col2.metric("🏷️ Segments Found", n_segments)
col3.metric("🏆 Top Segment", top_segment)
col4.metric("💰 Avg Lifetime Value", f"${avg_ltv:,.0f}")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Charts Row 1: Segment distribution + Pie
# ─────────────────────────────────────────────────────────────────────────────
chart1, chart2 = st.columns(2)

with chart1:
    seg_counts = result_df["Segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Count"]
    fig_bar = px.bar(
        seg_counts, x="Segment", y="Count",
        color="Segment", title="Customer Segment Distribution",
        color_discrete_sequence=px.colors.qualitative.Set2,
        text="Count",
    )
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(showlegend=False, height=380)
    st.plotly_chart(fig_bar, use_container_width=True)

with chart2:
    fig_pie = px.pie(
        seg_counts, values="Count", names="Segment",
        title="Segment Share",
        color_discrete_sequence=px.colors.qualitative.Set2,
        hole=0.4,
    )
    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
    fig_pie.update_layout(height=380)
    st.plotly_chart(fig_pie, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PCA Cluster Visualization
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("🔵 PCA Cluster Visualization")

from src.data_download import engineer_features, preprocess_data

try:
    proc = demo_df.copy()
    # Quick preprocessing
    num_cols = proc.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        proc[c] = proc[c].fillna(proc[c].median())
    if "Gender" in proc.columns:
        proc["Gender_Encoded"] = (proc["Gender"] == "Male").astype(int)
    proc = engineer_features(proc)

    for col in feature_cols:
        if col not in proc.columns:
            proc[col] = 0
    proc[feature_cols] = proc[feature_cols].fillna(0)

    X = scaler.transform(proc[feature_cols].values)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    ev = pca.explained_variance_ratio_

    pca_df = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "Segment": result_df["Segment"].values,
        "Cluster": result_df["Cluster"].astype(str).values,
    })

    fig_pca = px.scatter(
        pca_df, x="PC1", y="PC2", color="Segment",
        title=f"PCA Plot — PC1 ({ev[0]:.1%}) × PC2 ({ev[1]:.1%})",
        color_discrete_sequence=px.colors.qualitative.Set2,
        opacity=0.6,
    )
    fig_pca.update_traces(marker=dict(size=5))
    fig_pca.update_layout(height=450)
    st.plotly_chart(fig_pca, use_container_width=True)
except Exception as e:
    st.warning(f"PCA visualization skipped: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Cluster Statistics
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("📊 Cluster Statistics")

numeric_summary_cols = ["Lifetime_Value", "Total_Purchases", "Days_Since_Last_Purchase",
                        "Login_Frequency", "Average_Order_Value", "Membership_Years"]
available_summary = [c for c in numeric_summary_cols if c in result_df.columns]

if available_summary:
    cluster_stats = result_df.groupby("Segment")[available_summary].mean().round(2)
    st.dataframe(
        cluster_stats.style.background_gradient(cmap="YlOrRd", axis=0),
        use_container_width=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Segment Descriptions
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("🏷️ Segment Descriptions & Marketing Strategies")
cols = st.columns(min(n_segments, 3))
for i, seg in enumerate(result_df["Segment"].unique()):
    info = SEGMENT_DESCRIPTIONS.get(seg, {})
    with cols[i % len(cols)]:
        count = (result_df["Segment"] == seg).sum()
        pct = count / n_customers * 100
        st.markdown(f"""
**{info.get('icon','❓')} {seg}**
- 👥 Customers: **{count:,}** ({pct:.1f}%)
- 📝 {info.get('description', '')}
- 🎯 *{info.get('strategy', '')}*
""")

# ─────────────────────────────────────────────────────────────────────────────
# Data Table
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("📋 Customer Assignment Table")
display_cols = ["Segment", "Cluster", "Icon"] + [
    c for c in ["Age", "Lifetime_Value", "Total_Purchases", "Days_Since_Last_Purchase"]
    if c in result_df.columns
]
st.dataframe(result_df[display_cols].head(200), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Download
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("⬇️ Download Results")
csv_buffer = io.StringIO()
result_df.to_csv(csv_buffer, index=False)
st.download_button(
    label="📥 Download Segmented CSV",
    data=csv_buffer.getvalue(),
    file_name="customer_segments.csv",
    mime="text/csv",
)

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation metrics (if available)
# ─────────────────────────────────────────────────────────────────────────────
metrics_path = "artifacts/evaluation_metrics.json"
if Path(metrics_path).exists():
    st.markdown("---")
    st.subheader("📈 Model Evaluation Metrics")
    with open(metrics_path) as f:
        metrics = json.load(f)
    m1, m2, m3 = st.columns(3)
    m1.metric("Silhouette Score", f"{metrics.get('silhouette_score', 0):.4f}", help="Higher is better (max 1.0)")
    m2.metric("Davies-Bouldin Index", f"{metrics.get('davies_bouldin_index', 0):.4f}", help="Lower is better")
    m3.metric("Calinski-Harabasz Score", f"{metrics.get('calinski_harabasz_score', 0):,.1f}", help="Higher is better")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit · Scikit-learn · MLflow | Customer Segmentation ML Project")
