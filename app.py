from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from src.fraud_detection import AnomalyFraudPipeline, load_dataset, make_plots, results_frame, split_data

st.set_page_config(page_title="Fraud Anomaly Lab", page_icon="💳", layout="wide")
st.title("Credit Card Fraud Detection")
st.caption("Anomaly detection trained on normal transactions only")

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload creditcard.csv", type="csv")
    default_path = Path("data/creditcard.csv")
    model_path = Path("artifacts/best_model.joblib")
    use_attached = default_path.exists()
    st.info("Expected columns: Time, V1–V28, Amount, Class")
    if use_attached:
        st.success("Attached dataset detected")
    retrain = st.checkbox("Retrain models", value=False, help="Leave unchecked to reuse artifacts/built-in Streamlit cache for a fast dashboard.")

if uploaded is None and not use_attached:
    st.warning("Upload the ULB/Kaggle credit-card dataset to begin.")
    st.stop()

@st.cache_data(show_spinner="Loading and validating data…")
def load_uploaded(raw: bytes) -> pd.DataFrame:
    import io
    return load_dataset(io.BytesIO(raw))

@st.cache_data(show_spinner="Loading attached dataset…")
def load_default(path: str) -> pd.DataFrame:
    return load_dataset(path)

try:
    df = load_uploaded(uploaded.getvalue()) if uploaded is not None else load_default(str(default_path))
except Exception as exc:
    st.error(f"Could not load the CSV: {exc}")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Transactions", f"{len(df):,}")
c2.metric("Fraud cases", f"{int(df.Class.sum()):,}")
c3.metric("Fraud rate", f"{df.Class.mean():.3%}")
c4.metric("Median amount", f"${df.Amount.median():,.2f}")

with st.expander("Dataset preview"):
    st.dataframe(df.head(10), use_container_width=True)

@st.cache_resource(show_spinner="Training three anomaly models on normal transactions…")
def train_pipeline(normal_train: pd.DataFrame) -> AnomalyFraudPipeline:
    return AnomalyFraudPipeline().fit(normal_train)


normal_train, test_df = split_data(df)
pipeline = None
if uploaded is None and model_path.exists() and not retrain:
    try:
        pipeline = AnomalyFraudPipeline.load(model_path)
        st.sidebar.success("Loaded saved model artifact (fast mode)")
    except Exception as exc:
        st.sidebar.warning(f"Saved model could not be loaded; training instead ({exc})")

if pipeline is None:
    pipeline = train_pipeline(normal_train)
    st.sidebar.info("Models trained and cached for this session")

results = pipeline.evaluate(test_df)

st.subheader("Model comparison")
metrics = results_frame(results)
st.dataframe(metrics.style.format({k: "{:.4f}" for k in ["precision", "recall", "f1", "roc_auc", "pr_auc"]}), use_container_width=True, hide_index=True)
best = pipeline.best_model_name()
st.success(f"Recommended model by PR-AUC: **{best}**")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Confusion matrix")
    selected = st.selectbox("Model", list(results), index=list(results).index(best))
    from sklearn.metrics import confusion_matrix
    st.dataframe(pd.DataFrame(confusion_matrix(test_df.Class, results[selected].predictions), index=["Actual normal", "Actual fraud"], columns=["Predicted normal", "Predicted fraud"]), use_container_width=True)
with col2:
    st.subheader("Fraud review queue")
    scored = test_df[["Time", "Amount", "Class"]].copy()
    scored["anomaly_score"] = results[selected].scores
    scored["prediction"] = results[selected].predictions
    st.dataframe(scored.sort_values("anomaly_score", ascending=False).head(20), use_container_width=True, hide_index=True)

st.subheader("Visual diagnostics")
@st.cache_data(show_spinner=False)
def render_plots(df: pd.DataFrame, _pipeline: AnomalyFraudPipeline):
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        make_plots(_pipeline, df, tmp)
        return {p.name: p.read_bytes() for p in Path(tmp).glob("*.png")}

plots = render_plots(test_df, pipeline)
for name, label in [("model_comparison.png", "Model comparison"), ("confusion_matrices.png", "Confusion matrices"), ("score_distributions.png", "Anomaly score distributions"), ("pca_projection.png", "PCA projection")]:
    st.image(plots[name], caption=label, use_container_width=True)

st.download_button("Download model metrics", metrics.to_csv(index=False), "metrics.csv", "text/csv")
