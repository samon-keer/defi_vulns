from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

st.set_page_config(layout="wide")

# ============================================================
# PATHS
# ============================================================

DATA_PATH = "/Users/sk/Documents/Liverpool/Dissertation/strict_risk_tfidf_codebert_v1.csv"
VULN_PATH = "/Users/sk/Documents/Liverpool/Dissertation/vuln_heatmap_strict_top15.csv"

# ============================================================
# DATA LOADING
# ============================================================


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load model output data and vulnerability summary data."""
    df = pd.read_csv(DATA_PATH)
    vuln_df = pd.read_csv(VULN_PATH)
    return df, vuln_df


def validate_columns(df: pd.DataFrame, required_columns: list[str], df_name: str) -> None:
    """Stop the app if required columns are missing."""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        st.error(f"Missing required columns in {df_name}: {missing}")
        st.stop()


def get_eval_df(df: pd.DataFrame, model_choice: str) -> pd.DataFrame:
    """Return a clean evaluation dataframe for the selected model."""
    if model_choice == "TF-IDF":
        required = ["contract_id", "true_label", "tfidf_pred_label", "tfidf_risk_prob"]
    elif model_choice == "CodeBERT":
        required = ["contract_id", "true_label", "codebert_pred_label", "codebert_risk_prob"]
    else:
        st.error(f"Unexpected model choice: {model_choice}")
        st.stop()

    validate_columns(df, required, "model output dataset")

    out = df[required].dropna().copy()
    out.columns = ["contract_id", "true_label", "pred_label", "risk_prob"]
    out["true_label"] = out["true_label"].astype(int)
    out["pred_label"] = out["pred_label"].astype(int)

    return out


# ============================================================
# LOAD AND VALIDATE
# ============================================================

df, vuln_df = load_data()

if df.empty:
    st.error("Model output dataset is empty. Check the CSV export.")
    st.stop()

if vuln_df.empty:
    st.error("Vulnerability summary dataset is empty. Check the CSV export.")
    st.stop()

validate_columns(
    df,
    [
        "contract_id",
        "true_label",
        "tfidf_pred_label",
        "tfidf_risk_prob",
        "codebert_pred_label",
        "codebert_risk_prob",
    ],
    "model output dataset",
)

validate_columns(vuln_df, ["check_name", "high_risk"], "vulnerability summary dataset")

# ============================================================
# TITLE AND SIDEBAR
# ============================================================

st.title("AI-Driven Risk Detection Dashboard for DeFi Smart Contracts")

st.sidebar.header("Governance Filters")
model_choice = st.sidebar.selectbox("Select Risk Model", ["TF-IDF", "CodeBERT"])
show_high_risk_only = st.sidebar.checkbox("Show High-Risk Only")

# ============================================================
# MODEL-SPECIFIC DATA
# ============================================================

eval_df = get_eval_df(df, model_choice)

display_df = df.copy()
if model_choice == "TF-IDF":
    display_df["risk_pred"] = display_df["tfidf_pred_label"]
else:
    display_df["risk_pred"] = display_df["codebert_pred_label"]

filtered_df = display_df.copy()
if show_high_risk_only:
    filtered_df = filtered_df[filtered_df["risk_pred"] == 1]

# ============================================================
# KPI ROW
# ============================================================

total_contracts = len(display_df)
high_risk = pd.to_numeric(display_df["risk_pred"], errors="coerce").fillna(0).sum()
low_risk = total_contracts - high_risk
risk_rate = high_risk / total_contracts if total_contracts > 0 else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Contracts Analysed", total_contracts)
col2.metric("High Risk Detected", int(high_risk))
col3.metric("Low Risk", int(low_risk))
col4.metric("Risk Rate", f"{risk_rate:.2%}")

# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.subheader("Model Performance")

acc = accuracy_score(eval_df["true_label"], eval_df["pred_label"])
prec = precision_score(eval_df["true_label"], eval_df["pred_label"], zero_division=0)
rec = recall_score(eval_df["true_label"], eval_df["pred_label"], zero_division=0)
f1 = f1_score(eval_df["true_label"], eval_df["pred_label"], zero_division=0)
auc = roc_auc_score(eval_df["true_label"], eval_df["risk_prob"])

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Accuracy", f"{acc:.3f}")
m2.metric("Precision", f"{prec:.3f}")
m3.metric("Recall", f"{rec:.3f}")
m4.metric("F1 Score", f"{f1:.3f}")
m5.metric("AUC-ROC", f"{auc:.3f}")

# ============================================================
# CONFUSION MATRIX AND ROC
# ============================================================

st.subheader("Confusion Matrix and ROC Curve")

c1, c2 = st.columns(2)

with c1:
    cm = confusion_matrix(eval_df["true_label"], eval_df["pred_label"])
    cm_df = pd.DataFrame(
        cm,
        index=["Actual Low Risk", "Actual High Risk"],
        columns=["Predicted Low Risk", "Predicted High Risk"],
    )

    fig_cm = px.imshow(
        cm_df,
        text_auto=True,
        aspect="auto",
        title=f"{model_choice} Confusion Matrix",
    )
    st.plotly_chart(fig_cm, width="stretch")

with c2:
    fpr, tpr, _ = roc_curve(eval_df["true_label"], eval_df["risk_prob"])

    fig_roc = go.Figure()
    fig_roc.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"{model_choice} (AUC = {auc:.3f})",
        )
    )
    fig_roc.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Baseline",
            line=dict(dash="dash"),
        )
    )
    fig_roc.update_layout(
        title=f"{model_choice} ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
    )
    st.plotly_chart(fig_roc, width="stretch")

# ============================================================
# TOP 10 VULNERABILITY CHECKS
# ============================================================

st.subheader("Top 10 High-Risk Vulnerability Checks")

top10_vuln = vuln_df.sort_values("high_risk", ascending=False).head(10)

fig_top10 = px.bar(
    top10_vuln,
    x="high_risk",
    y="check_name",
    orientation="h",
    title="Top 10 Vulnerability Checks by High-Risk Count",
)
fig_top10.update_layout(yaxis={"categoryorder": "total ascending"})
st.plotly_chart(fig_top10, width="stretch")

# ============================================================
# TOP 10 HIGHEST-RISK CONTRACTS
# ============================================================

st.subheader("Top 10 Highest-Risk Contracts")

top10_contracts = eval_df.sort_values("risk_prob", ascending=False).head(10)
st.dataframe(
    top10_contracts[["contract_id", "true_label", "pred_label", "risk_prob"]],
    width="stretch",
)

# ============================================================
# CONTRACT TABLE
# ============================================================

st.subheader("Contract-Level Governance Risk View")

st.dataframe(
    filtered_df[
        [
            "contract_id",
            "true_label",
            "tfidf_pred_label",
            "tfidf_risk_prob",
            "codebert_pred_label",
            "codebert_risk_prob",
        ]
    ],
    width="stretch",
)

# ============================================================
# GOVERNANCE INSIGHTS
# ============================================================

st.subheader("Governance Insights")

st.markdown(
    """
**Key Findings**

- AI models can identify high-risk smart contracts prior to deployment.
- TF-IDF and CodeBERT produce broadly comparable detection performance.
- Fine-tuned transformer models may overfit when labelled datasets are small.
- Graph-based learning showed weaker performance in this experimental setting.

**Governance Implications**

- Automated contract screening before DAO voting
- Risk-weighted smart contract approval pipelines
- AI-assisted audit prioritisation
- Regulatory risk monitoring for DeFi platforms
"""
)