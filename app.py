"""
BITS Pilani WILP - M.Tech (DSE) Machine Learning Assignment 2
File: app.py
Description: Interactive Streamlit Web Application for Automotive OBD-II Engine &
             Powertrain Fault Classification demonstrating 5 ML Models, CSV Test
             Upload, 6 Evaluation Metrics, Confusion Matrix, ROC Curve, and Report.
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report, roc_curve
)

# 1. Page Configuration & Custom Automotive UI Styling
st.set_page_config(
    page_title="AutoPulse ML | OBD-II Powertrain Diagnostics",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .auto-banner {
        background: linear-gradient(90deg, #0F2C59 0%, #1F4E78 100%);
        padding: 1.4rem 1.8rem;
        border-radius: 10px;
        border-left: 6px solid #C84B31;
        color: white;
        margin-bottom: 1.2rem;
    }
    .auto-banner h1 { color: #FFFFFF; font-size: 1.75rem; margin-bottom: 0.2rem; }
    .auto-banner p { color: #E2E8F0; font-size: 0.95rem; margin: 0; }
    div[data-testid="stMetric"] {
        background-color: #F8FAFC;
        border: 1px solid #CBD5E1;
        border-left: 5px solid #0F2C59;
        padding: 12px 16px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="auto-banner">
    <h1>🏎️ AutoPulse ML: OBD-II Engine & Powertrain Fault Diagnostics</h1>
    <p>BITS Pilani WILP — M.Tech (DSE) Machine Learning Assignment 2 | Multi-Model Telemetry Evaluation Workbench</p>
</div>
""", unsafe_allow_html=True)

# 2. Model Registry & Cached Loader
MODEL_FILES = {
    "Support Vector Machine (RBF Kernel) — Winner": "model/svm_classifier.pkl",
    "k-Nearest Neighbors (kNN, k=7)": "model/knn_classifier.pkl",
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree Classifier": "model/decision_tree.pkl",
    "Naive Bayes (Gaussian NB)": "model/naive_bayes.pkl"
}

@st.cache_resource
def load_trained_models():
    loaded = {}
    for label, path in MODEL_FILES.items():
        if os.path.exists(path):
            loaded[label] = joblib.load(path)
    return loaded

models_dict = load_trained_models()

# 3. Sidebar: (a) Dataset Upload Option (CSV) & (b) Model Selection Dropdown
with st.sidebar:
    st.header("🔧 ECU Diagnostic Controls")
    st.subheader("1. Upload Test Telemetry (CSV)")
    uploaded_file = st.file_uploader(
        "Upload test_data.csv (with target 'engine_fault_status')",
        type=["csv"],
        help="Upload the hold-out test_data.csv file. If no file is uploaded, the repository default test_data.csv (320 vehicles) is loaded automatically."
    )

    st.divider()
    st.subheader("2. Select Classification Model")
    selected_model_label = st.selectbox(
        "Choose an ML Model to Evaluate:",
        options=list(MODEL_FILES.keys()),
        index=0
    )

    st.divider()
    st.caption("**Automotive Telemetry Specs:** 14 OBD-II / CAN-Bus Sensor Channels | Target: `engine_fault_status` (0 = Normal, 1 = Fault)")

# 4. Load Test Data (Uploaded CSV or Default Repository Fallback)
if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    data_source_msg = f"✅ Evaluating uploaded test file: `{uploaded_file.name}` ({len(test_df)} vehicle snapshots)"
elif os.path.exists("test_data.csv"):
    test_df = pd.read_csv("test_data.csv")
    data_source_msg = f"ℹ️ Evaluating repository default `test_data.csv` ({len(test_df)} vehicle snapshots). Upload a custom CSV in the sidebar anytime."
else:
    st.error("Neither uploaded CSV nor default 'test_data.csv' was found. Please run `python model/train_models.py` first.")
    st.stop()

st.info(data_source_msg)

target_col = "engine_fault_status"
if target_col not in test_df.columns:
    st.error(f"Uploaded CSV must contain the target column `'{target_col}'` (0 = Normal, 1 = Fault) along with the 14 OBD-II feature columns.")
    st.stop()

X_test = test_df.drop(columns=[target_col])
y_test = test_df[target_col]

if not models_dict:
    st.error("Trained model files (`model/*.pkl`) not found. Please run `python model/train_models.py`.")
    st.stop()

# 5. Evaluate Selected Model
active_model = models_dict[selected_model_label]
y_pred = active_model.predict(X_test)
y_prob = active_model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
mcc = matthews_corrcoef(y_test, y_pred)

# (c) Display of Evaluation Metrics (6 Mandatory Metrics)
st.subheader(f"📊 Evaluation Metrics on Test Data — {selected_model_label}")
m_cols = st.columns(6)
m_cols[0].metric("1. Accuracy", f"{acc:.4f}", f"{acc*100:.2f}%")
m_cols[1].metric("2. AUC Score", f"{auc:.4f}")
m_cols[2].metric("3. Precision", f"{prec:.4f}")
m_cols[3].metric("4. Recall", f"{rec:.4f}")
m_cols[4].metric("5. F1 Score", f"{f1:.4f}")
m_cols[5].metric("6. MCC Score", f"{mcc:.4f}")

st.divider()

# (d) Confusion Matrix, ROC Curve & Classification Report Tabs
tab1, tab2, tab3 = st.tabs([
    "🎯 Confusion Matrix & Classification Report",
    "🏆 All 5 Models Comparison Leaderboard",
    "🔍 OBD-II Telemetry Explorer & Predictions"
])

with tab1:
    col_left, col_right = st.columns([1.05, 1.15])
    with col_left:
        st.markdown("#### Confusion Matrix Heatmap")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(4.8, 3.8), dpi=150)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax_cm,
            annot_kws={"size": 13, "weight": "bold"},
            xticklabels=["Predicted Normal (0)", "Predicted Fault (1)"],
            yticklabels=["Actual Normal (0)", "Actual Fault (1)"]
        )
        ax_cm.set_title(selected_model_label.split("—")[0].strip(), fontsize=11, fontweight="bold", color="#0F2C59")
        st.pyplot(fig_cm)

    with col_right:
        st.markdown("#### Detailed Classification Report")
        report_dict = classification_report(
            y_test, y_pred,
            target_names=["Class 0: Normal Engine", "Class 1: Powertrain Fault"],
            output_dict=True
        )
        report_df = pd.DataFrame(report_dict).transpose().round(4)
        st.dataframe(report_df, use_container_width=True)

        st.markdown("#### Receiver Operating Characteristic (ROC Curve)")
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig_roc, ax_roc = plt.subplots(figsize=(4.8, 2.7), dpi=150)
        ax_roc.plot(fpr, tpr, color="#C84B31", lw=2.2, label=f"AUC = {auc:.4f}")
        ax_roc.plot([0, 1], [0, 1], linestyle="--", color="#888888", lw=1.1)
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate (Recall)")
        ax_roc.legend(loc="lower right")
        ax_roc.grid(alpha=0.3)
        st.pyplot(fig_roc)

with tab2:
    st.markdown("#### Side-by-Side Evaluation Metrics Across All 5 Models on Current Test Data")
    all_rows = []
    for m_label, m_pipe in models_dict.items():
        p_y = m_pipe.predict(X_test)
        p_prob = m_pipe.predict_proba(X_test)[:, 1]
        all_rows.append({
            "ML Model Name": m_label.split("—")[0].strip(),
            "Accuracy": round(accuracy_score(y_test, p_y), 4),
            "AUC": round(roc_auc_score(y_test, p_prob), 4),
            "Precision": round(precision_score(y_test, p_y, zero_division=0), 4),
            "Recall": round(recall_score(y_test, p_y, zero_division=0), 4),
            "F1": round(f1_score(y_test, p_y, zero_division=0), 4),
            "MCC": round(matthews_corrcoef(y_test, p_y), 4)
        })
    leaderboard_df = pd.DataFrame(all_rows)
    st.dataframe(leaderboard_df, use_container_width=True, hide_index=True)

with tab3:
    st.markdown("#### Test Dataset Preview with Model Predictions")
    preview_df = test_df.copy()
    preview_df["Predicted_Status"] = y_pred
    preview_df["Fault_Probability"] = np.round(y_prob, 4)
    st.dataframe(preview_df.head(50), use_container_width=True)