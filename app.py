import streamlit as st
import pandas as pd
import numpy as np
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_curve
)

# -------------------------------------------------
# Streamlit configuration
# -------------------------------------------------
st.set_page_config(
    page_title="F1 Podium Prediction – Model Evaluation",
    layout="wide"
)

st.title("🏎️ F1 Podium Prediction – ML Model Evaluation")
st.caption(
    "Evaluate trained ML models for predicting **podium finishes** in Formula 1 "
    "using robust classification metrics."
)

# -------------------------------------------------
# Configuration
# -------------------------------------------------
TARGET_COL = "podium_finish"

MODEL_FILES = {
    "Logistic Regression": "models/logistic_regression.pkl",
    "Decision Tree": "models/decision_tree.pkl",
    "KNN": "models/knn.pkl",
    "Gaussian Naive Bayes": "models/gaussian_nb.pkl",
    "Random Forest": "models/random_forest.pkl",
    "XGBoost": "models/xgboost.pkl"
}

SAMPLE_FILE_PATH = "data/processed/f1_test.csv"

# -------------------------------------------------
# Sidebar – Model Selection
# -------------------------------------------------
st.sidebar.header("⚙️ Model Settings")

selected_model_name = st.sidebar.selectbox(
    "Select a trained model",
    list(MODEL_FILES.keys())
)

st.sidebar.info(
    "ℹ️ **Test data only** should be uploaded.\n\n"
    "This app evaluates pre-trained models for\n"
    "**binary podium prediction (Yes / No)**."
)

# -------------------------------------------------
# Top Section – Sample + Upload
# -------------------------------------------------
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.subheader("📥 Sample Test File")
    try:
        with open(SAMPLE_FILE_PATH, "rb") as f:
            st.download_button(
                label="Download Sample Test CSV",
                data=f,
                file_name="f1_test_sample.csv",
                mime="text/csv"
            )
    except FileNotFoundError:
        st.warning("Sample test file not found.")

with col2:
    st.subheader("📤 Upload Test Dataset")
    uploaded_file = st.file_uploader(
        "Upload CSV file (TEST DATA ONLY)",
        type=["csv"]
    )

if uploaded_file is None:
    st.info("Please upload a CSV file to proceed.")
    st.stop()

# -------------------------------------------------
# Load & Preview Data
# -------------------------------------------------
df = pd.read_csv(uploaded_file)

st.markdown("---")
st.subheader("🔍 Dataset Preview")
st.dataframe(df.head(), use_container_width=True)

# -------------------------------------------------
# Validation
# -------------------------------------------------
if TARGET_COL not in df.columns:
    st.error(f"Target column '{TARGET_COL}' not found in dataset.")
    st.stop()

X = df.drop(columns=[TARGET_COL])
y_true = df[TARGET_COL]

if y_true.nunique() != 2:
    st.error("Target column must be binary (0 / 1).")
    st.stop()

# -------------------------------------------------
# Why these metrics? (F1-specific explanation)
# -------------------------------------------------
st.markdown("---")
st.info(
    "🏁 **Why F1-Score & MCC?**\n\n"
    "- Podium finishes are **rare events** in Formula 1\n"
    "- Accuracy is misleading due to **class imbalance**\n"
    "- **F1-Score** balances Precision & Recall\n"
    "- **MCC** measures true correlation across all outcomes"
)

# -------------------------------------------------
# Evaluation
# -------------------------------------------------
st.markdown("---")
if st.button("🚀 Evaluate Model", use_container_width=True):

    with st.spinner("Evaluating model..."):
        model = joblib.load(MODEL_FILES[selected_model_name])

        y_pred = model.predict(X)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X)[:, 1]
            auc = roc_auc_score(y_true, y_prob)
        else:
            y_prob = None
            auc = np.nan

        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "AUC": auc,
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1 Score": f1_score(y_true, y_pred, zero_division=0),
            "MCC": matthews_corrcoef(y_true, y_pred)
        }

    # -------------------------------------------------
    # Metrics Cards
    # -------------------------------------------------
    st.subheader("📈 Evaluation Metrics")

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{metrics['Accuracy']:.3f}")
    c2.metric("AUC", f"{metrics['AUC']:.3f}")
    c3.metric("F1 Score", f"{metrics['F1 Score']:.3f}")

    c4, c5, c6 = st.columns(3)
    c4.metric("Precision", f"{metrics['Precision']:.3f}")
    c5.metric("Recall", f"{metrics['Recall']:.3f}")
    c6.metric("MCC", f"{metrics['MCC']:.3f}")

    # -------------------------------------------------
    # Metrics Visualization
    # -------------------------------------------------
    st.markdown("---")
    st.subheader("📊 Metrics Comparison")

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(
        x=list(metrics.keys()),
        y=list(metrics.values()),
        ax=ax
    )

    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Metrics")
    plt.xticks(rotation=25)

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # -------------------------------------------------
    # ROC Curve
    # -------------------------------------------------
    if y_prob is not None:
        st.markdown("---")
        st.subheader("📉 ROC Curve")

        fpr, tpr, _ = roc_curve(y_true, y_prob)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()

        st.pyplot(fig)
        plt.close(fig)

    # -------------------------------------------------
    # Feature Importance (Tree-based models)
    # -------------------------------------------------
    if hasattr(model, "feature_importances_"):
        st.markdown("---")
        st.subheader("🔍 Feature Importance")

        fi_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        st.bar_chart(fi_df.set_index("Feature").head(10))

    # -------------------------------------------------
    # Confusion Matrix
    # -------------------------------------------------
    st.markdown("---")
    st.subheader("🧩 Confusion Matrix")

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix – {selected_model_name}")

    st.pyplot(fig)
    plt.close(fig)

    st.success("✅ Model evaluation completed successfully!")
