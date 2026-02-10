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
    confusion_matrix
)

# -------------------------------------------------
# Streamlit configuration
# -------------------------------------------------
st.set_page_config(
    page_title="ML Model Evaluation App",
    layout="centered"
)

st.title("ML Model Evaluation App")
st.write(
    "Upload **test data only** (CSV), select a trained model, "
    "and view evaluation metrics and visualizations."
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

# -------------------------------------------------
# Sample Test File Download (BEFORE user input)
# -------------------------------------------------
SAMPLE_FILE_PATH = "data/processed/f1_test.csv"

st.subheader("Sample Test File")

try:
    with open(SAMPLE_FILE_PATH, "rb") as f:
        st.download_button(
            label="Download Sample Test CSV",
            data=f,
            file_name="f1_test_sample.csv",
            mime="text/csv"
        )
except FileNotFoundError:
    st.warning(
        f"Sample file not found at `{SAMPLE_FILE_PATH}`. "
        "Please ensure the file exists."
    )

# -------------------------------------------------
# Dataset upload (Requirement a)
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload CSV file (TEST DATA ONLY)",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload a CSV file to continue.")
    st.stop()

df = pd.read_csv(uploaded_file)

st.subheader("Uploaded Dataset Preview")
st.dataframe(df.head())

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
# Model selection dropdown (Requirement b)
# -------------------------------------------------
st.subheader("Model Selection")

selected_model_name = st.selectbox(
    "Select a trained model",
    list(MODEL_FILES.keys())
)

# -------------------------------------------------
# Evaluation
# -------------------------------------------------
if st.button("Evaluate Model"):

    model_path = MODEL_FILES[selected_model_name]
    model = joblib.load(model_path)

    # Predictions
    y_pred = model.predict(X)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y_true, y_prob)
    else:
        auc = np.nan

    # -------------------------------------------------
    # Metrics table (Requirement c)
    # -------------------------------------------------
    st.subheader("Evaluation Metrics")

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred)
    }

    metrics_df = (
        pd.DataFrame.from_dict(metrics, orient="index", columns=["Value"])
        .round(4)
    )

    st.table(metrics_df)

    # -------------------------------------------------
    # Metrics visualization (Matplotlib + Seaborn)
    # -------------------------------------------------
    st.subheader("Metrics Visualization")

    fig, ax = plt.subplots(figsize=(6, 3))
    sns.barplot(
        x=list(metrics.keys()),
        y=list(metrics.values()),
        ax=ax
    )

    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Evaluation Metrics")
    plt.xticks(rotation=30)

    st.pyplot(fig)
    plt.close(fig)

    # -------------------------------------------------
    # Confusion Matrix (Requirement d)
    # -------------------------------------------------
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(4, 3))
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

    st.success("Model evaluation completed successfully!")
