import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

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
# Streamlit config
# -------------------------------------------------
st.set_page_config(
    page_title="ML Model Evaluation App",
    layout="centered"
)

st.title("📊 ML Model Evaluation App")
st.write(
    "Select a trained model, upload a CSV file, and evaluate it on binary classification metrics."
)

# -------------------------------------------------
# Model selection
# -------------------------------------------------
MODEL_DIR = "models"

MODEL_PATHS = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree Classifier": "decision_tree.pkl",
    "K-Nearest Neighbor Classifier": "knn.pkl",
    "Naive Bayes - Gaussian": "gaussian_nb.pkl",
    "Random Forest Classifier": "random_forest.pkl",
    "XGBoost Classifier": "xgboost.pkl"
}

model_choice = st.selectbox(
    "Select Model",
    list(MODEL_PATHS.keys())
)

model_path = os.path.join(MODEL_DIR, MODEL_PATHS[model_choice])

# -------------------------------------------------
# CSV upload
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

if uploaded_file is None:
    st.info("⬆️ Please upload a CSV file to continue.")
    st.stop()

df = pd.read_csv(uploaded_file)

st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

# -------------------------------------------------
# Fixed target column
# -------------------------------------------------
TARGET_COL = "podium_finish"

if TARGET_COL not in df.columns:
    st.error(f"❌ Required target column '{TARGET_COL}' not found in dataset.")
    st.stop()

X = df.drop(columns=[TARGET_COL])
y_true = df[TARGET_COL]

# -------------------------------------------------
# Basic validation
# -------------------------------------------------
if y_true.nunique() != 2:
    st.error("❌ Target column must be binary (0/1).")
    st.stop()

# -------------------------------------------------
# Evaluate model
# -------------------------------------------------
if st.button("🚀 Evaluate Model"):

    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        st.stop()

    model = joblib.load(model_path)

    # Predictions
    y_pred = model.predict(X)

    # Probabilities (if supported)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y_true, y_prob)
    else:
        auc = np.nan

    # Metrics
    results = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred)
    }

    st.subheader("📈 Evaluation Metrics")
    st.table(
        pd.DataFrame(results.items(), columns=["Metric", "Value"]).round(4)
    )

    # -------------------------------------------------
    # Confusion Matrix
    # -------------------------------------------------
    cm = confusion_matrix(y_true, y_pred)

    st.subheader("🧩 Confusion Matrix")

    cm_df = pd.DataFrame(
        cm,
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"]
    )

    st.table(cm_df)

    # Heatmap
    fig, ax = plt.subplots()
    ax.imshow(cm)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")

    st.pyplot(fig)

    st.success("✅ Evaluation completed successfully!")
