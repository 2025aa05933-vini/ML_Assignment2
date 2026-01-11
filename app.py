import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

st.set_page_config(page_title="Model Evaluation App", layout="centered")

st.title("📊 ML Model Evaluation (PKL Models)")

st.write("""
Upload a CSV file and evaluate **pre-trained models** using:
- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- MCC
""")

# -----------------------------------
# Model Path Mapping
# -----------------------------------
MODEL_DIR = "models"

MODEL_PATHS = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree Classifier": "decision_tree.pkl",
    "K-Nearest Neighbor Classifier": "knn.pkl",
    "Naive Bayes - Gaussian": "naive_bayes_gaussian.pkl",
    "Naive Bayes - Multinomial": "naive_bayes_multinomial.pkl",
    "Random Forest (Ensemble)": "random_forest.pkl",
    "XGBoost (Ensemble)": "xgboost.pkl",
}

# -----------------------------------
# File Upload
# -----------------------------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # -----------------------------------
    # Target Column Selection
    # -----------------------------------
    target_col = st.selectbox("Select Target Column", df.columns)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # -----------------------------------
    # Model Selection
    # -----------------------------------
    model_name = st.selectbox(
        "Select Pre-trained Model",
        list(MODEL_PATHS.keys())
    )

    model_path = os.path.join(MODEL_DIR, MODEL_PATHS[model_name])

    # -----------------------------------
    # Load Model & Evaluate
    # -----------------------------------
    if st.button("🚀 Evaluate Model"):
        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
        else:
            model = joblib.load(model_path)

            y_pred = model.predict(X)

            # AUC handling
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X)[:, 1]
                auc = roc_auc_score(y, y_prob)
            else:
                auc = np.nan

            results = {
                "Accuracy": accuracy_score(y, y_pred),
                "AUC Score": auc,
                "Precision": precision_score(y, y_pred, zero_division=0),
                "Recall": recall_score(y, y_pred, zero_division=0),
                "F1 Score": f1_score(y, y_pred, zero_division=0),
                "MCC Score": matthews_corrcoef(y, y_pred),
            }

            st.subheader("📈 Evaluation Metrics")
            metrics_df = pd.DataFrame.from_dict(
                results, orient="index", columns=["Score"]
            )

            st.dataframe(metrics_df.style.format("{:.4f}"))
