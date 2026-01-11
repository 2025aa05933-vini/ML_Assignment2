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

st.set_page_config(page_title="ML Model Evaluation App")
st.title("📊 ML Model Evaluation App")

st.write(
    "Select a model first, then upload a CSV file to evaluate it."
)

# ---------------- Model selection FIRST ----------------
model_choice = st.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree Classifier",
        "K-Nearest Neighbor Classifier",
        "Naive Bayes - Gaussian",
        "Naive Bayes - Multinomial",
        "Random Forest Classifier",
        "XGBoost Classifier"
    ]
)

MODEL_DIR = "models"
MODEL_PATHS = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree Classifier": "decision_tree.pkl",
    "K-Nearest Neighbor Classifier": "knn.pkl",
    "Naive Bayes - Gaussian": "gaussian_nb.pkl",
    "Naive Bayes - Multinomial": "multinomial_nb.pkl",
    "Random Forest Classifier": "random_forest.pkl",
    "XGBoost Classifier": "xgboost.pkl"
}

model_path = os.path.join(MODEL_DIR, MODEL_PATHS[model_choice])

# ---------------- CSV upload AFTER model ----------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is None:
    st.info("⬆️ Upload a CSV file to continue.")
    st.stop()

df = pd.read_csv(uploaded_file)

st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

# ---------------- Target selection ----------------
target_col = st.selectbox("Select Target Column", df.columns)

X = df.drop(columns=[target_col])
y_true = df[target_col]

# ---------------- Evaluate button ----------------
if st.button("Evaluate Model"):
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        st.stop()

    model = joblib.load(model_path)

    y_pred = model.predict(X)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
        auc_score = roc_auc_score(y_true, y_prob)
    else:
        auc_score = np.nan

    results = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc_score,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }

    st.subheader("📈 Evaluation Metrics")
    st.table(pd.DataFrame(results.items(), columns=["Metric", "Value"]).round(4))
    st.success("✅ Evaluation complete!")
    