import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

# -------------------------------------------------
# Streamlit config
# -------------------------------------------------
st.set_page_config(
    page_title="ML Model Evaluation App",
    layout="centered"
)

st.title("ML Model Evaluation App")
st.write(
    "Upload a raw CSV file and evaluate all trained models."
)

# -------------------------------------------------
# Config
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
# CSV upload
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload CSV file (RAW data)",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload a CSV file to continue.")
    st.stop()

df = pd.read_csv(uploaded_file)

st.subheader("Dataset Preview")
st.dataframe(df.head())

# -------------------------------------------------
# Validation
# -------------------------------------------------
if TARGET_COL not in df.columns:
    st.error(f"Required target column '{TARGET_COL}' not found.")
    st.stop()

X = df.drop(columns=[TARGET_COL])
y_true = df[TARGET_COL]

if y_true.nunique() != 2:
    st.error("Target column must be binary (0/1).")
    st.stop()

# -------------------------------------------------
# Evaluate all models
# -------------------------------------------------
if st.button("Evaluate Models"):

    results = []

    for model_name, model_path in MODEL_FILES.items():

        model = joblib.load(model_path)

        # RAW data is passed – preprocessing happens inside the pipeline
        y_pred = model.predict(X)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X)[:, 1]
            auc = roc_auc_score(y_true, y_prob)
        else:
            auc = np.nan

        results.append({
            "Model": model_name,
            "Accuracy": accuracy_score(y_true, y_pred),
            "AUC": auc,
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1 Score": f1_score(y_true, y_pred, zero_division=0),
            "MCC": matthews_corrcoef(y_true, y_pred)
        })

    results_df = pd.DataFrame(results).round(4)

    # -------------------------------------------------
    # Display results
    # -------------------------------------------------
    st.subheader("Evaluation Results")
    st.dataframe(results_df)

    # -------------------------------------------------
    # Save output CSV
    # -------------------------------------------------
    results_df.to_csv("output.csv", index=False)

    st.success("Evaluation completed successfully!")
    st.write("Results saved as `output.csv`")

    with open("output.csv", "rb") as f:
        st.download_button(
            "Download output.csv",
            f,
            file_name="output.csv",
            mime="text/csv"
        )
    