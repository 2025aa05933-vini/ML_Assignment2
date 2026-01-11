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

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(page_title="ML Model Evaluation", layout="centered")
st.title("📊 ML Model Evaluation App")

st.write(
    """
    Upload a CSV file, select a trained model, and evaluate it using
    Accuracy, AUC, Precision, Recall, F1 Score, and MCC.
    """
)

# -------------------------------------------------
# Model Selection
# -------------------------------------------------
model_choice = st.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree Classifier",
        "K-Nearest Neighbor Classifier",
        "Naive Bayes - Multinomial",
        "Random Forest Classifier",
        "XGBoost Classifier"
    ]
)

# -------------------------------------------------
# Model Paths
# -------------------------------------------------
MODEL_DIR = "models"

MODEL_PATHS = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree Classifier": "decision_tree.pkl",
    "K-Nearest Neighbor Classifier": "knn.pkl",
    "Naive Bayes - Multinomial": "multinomial_nb.pkl",
    "Random Forest Classifier": "random_forest.pkl",
    "XGBoost Classifier": "xgboost.pkl"
}

MODEL_PATH = os.path.join(MODEL_DIR, MODEL_PATHS[model_choice])

# -------------------------------------------------
# Load Model
# -------------------------------------------------
@st.cache_resource
def load_model(path):
    return joblib.load(path)

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"❌ Could not load model: {e}")
    st.stop()

# -------------------------------------------------
# Upload CSV
# -------------------------------------------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is None:
    st.info("⬆️ Upload a CSV file to continue.")
    st.stop()

df = pd.read_csv(uploaded_file)

st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

# -------------------------------------------------
# Target Selection
# -------------------------------------------------
target_col = st.selectbox("Select Target Column", df.columns)

X = df.drop(columns=[target_col])
y_true = df[target_col]

# -------------------------------------------------
# Prediction
# -------------------------------------------------
try:
    y_pred = model.predict(X)

    # AUC Handling
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
        auc_score = roc_auc_score(y_true, y_prob)
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X)
        auc_score = roc_auc_score(y_true, y_score)
    else:
        auc_score = np.nan

except Exception as e:
    st.error(f"❌ Prediction failed: {e}")
    st.stop()

# -------------------------------------------------
# Metrics Calculation
# -------------------------------------------------
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
mcc = matthews_corrcoef(y_true, y_pred)

# -------------------------------------------------
# Display Metrics
# -------------------------------------------------
st.subheader("📈 Evaluation Metrics")

metrics_df = pd.DataFrame(
    {
        "Metric": [
            "Accuracy",
            "AUC Score",
            "Precision",
            "Recall",
            "F1 Score",
            "Matthews Correlation Coefficient"
        ],
        "Value": [
            accuracy,
            auc_score,
            precision,
            recall,
            f1,
            mcc
        ],
    }
)

st.table(metrics_df.round(4))
st.success("✅ Model evaluation completed successfully!")   