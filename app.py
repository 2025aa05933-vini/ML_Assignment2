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

# -------------------------------------------------
# F1 Dashboard CSS (Aesthetic only)
# -------------------------------------------------
st.markdown("""
<style>
/* Page padding */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Metric cards styling */
div[data-testid="metric-container"] {
    background-color: #1C1C1C;
    border: 1px solid #E10600;
    padding: 16px;
    border-radius: 12px;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: #0F0F0F;
}

/* Center plots */
.plot-container {
    display: flex;
    justify-content: center;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# App Header (F1 branding)
# -------------------------------------------------
st.markdown("## 🏁 F1 Podium Prediction Dashboard")
st.caption(
    "Formula 1 • Machine Learning Model Evaluation • "
    "Pre-trained models evaluated on unseen race data"
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
# Sidebar – Controls (Clean & minimal)
# -------------------------------------------------
st.sidebar.markdown("## ⚙️ Race Controls")
st.sidebar.markdown("---")

selected_model_name = st.sidebar.selectbox(
    "Select Model",
    list(MODEL_FILES.keys())
)

st.sidebar.caption(
    "Only **test data** should be uploaded.\n\n"
    "This dashboard evaluates **binary podium predictions** "
    "(Podium / No Podium)."
)

# -------------------------------------------------
# Input Section – Sample + Upload
# -------------------------------------------------
st.divider()
st.subheader("📥 Input Data")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Sample Test File**")
    try:
        with open(SAMPLE_FILE_PATH, "rb") as f:
            st.download_button(
                "Download Sample CSV",
                data=f,
                file_name="f1_test_sample.csv",
                mime="text/csv"
            )
    except FileNotFoundError:
        st.warning("Sample test file not found.")

with col2:
    st.markdown("**Upload Test Dataset**")
    uploaded_file = st.file_uploader(
        "CSV file (test data only)",
        type=["csv"]
    )

if uploaded_file is None:
    st.info("Upload a test CSV file to begin evaluation.")
    st.stop()

# -------------------------------------------------
# Data Preview
# -------------------------------------------------
df = pd.read_csv(uploaded_file)

st.divider()
st.subheader("🔍 Race Data Snapshot")
st.caption("Preview of uploaded test dataset (first 5 rows)")
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
# Metric Justification (F1-aware)
# -------------------------------------------------
st.divider()
st.subheader("🏁 Metric Rationale")
st.caption(
    "Podium prediction is an **imbalanced classification problem**.\n\n"
    "• Accuracy can be misleading\n"
    "• F1-score balances precision & recall\n"
    "• MCC captures overall prediction quality"
)

# -------------------------------------------------
# Evaluation Trigger
# -------------------------------------------------
st.divider()
if st.button("🏎️ Run Model Evaluation", use_container_width=True):

    with st.spinner("Running race simulation..."):
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
    # Metric Cards
    # -------------------------------------------------
    st.subheader("📈 Race Outcome Metrics")

    r1, r2, r3 = st.columns(3)
    r1.metric("Accuracy", f"{metrics['Accuracy']:.3f}")
    r2.metric("AUC", f"{metrics['AUC']:.3f}")
    r3.metric("F1 Score", f"{metrics['F1 Score']:.3f}")

    r4, r5, r6 = st.columns(3)
    r4.metric("Precision", f"{metrics['Precision']:.3f}")
    r5.metric("Recall", f"{metrics['Recall']:.3f}")
    r6.metric("MCC", f"{metrics['MCC']:.3f}")

    # -------------------------------------------------
    # Metrics Comparison Chart
    # -------------------------------------------------
    st.divider()
    st.subheader("📊 Performance Comparison")

    _, mid, _ = st.columns([1, 6, 1])
    with mid:
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
        st.pyplot(fig)
        plt.close(fig)

    # -------------------------------------------------
    # ROC Curve
    # -------------------------------------------------
    if y_prob is not None:
        st.divider()
        st.subheader("📉 Podium Probability Curve (ROC)")

        _, mid, _ = st.columns([1, 5, 1])
        with mid:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
            ax.plot([0, 1], [0, 1], linestyle="--")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

    # -------------------------------------------------
    # Feature Importance
    # -------------------------------------------------
    if hasattr(model, "feature_importances_"):
        st.divider()
        st.subheader("🔧 Performance Drivers")

        fi_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        st.bar_chart(fi_df.set_index("Feature").head(10))

    # -------------------------------------------------
    # Confusion Matrix
    # -------------------------------------------------
    st.divider()
    st.subheader("🧩 Prediction Breakdown")

    cm = confusion_matrix(y_true, y_pred)

    _, mid, _ = st.columns([1, 4, 1])
    with mid:
        fig, ax = plt.subplots(figsize=(4, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        plt.close(fig)

    st.success("✅ Model evaluation completed successfully!")

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.divider()
st.caption("🏎️ Formula 1 Podium Prediction • ML Evaluation Dashboard")
