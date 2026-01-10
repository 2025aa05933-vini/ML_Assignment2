# app.py
import streamlit as st
import pandas as pd
from pathlib import Path

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="F1 ML Assignment",
    layout="wide"
)

st.title("🏎️ Formula 1 – ML Assignment App")
st.caption("Streamlit app scaffold (model-ready)")

# ---------------------------
# Paths (future-proof)
# ---------------------------
DATA_DIR = Path("data/processed")

# ---------------------------
# Data loading
# ---------------------------
@st.cache_data
def load_data(filename):
    path = DATA_DIR / filename
    if not path.exists():
        st.error(f"File not found: {path}")
        return None
    return pd.read_csv(path)

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Dataset Overview", "Feature Inspection", "Model (Coming Soon)"]
)

# ---------------------------
# Pages
# ---------------------------
if page == "Dataset Overview":
    st.subheader("📊 Processed Datasets")

    train_df = load_data("f1_train.csv")
    test_df = load_data("f1_test.csv")

    if train_df is not None:
        st.markdown("### Training Data")
        st.write(f"Shape: {train_df.shape}")
        st.dataframe(train_df.head())

    if test_df is not None:
        st.markdown("### Test Data")
        st.write(f"Shape: {test_df.shape}")
        st.dataframe(test_df.head())

elif page == "Feature Inspection":
    st.subheader("🔍 Feature Inspection")

    df = load_data("f1_train.csv")

    if df is not None:
        feature = st.selectbox("Select a feature", df.columns)

        st.write("Summary statistics")
        st.write(df[feature].describe())

        st.write("Missing values")
        st.write(df[feature].isna().sum())

elif page == "Model (Coming Soon)":
    st.subheader("🤖 Model Status")

    st.warning("Model not trained yet.")

    st.markdown("""
    **Planned steps:**
    - Train model using `Scripts/prepare_data.py`
    - Save model to `model/`
    - Load model here using `joblib`
    - Enable predictions
    """)

    st.info("App structure is ready for model integration.")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.caption("BITS WILP – AIML | ML Assignment")
st.caption("Developed by Vineet Puram")