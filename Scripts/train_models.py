import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
DATA_PATH = "data/processed/f1_train.csv"
TARGET_COL = "podium_finish"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------------------------------
# Load dataset
# -------------------------------------------------
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# -------------------------------------------------
# Train-test split
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------------------------
# Define models (deployment-safe)
# -------------------------------------------------
models = {
    "logistic_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),

    "decision_tree": DecisionTreeClassifier(
        random_state=42
    ),

    "knn": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ]),

    "gaussian_nb": Pipeline([
        ("scaler", StandardScaler()),
        ("model", GaussianNB())
    ]),

    "random_forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "xgboost": xgb.XGBClassifier(
        eval_metric="logloss",
        random_state=42
    )
}

# -------------------------------------------------
# Train & save models
# -------------------------------------------------
for name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, f"{MODEL_DIR}/{name}.pkl")
    print(f"✅ Saved {name}.pkl")

print("\n🎉 All models trained and saved successfully!")
