import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

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

# -------------------------------------------------
# Load dataset
# -------------------------------------------------
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# -------------------------------------------------
# Define feature groups (RAW FEATURES)
# -------------------------------------------------
numerical_features = [
    "grid",
    "laps",
    "year",
    "round",
    "driver_age",
    "driver_experience",
    "constructor_experience"
]

categorical_features = [
    "constructorId",
    "circuitId",
    "avg_grid_last_5_cat",
    "avg_finish_last_5_cat"
]

# -------------------------------------------------
# Preprocessing
# -------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
    ]
)

# -------------------------------------------------
# Train-validation split
# -------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------------------------
# Define models
# -------------------------------------------------
models = {
    "logistic_regression": Pipeline([
        ("preprocessing", preprocessor),
        ("model", LogisticRegression(max_iter=1000))
    ]),

    "decision_tree": Pipeline([
        ("preprocessing", preprocessor),
        ("model", DecisionTreeClassifier(random_state=42))
    ]),

    "knn": Pipeline([
        ("preprocessing", preprocessor),
        ("model", KNeighborsClassifier(n_neighbors=7, weights="distance"))
    ]),

    "gaussian_nb": Pipeline([
        ("preprocessing", preprocessor),
        ("model", GaussianNB())
    ]),

    "random_forest": Pipeline([
        ("preprocessing", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ))
    ]),

    "xgboost": Pipeline([
        ("preprocessing", preprocessor),
        ("model", xgb.XGBClassifier(
            eval_metric="logloss",
            random_state=42
        ))
    ])
}

# -------------------------------------------------
# Train & save models
# -------------------------------------------------
for name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, f"{MODEL_DIR}/{name}.pkl")
    print(f"✅ Saved {name}.pkl")

print("\n🎉 All models trained and saved successfully!")
