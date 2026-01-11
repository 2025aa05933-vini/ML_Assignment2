import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ==============================
# 1. Load Raw Datasets
# ==============================
results = pd.read_csv("data/raw/results.csv")
races = pd.read_csv("data/raw/races.csv")
circuits = pd.read_csv("data/raw/circuits.csv")
constructors = pd.read_csv("data/raw/constructors.csv")

# ==============================
# 2. Replace '\N' with 0 (NOT NaN)
# ==============================
for df_ in [results, races, circuits, constructors]:
    df_.replace("\\N", 0, inplace=True)

# ==============================
# 3. Merge datasets
# ==============================
df = results.merge(
    races[["raceId", "year", "round", "circuitId"]],
    on="raceId",
    how="left"
)

df = df.merge(
    constructors[["constructorId"]],
    on="constructorId",
    how="left"
)

# ==============================
# 4. Create target variable
# ==============================
df["podium_finish"] = (df["positionOrder"] <= 3).astype(int)

# ==============================
# 5. Select final features
# ==============================
final_df = df[
    [
        "grid",
        "laps",
        "points",
        "milliseconds",
        "fastestLapSpeed",
        "fastestLapTime",
        "rank",
        "year",
        "round",
        "constructorId",
        "circuitId",
        "podium_finish"
    ]
]

# ==============================
# 6. Convert to numeric
# ==============================
final_df = final_df.apply(pd.to_numeric, errors="coerce")

# ==============================
# 7. Drop REAL NaNs only
# ==============================
final_df = final_df.dropna()

# ==============================
# 8. Feature Engineering
# ==============================

# ---- Time features → seconds
final_df["milliseconds"] = final_df["milliseconds"] / 1000
final_df["fastestLapTime"] = final_df["fastestLapTime"] / 1000
# fastestLapSpeed remains in km/h

# ---- Nominal encoding for categorical IDs
label_encoders = {}
for col in ["constructorId", "circuitId"]:
    le = LabelEncoder()
    final_df[col] = le.fit_transform(final_df[col])
    label_encoders[col] = le

# ---- Normalize numerical features
numeric_features = [
    "grid",
    "laps",
    "points",
    "milliseconds",
    "fastestLapSpeed",
    "fastestLapTime",
    "rank",
    "year",
    "round"
]

scaler = StandardScaler()
final_df[numeric_features] = scaler.fit_transform(
    final_df[numeric_features]
)

# ==============================
# 9. Save full processed dataset
# ==============================
final_df.to_csv(
    "data/processed/f1_top10_classification.csv",
    index=False
)

# ==============================
# 10. Fixed-size stratified split
# ==============================
TRAIN_SIZE = 1000
TEST_SIZE = 500
TOTAL_SIZE = TRAIN_SIZE + TEST_SIZE

if len(final_df) < TOTAL_SIZE:
    raise ValueError(
        f"Not enough data: {len(final_df)} rows available, "
        f"but {TOTAL_SIZE} required."
    )

X = final_df.drop("podium_finish", axis=1)
y = final_df["podium_finish"]

# Sample exactly 1500 rows
X_sampled, _, y_sampled, _ = train_test_split(
    X,
    y,
    train_size=TOTAL_SIZE,
    random_state=42,
    stratify=y
)

# Split into 1000 train / 500 test
X_train, X_test, y_train, y_test = train_test_split(
    X_sampled,
    y_sampled,
    train_size=TRAIN_SIZE,
    test_size=TEST_SIZE,
    random_state=42,
    stratify=y_sampled
)

# ==============================
# 11. Save train & test datasets
# ==============================
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

train_df.to_csv("data/processed/f1_train.csv", index=False)
test_df.to_csv("data/processed/f1_test.csv", index=False)

# ==============================
# 12. Logs
# ==============================
print("Dataset preparation completed successfully!")
print("Full dataset shape:", final_df.shape)
print("Train dataset shape:", train_df.shape)
print("Test dataset shape:", test_df.shape)
print("\nClass distribution:")
print("Train:\n", train_df["podium_finish"].value_counts(normalize=True))
print("Test:\n", test_df["podium_finish"].value_counts(normalize=True))
print("\nSample rows:")
print(final_df.head())
