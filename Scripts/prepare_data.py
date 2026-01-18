import pandas as pd
from sklearn.model_selection import train_test_split

# ==============================
# CONFIG
# ==============================
CLASSIFICATION_SIZE = 1500
TRAIN_SIZE = 1000
TEST_SIZE = 500

# ==============================
# 1. Load Raw Datasets
# ==============================
results = pd.read_csv("data/raw/results.csv")
races = pd.read_csv("data/raw/races.csv")
constructors = pd.read_csv("data/raw/constructors.csv")
drivers = pd.read_csv("data/raw/drivers.csv")

# ==============================
# 2. Replace '\N' with NaN
# ==============================
for df_ in [results, races, constructors, drivers]:
    df_.replace("\\N", pd.NA, inplace=True)

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

df = df.merge(
    drivers[["driverId", "dob"]],
    on="driverId",
    how="left"
)

# ==============================
# 4. Create target variable
# ==============================
df["positionOrder"] = pd.to_numeric(df["positionOrder"], errors="coerce")
df["podium_finish"] = (df["positionOrder"] <= 3).astype(int)

# ==============================
# 5. Time-aware sorting
# ==============================
df = df.sort_values(["driverId", "year", "round"])

# ==============================
# 6. Feature Engineering (DOMAIN ONLY)
# ==============================

# Driver age
df["dob"] = pd.to_datetime(df["dob"], errors="coerce")
df["driver_age"] = df["year"] - df["dob"].dt.year

# Driver experience
df["driver_experience"] = df.groupby("driverId").cumcount()

# Constructor experience
df = df.sort_values(["constructorId", "year", "round"])
df["constructor_experience"] = df.groupby("constructorId").cumcount()

# Rolling averages (last 5 races)
df = df.sort_values(["driverId", "year", "round"])

df["avg_grid_last_5"] = (
    df.groupby("driverId")["grid"]
    .rolling(5, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)

df["avg_finish_last_5"] = (
    df.groupby("driverId")["positionOrder"]
    .rolling(5, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)

# Binning rolling features
df["avg_grid_last_5_cat"] = pd.cut(
    df["avg_grid_last_5"],
    bins=[0, 5, 10, 25],
    labels=["front", "mid", "back"]
)

df["avg_finish_last_5_cat"] = pd.cut(
    df["avg_finish_last_5"],
    bins=[0, 5, 10, 25],
    labels=["strong", "average", "weak"]
)

# ==============================
# 7. Select EXACTLY 12 logical columns
# ==============================
final_df = df.loc[:, [
    "grid",
    "laps",
    "year",
    "round",
    "driver_age",
    "driver_experience",
    "constructor_experience",
    "constructorId",
    "circuitId",
    "avg_grid_last_5_cat",
    "avg_finish_last_5_cat",
    "podium_finish"
]]

# ==============================
# 8. Convert numeric columns & drop NaNs
# ==============================
numeric_cols = [
    "grid",
    "laps",
    "year",
    "round",
    "driver_age",
    "driver_experience",
    "constructor_experience"
]

final_df[numeric_cols] = final_df[numeric_cols].apply(
    pd.to_numeric, errors="coerce"
)

final_df = final_df.dropna()

# ==============================
# 9. Sample EXACTLY 1500 rows
# ==============================
X = final_df.drop("podium_finish", axis=1)
y = final_df["podium_finish"]

X_cls, _, y_cls, _ = train_test_split(
    X,
    y,
    train_size=CLASSIFICATION_SIZE,
    random_state=42,
    stratify=y
)

classification_df = pd.concat([X_cls, y_cls], axis=1)

# ==============================
# 10. Split into TRAIN / TEST
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X_cls,
    y_cls,
    train_size=TRAIN_SIZE,
    test_size=TEST_SIZE,
    random_state=42,
    stratify=y_cls
)

train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

# ==============================
# 11. Save outputs (ONLY 3 FILES)
# ==============================
classification_df.to_csv(
    "data/processed/f1_top10_classification.csv",
    index=False
)

train_df.to_csv(
    "data/processed/f1_train.csv",
    index=False
)

test_df.to_csv(
    "data/processed/f1_test.csv",
    index=False
)

# ==============================
# 12. Logs
# ==============================
print("prepare_data.py completed successfully")
print("Classification rows:", classification_df.shape[0])
print("Train rows:", train_df.shape[0])
print("Test rows:", test_df.shape[0])
