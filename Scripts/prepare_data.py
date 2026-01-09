import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Loading Raw Datasets   
results = pd.read_csv("data/raw/results.csv")
races = pd.read_csv("data/raw/races.csv")
circuits = pd.read_csv("data/raw/circuits.csv")
constructors = pd.read_csv("data/raw/constructors.csv")

# 2. Merge results with races (year, round, circuit)
df = results.merge(
    races[["raceId", "year", "round", "circuitId"]],
    on="raceId",
    how="left"
)

# 3. Merge with constructors (team info)
df = df.merge(
    constructors[["constructorId"]],
    on="constructorId",
    how="left"
)
# 4. Create target variable (classification label)
df["podium_finish"] = (df["positionOrder"] <= 3).astype(int)

# 5. Select final features
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

# 6. Remove missing values
final_df = final_df.dropna()

# 7. Save processed dataset
final_df.to_csv(
    "data/processed/f1_top10_classification.csv",
    index=False
)

# 8. Train-test split (95% train, 5% test)
X = final_df.drop("podium_finish", axis=1)
y = final_df["podium_finish"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.05,
    random_state=42,
    stratify=y
)

# Combine features and target back
X = final_df.drop("podium_finish", axis=1)
y = final_df["podium_finish"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.05,
    random_state=42,
    stratify=y
)

# 9. Combine features and target back
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

# 10. Save train and test datasets
train_df.to_csv(
    "data/processed/f1_train.csv",
    index=False
)

test_df.to_csv(
    "data/processed/f1_test.csv",
    index=False
)

# 11. Logs
print("Dataset preparation completed successfully!")
print("Full dataset shape:", final_df.shape)
print("Train dataset shape:", train_df.shape)
print("Test dataset shape:", test_df.shape)
print("\nSample rows:")
print(final_df.head())