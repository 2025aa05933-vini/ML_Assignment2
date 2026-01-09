import pandas as pd
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

print("Dataset prepared successfully!")
print("Final shape:", final_df.shape)
print(final_df.head())