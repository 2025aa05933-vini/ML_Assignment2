
Formula 1 Podium Finish Prediction (ML Assignment 2)
1. Problem Statement
The objective of this project is to build and evaluate multiple machine learning classification models to predict whether a Formula 1 driver finishes on the podium (Top-3) in a race.
The prediction is based on race-level performance indicators, constructor information, and race metadata such as year and circuit. This is formulated as a binary classification problem.

2. Dataset Description
Dataset Source
The dataset is derived from the Formula 1 World Championship (1950–2023) dataset available on Kaggle (Ergast API based).
Multiple CSV files were used, primarily:
* results.csv
* races.csv
Data Preparation
Raw race result data was merged with race metadata to create a single tabular dataset suitable for classification.A new target variable was engineered based on finishing position.
Target Variable
Podium Finish (podium_finish)
* 1 → Driver finished in positions 1–3
* 0 → Driver finished outside the podium

3. Features Used (Raw Feature Count ≥ 12)
The dataset contains more than 12 raw features before preprocessing, satisfying the assignment requirement.
Numerical Features
* grid – Starting grid position
* laps – Number of laps completed
* points – Championship points scored
* milliseconds – Race completion time
* fastestLapSpeed – Fastest lap speed
* fastestLapTime – Fastest lap time
* rank – Fastest lap rank
* year – Race year
* round – Race round
Categorical Features
* constructorId – Team identifier
* circuitId – Circuit identifier
Target
* podium_finish

4. Data Cleaning and Preprocessing
* Missing values (\N) present in race performance attributes were handled by removing rows with incomplete data.
* This ensured that all machine learning models receive clean, numeric, and complete input.
* After cleaning, the dataset still contains well above the minimum required 500 instances.
The final processed dataset is stored as:

data/processed/f1_podium_classification.csv

5. Project Structure (So Far)
ML_Assignment2/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   │   ├── circuits.csv
│   │   ├── constructors.csv
│   │   ├── races.csv
│   │   │
│   └── processed/
│       └── f1_top10_classification.csv
│
├── Scripts/
│   └── prepare_data.py
│
├── model/
│
└── outputs/

* prepare_data.py handles dataset merging, target variable creation, feature selection, and data cleaning.
* Raw data files are preserved separately under data/raw/ to maintain data integrity and reproducibility.
* The cleaned and processed dataset used for model training is stored under data/processed/.
* The model/ directory is reserved for storing trained machine learning models.
* The outputs/ directory will store evaluation results such as metrics tables and plots.
* app.py will contain the Streamlit application code.
* requirements.txt lists all project dependencies required for deployment.

6. Next Steps (To Be Implemented)
* Encoding of categorical features
* Train–test split
* Implementation of six classification models:
    * Logistic Regression
    * Decision Tree
    * K-Nearest Neighbors
    * Naive Bayes
    * Random Forest
    * XGBoost
* Evaluation using Accuracy, AUC, Precision, Recall, F1-Score, and MCC
* Streamlit web application for interactive model evaluation
* Deployment on Streamlit Community Cloud
