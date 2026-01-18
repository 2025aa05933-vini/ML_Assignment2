# ML Assignment 2 – Formula 1 Podium Finish Prediction

## a. Problem Statement

The objective of this project is to build and evaluate multiple machine learning classification models to predict whether a Formula 1 driver finishes on the podium (Top 3) in a race. Podium finishes are relatively rare events, making this a **class-imbalanced classification problem**.

The task involves:

* Preparing a clean and meaningful dataset from raw Formula 1 race data
* Training multiple classification models
* Evaluating them using appropriate performance metrics
* Comparing model performance and drawing insights

The final outcome is a Streamlit-based application that loads pre-trained models and evaluates them consistently on the same test dataset.

---

## b. Dataset Description

The dataset is derived from publicly available Formula 1 historical data. Raw CSV files include information about races, drivers, constructors, circuits, and race results. These raw files are processed and merged to create a final dataset suitable for machine learning.

### Target Variable

* **podium_finish**: Binary variable

  * `1` → Driver finished in Top 3
  * `0` → Driver did not finish in Top 3

### Selected Features (12 columns)

* **grid** – Starting grid position
* **laps** – Number of laps completed
* **year** – Season year
* **round** – Race round number
* **driver_age** – Age of the driver during the race
* **driver_experience** – Number of races completed by the driver before the race
* **constructor_experience** – Number of races completed by the constructor before the race
* **constructorId** – Constructor identifier
* **circuitId** – Circuit identifier
* **avg_grid_last_5_cat** – Average grid position over last 5 races (categorised)
* **avg_finish_last_5_cat** – Average finishing position over last 5 races (categorised)
* **podium_finish** – Target variable

The **raw test CSV used for evaluation and upload in the Streamlit app is located at:**
`data/processed/f1_test.csv`

A time-aware train–test split is used to avoid data leakage across seasons.

---

## c. Models Used and Evaluation

Six different machine learning models were trained and evaluated using the same dataset and evaluation pipeline.

### Evaluation Metrics Used

Since podium finishes are rare, accuracy alone is not sufficient. Hence, the following metrics are used:

* **Accuracy**
* **ROC-AUC Score**
* **Precision**
* **Recall**
* **F1 Score**
* **Matthews Correlation Coefficient (MCC)**

### Comparison Table

| ML Model Name            | Accuracy | AUC    | Precision | Recall | F1 Score | MCC    |
| ------------------------ | -------- | ------ | --------- | ------ | -------- | ------ |
| Logistic Regression      | 0.914    | 0.9339 | 0.6716    | 0.6818 | 0.6767   | 0.6271 |
| Decision Tree            | 0.898    | 0.7871 | 0.6087    | 0.6364 | 0.6222   | 0.5635 |
| kNN                      | 0.886    | 0.8764 | 0.5692    | 0.5606 | 0.5649   | 0.4993 |
| Naive Bayes              | 0.586    | 0.6716 | 0.2122    | 0.7879 | 0.3344   | 0.2324 |
| Random Forest (Ensemble) | 0.920    | 0.9251 | 0.7955    | 0.5303 | 0.6364   | 0.6088 |
| XGBoost (Ensemble)       | 0.920    | 0.9197 | 0.7031    | 0.6818 | 0.6923   | 0.6464 |

*(Metrics computed using the uploaded test file: `data/processed/f1_test.csv` in the Streamlit application.)**

---

## Model Performance Observations

| ML Model Name            | Observation about Model Performance                                                                                                                                                                                |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Logistic Regression      | Performs reasonably well as a baseline model. It captures linear relationships but struggles with complex interactions between features. Precision and recall are balanced but not optimal for rare podium events. |
| Decision Tree            | Shows better performance than logistic regression on training data but tends to overfit. Performance varies depending on tree depth and data distribution.                                                         |
| kNN                      | Performance is sensitive to the choice of k and feature scaling. It struggles with high-dimensional data and does not generalize very well on the test set.                                                        |
| Naive Bayes              | Simple and fast model but makes strong independence assumptions between features, which limits its performance on this dataset. Recall is usually lower for podium prediction.                                     |
| Random Forest (Ensemble) | Performs significantly better due to ensemble learning. Handles non-linearity well and provides a good balance between precision and recall. Less prone to overfitting compared to a single decision tree.         |
| XGBoost (Ensemble)       | Achieves the best overall performance. It effectively captures complex patterns, handles class imbalance well, and provides higher AUC, F1, and MCC scores compared to other models.                               |

---

## Conclusion

Among all the models tested, **ensemble models (Random Forest and XGBoost)** perform the best on this dataset. Since podium finishes are rare, metrics like **F1 Score and MCC** are more meaningful than accuracy alone. The results highlight the importance of choosing appropriate evaluation metrics and using time-aware data splitting to avoid data leakage.

This project demonstrates a complete machine learning pipeline—from raw data processing to model evaluation and deployment using Streamlit—at an undergraduate level.
