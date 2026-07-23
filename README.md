# Medical Insurance Cost Prediction using Multiple Linear Regression (Assignment1)

## Overview
This project builds a **Multiple Linear Regression** model to predict medical insurance
charges for customers based on their personal and health-related information, using the
[Medical Cost Personal Insurance Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)
from Kaggle.

## Dataset
- **File:** `insurance.csv`
- **Records:** 1,338
- **Columns:**
  | Column | Type | Description |
  |---|---|---|
  | age | Numerical | Age of the primary beneficiary |
  | sex | Categorical | Gender (male/female) |
  | bmi | Numerical | Body Mass Index |
  | children | Numerical | Number of dependents covered |
  | smoker | Categorical | Smoking status (yes/no) |
  | region | Categorical | Residential region in the US |
  | charges | Numerical (Target) | Medical insurance cost billed |

## Project Structure
```
├── insurance.csv                    # Raw dataset
├── task1_data_understanding.py      # Task 1: Load data, inspect features & target
├── task2_data_preprocessing.py      # Task 2: Missing value check, encoding, train/test split
├── task3_model_development.py       # Task 3: Train Linear Regression, generate predictions
├── task4_model_evaluation.py        # Task 4: MAE, MSE, R2, Actual vs Predicted plot
├── task5_conclusion.md              # Task 5: Written conclusion
├── actual_vs_predicted.png          # Output plot from Task 4
└── README.md                        # Project documentation (this file)
```

## Requirements
```
pandas
numpy
matplotlib
scikit-learn
joblib
```
Install with:
```bash
pip install pandas numpy matplotlib scikit-learn joblib
```

## How to Run
Each script is standalone and can be run independently and in any order:
```bash
python task1_data_understanding.py
python task2_data_preprocessing.py
python task3_model_development.py
python task4_model_evaluation.py
```
Running `task2`, `task3`, and `task4` will additionally generate:
- `insurance_encoded.csv` — preprocessed dataset
- `predictions.csv` — actual vs predicted charges on the test set
- `linear_regression_model.pkl` — the trained model
- `actual_vs_predicted.png` — evaluation scatter plot

## Methodology
1. **Data Understanding:** Loaded the dataset and identified 3 numerical features (age, bmi,
   children), 3 categorical features (sex, smoker, region), and the target variable (charges).
2. **Preprocessing:** Confirmed there are no missing values. Label-encoded `sex` and `smoker`
   (binary), one-hot encoded `region` (drop_first=True), and split the data 80/20 into
   training and test sets (`random_state=42`).
3. **Model Development:** Trained a `LinearRegression` model from scikit-learn on all 6
   features (age, sex, bmi, children, smoker, region) to predict `charges`.
4. **Evaluation:** Assessed performance using MAE, MSE, and R², and visualized results with an
   Actual vs Predicted scatter plot.

## Results
| Metric | Value |
|---|---|
| MAE | ≈ 4,181.19 |
| MSE | ≈ 33,596,915.85 |
| RMSE | ≈ 5,796.28 |
| R² Score | ≈ 0.7836 |

**Key findings:**
- The model explains ~78% of the variance in insurance charges.
- **Smoking status** is by far the strongest predictor of charges, followed by **BMI**, **age**,
  and **number of children**.
- `sex` and `region` have only a marginal effect on predicted charges.

## Limitation
Linear Regression assumes a purely linear, additive relationship between the features and the
target. In reality, the effect of BMI and age on charges is much steeper for smokers than for
non-smokers (an interaction effect), and the charges distribution is skewed rather than normal.
Capturing this would likely require interaction terms, a log-transformed target, or a
non-linear model such as a decision tree or gradient boosting regressor.
