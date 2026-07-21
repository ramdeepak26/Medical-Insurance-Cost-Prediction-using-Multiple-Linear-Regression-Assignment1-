"""
Task 1: Data Understanding (2 Marks)
--------------------------------------
1. Load the dataset using Pandas.
2. Display the first five records.
3. Identify numerical features, categorical features, and the target variable.
"""

import pandas as pd

# 1. Load the dataset
df = pd.read_csv("insurance.csv")

# 2. Display first five records
print("First five records:")
print(df.head())

print("\nDataset shape:", df.shape)

print("\nColumn data types:")
print(df.dtypes)

# 3. Identify feature types
numerical_features = ["age", "bmi", "children"]
categorical_features = ["sex", "smoker", "region"]
target_variable = "charges"

print("\nNumerical features   :", numerical_features)
print("Categorical features :", categorical_features)
print("Target variable      :", target_variable)

print("\nSummary statistics (numerical columns):")
print(df.describe())
