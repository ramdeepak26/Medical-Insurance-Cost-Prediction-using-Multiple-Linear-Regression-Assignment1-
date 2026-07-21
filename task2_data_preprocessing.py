"""
Task 2: Data Preprocessing (2 Marks)
--------------------------------------
1. Check for missing values.
2. Encode categorical variables (sex, smoker, region).
3. Split the dataset into 80% training and 20% testing.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("insurance.csv")

# 1. Check for missing values
print("Missing values per column:")
print(df.isnull().sum())

# 2. Encode categorical variables
df_encoded = df.copy()

# Binary categorical variables -> label encoding
df_encoded["sex"] = df_encoded["sex"].map({"male": 1, "female": 0})
df_encoded["smoker"] = df_encoded["smoker"].map({"yes": 1, "no": 0})

# Multi-category variable -> one-hot encoding (drop_first avoids the dummy variable trap)
df_encoded = pd.get_dummies(df_encoded, columns=["region"], drop_first=True)

print("\nEncoded dataframe (first 5 rows):")
print(df_encoded.head())

# Save the encoded dataset so later tasks/scripts can reuse it directly
df_encoded.to_csv("insurance_encoded.csv", index=False)
print("\nSaved encoded dataset to insurance_encoded.csv")

# 3. Train/test split (80/20)
X = df_encoded.drop("charges", axis=1)
y = df_encoded["charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining set shape:", X_train.shape)
print("Testing set shape :", X_test.shape)
