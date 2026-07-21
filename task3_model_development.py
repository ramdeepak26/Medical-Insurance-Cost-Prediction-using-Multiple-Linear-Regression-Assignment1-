"""
Task 3: Model Development (3 Marks)
--------------------------------------
Build a Multiple Linear Regression model using: Age, Sex, BMI, Children, Smoker, Region.
Target: Charges.
Train the model and predict charges for the test dataset.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# ---- Load & preprocess (same steps as Task 2, kept here so this script runs standalone) ----
df = pd.read_csv("insurance.csv")

df_encoded = df.copy()
df_encoded["sex"] = df_encoded["sex"].map({"male": 1, "female": 0})
df_encoded["smoker"] = df_encoded["smoker"].map({"yes": 1, "no": 0})
df_encoded = pd.get_dummies(df_encoded, columns=["region"], drop_first=True)

X = df_encoded.drop("charges", axis=1)
y = df_encoded["charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---- Build and train the Multiple Linear Regression model ----
model = LinearRegression()
model.fit(X_train, y_train)

# ---- Predict charges on the test set ----
y_pred = model.predict(X_test)

# Show model coefficients
coeff_df = pd.DataFrame({"Feature": X.columns, "Coefficient": model.coef_})
print("Intercept:", round(model.intercept_, 2))
print("\nModel coefficients:")
print(coeff_df)

# Show a sample of actual vs predicted values
results_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
print("\nSample predictions (first 10 test rows):")
print(results_df.head(10))

# Save predictions for use in Task 4 (evaluation)
results_df.to_csv("predictions.csv", index=False)
print("\nSaved predictions to predictions.csv")

# Save the trained model for reuse
joblib.dump(model, "linear_regression_model.pkl")
print("Saved trained model to linear_regression_model.pkl")
