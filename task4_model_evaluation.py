"""
Task 4: Model Evaluation (2 Marks)
--------------------------------------
Evaluate the model using MAE, MSE, and R2 Score.
Create an Actual vs Predicted scatter plot.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---- Load & preprocess (kept here so this script runs standalone) ----
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

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ---- Evaluation metrics ----
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE) : {mae:.2f}")
print(f"Mean Squared Error (MSE)  : {mse:.2f}")
print(f"Root MSE (RMSE)           : {rmse:.2f}")
print(f"R2 Score                  : {r2:.4f}")

# ---- Actual vs Predicted scatter plot ----
plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color="#4C72B0", edgecolor="white", s=50)
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
plt.plot(lims, lims, "r--", linewidth=2, label="Perfect Prediction")
plt.xlabel("Actual Charges ($)")
plt.ylabel("Predicted Charges ($)")
plt.title("Actual vs Predicted Insurance Charges")
plt.legend()
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150)
plt.show()

print("\nPlot saved as actual_vs_predicted.png")

"""
Observations:
1. Good overall fit with a systematic pattern in the residuals. The model explains about 78%
   of the variance in charges (R2 ~ 0.78), and points cluster reasonably close to the red
   "perfect prediction" line for low-to-mid charge ranges.
2. Two visible clusters/bands appear in the scatter plot. This is a direct effect of the
   smoker variable: non-smokers occupy the lower-charge band while smokers occupy a much
   higher, more spread-out band - a single straight-line (linear) relationship cannot
   perfectly capture this split.
3. The model under-predicts very high charges. For customers with very large actual charges
   (mostly smokers with high BMI), predictions tend to fall below the actual value,
   suggesting a non-linear interaction between smoker and bmi that a plain linear model
   can't capture.
"""
