# ================================================================
# Chapter 3: Handling Outliers in Regression Analysis using Python
# Description: Demonstrates how to detect and handle outliers using
# Z-score filtering and robust regression (RANSAC) on synthetic data.
# ================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------------
# Step 1: Generate Synthetic Linear Data with Noise
# --------------------------------------------------

np.random.seed(0)  # For reproducibility

# Generate 100 data points: X in [0, 2), y = 4 + 3X + noise
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# --------------------------------------------------
# Step 2: Inject Artificial Outliers into the Dataset
# --------------------------------------------------

# Add 3 outlier points that deviate significantly from the trend
X_outliers = np.append(X, [[1.5], [1.8], [2.2]], axis=0)
y_outliers = np.append(y, [[20], [22], [25]], axis=0)

# --------------------------------------------------
# Step 3: Fit Standard Linear Regression on Noisy Data
# --------------------------------------------------

model = LinearRegression()
model.fit(X_outliers, y_outliers)
y_pred = model.predict(X_outliers)

# 📊 Visualize regression line with outliers
plt.figure(figsize=(8, 6))
plt.scatter(X_outliers, y_outliers, color='blue', label='Data with Outliers')
plt.plot(X_outliers, y_pred, color='red', label='Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with Outliers')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --------------------------------------------------
# Step 4: Detect Outliers Using Z-Score Method
# --------------------------------------------------

# Compute Z-scores for X values
z_scores = np.abs((X_outliers - X_outliers.mean()) / X_outliers.std())

# Identify indices where Z-score exceeds threshold (e.g., 3)
outliers = np.where(z_scores > 3)
print(f"🔍 Outliers detected based on Z-score: {outliers}")

# --------------------------------------------------
# Step 5: Remove Outliers and Refit the Model
# --------------------------------------------------

# Remove outlier rows from X and y
X_cleaned = np.delete(X_outliers, outliers, axis=0)
y_cleaned = np.delete(y_outliers, outliers, axis=0)

# Fit linear regression on cleaned data
model_cleaned = LinearRegression()
model_cleaned.fit(X_cleaned, y_cleaned)
y_pred_cleaned = model_cleaned.predict(X_cleaned)

# 📊 Visualize regression line after outlier removal
plt.figure(figsize=(8, 6))
plt.scatter(X_cleaned, y_cleaned, color='green', label='Cleaned Data')
plt.plot(X_cleaned, y_pred_cleaned, color='red', label='Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression After Outlier Removal')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --------------------------------------------------
# Step 6: Evaluate Cleaned Model Performance
# --------------------------------------------------

mae_cleaned = mean_absolute_error(y_cleaned, y_pred_cleaned)
mse_cleaned = mean_squared_error(y_cleaned, y_pred_cleaned)
rmse_cleaned = np.sqrt(mse_cleaned)
r2_cleaned = r2_score(y_cleaned, y_pred_cleaned)

print(f"📈 MAE (cleaned data): {mae_cleaned:.2f}")
print(f"📈 MSE (cleaned data): {mse_cleaned:.2f}")
print(f"📈 RMSE (cleaned data): {rmse_cleaned:.2f}")
print(f"📈 R² (cleaned data): {r2_cleaned:.2f}")

# --------------------------------------------------
# Step 7: Apply Robust Regression using RANSAC
# --------------------------------------------------

# RANSAC is robust to outliers by design
ransac = RANSACRegressor()
ransac.fit(X_outliers, y_outliers)
y_ransac_pred = ransac.predict(X_outliers)

# 📊 Visualize RANSAC regression line
plt.figure(figsize=(8, 6))
plt.scatter(X_outliers, y_outliers, color='blue', label='Data with Outliers')
plt.plot(X_outliers, y_ransac_pred, color='red', label='RANSAC Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Robust Regression using RANSAC')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --------------------------------------------------
# Step 8: Evaluate RANSAC Model Performance
# --------------------------------------------------

mae_ransac = mean_absolute_error(y_outliers, y_ransac_pred)
mse_ransac = mean_squared_error(y_outliers, y_ransac_pred)
rmse_ransac = np.sqrt(mse_ransac)
r2_ransac = r2_score(y_outliers, y_ransac_pred)

print(f"🛡️ MAE (RANSAC): {mae_ransac:.2f}")
print(f"🛡️ MSE (RANSAC): {mse_ransac:.2f}")
print(f"🛡️ RMSE (RANSAC): {rmse_ransac:.2f}")
print(f"🛡️ R² (RANSAC): {r2_ransac:.2f}")
