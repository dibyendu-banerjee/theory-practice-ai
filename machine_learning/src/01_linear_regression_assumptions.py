"""# Chapter 3: Example of Python code for Handling Outliers"""

import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import RANSACRegressor



# Generate synthetic data

np.random.seed(0)

X = 2 * np.random.rand(100, 1)

y = 4 + 3 * X + np.random.randn(100, 1)



# Add some outliers

X_outliers = np.append(X, [[1.5], [1.8], [2.2]], axis=0)

y_outliers = np.append(y, [[20], [22], [25]], axis=0)



# Fit a linear regression model

model = LinearRegression()

model.fit(X_outliers, y_outliers)

y_pred = model.predict(X_outliers)



# Plot data with outliers

plt.scatter(X_outliers, y_outliers, color='blue', label='Data with Outliers')

plt.plot(X_outliers, y_pred, color='red', label='Regression Line')

plt.xlabel('X')

plt.ylabel('y')

plt.legend()

plt.show()



# Identify outliers using Z-score

z_scores = np.abs((X_outliers - X_outliers.mean()) / X_outliers.std())

outliers = np.where(z_scores > 3)

print(f"Outliers based on Z-score: {outliers}")



# Remove outliers

X_cleaned = np.delete(X_outliers, outliers, axis=0)

y_cleaned = np.delete(y_outliers, outliers, axis=0)



# Fit a linear regression model on cleaned data

model_cleaned = LinearRegression()

model_cleaned.fit(X_cleaned, y_cleaned)

y_pred_cleaned = model_cleaned.predict(X_cleaned)



# Plot data without outliers

plt.scatter(X_cleaned, y_cleaned, color='green', label='Cleaned Data')

plt.plot(X_cleaned, y_pred_cleaned, color='red', label='Regression Line')

plt.xlabel('X')

plt.ylabel('y')

plt.legend()

plt.show()



# Calculate evaluation metrics

mae_cleaned = mean_absolute_error(y_cleaned, y_pred_cleaned)

mse_cleaned = mean_squared_error(y_cleaned, y_pred_cleaned)

rmse_cleaned = np.sqrt(mse_cleaned)

r2_cleaned = r2_score(y_cleaned, y_pred_cleaned)



print(f"MAE (cleaned data): {mae_cleaned:.2f}")

print(f"MSE (cleaned data): {mse_cleaned:.2f}")

print(f"RMSE (cleaned data): {rmse_cleaned:.2f}")

print(f"R-squared (cleaned data): {r2_cleaned:.2f}")



# Using robust regression (RANSAC)

ransac = RANSACRegressor()

ransac.fit(X_outliers, y_outliers)

y_ransac_pred = ransac.predict(X_outliers)



# Plot data with robust regression

plt.scatter(X_outliers, y_outliers, color='blue', label='Data with Outliers')

plt.plot(X_outliers, y_ransac_pred, color='red', label='RANSAC Regression Line')

plt.xlabel('X')



plt.ylabel('y')

plt.legend()

plt.show()



# Calculate evaluation metrics for RANSAC

mae_ransac = mean_absolute_error(y_outliers, y_ransac_pred)

mse_ransac = mean_squared_error(y_outliers, y_ransac_pred)

rmse_ransac = np.sqrt(mse_ransac)

r2_ransac = r2_score(y_outliers, y_ransac_pred)



print(f"MAE (RANSAC): {mae_ransac:.2f}")

print(f"MSE (RANSAC): {mse_ransac:.2f}")

print(f"RMSE (RANSAC): {rmse_ransac:.2f}")

print(f"R-squared (RANSAC): {r2_ransac:.2f}")