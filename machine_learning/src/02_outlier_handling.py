# ================================================================
# Chapter 3: Underfitting vs. Overfitting in Regression Models
# Description: Illustrates underfitting using linear regression
# and overfitting using high-degree polynomial regression.
# ================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# --------------------------------------------------
# Step 1: Generate Synthetic Data with Noise
# --------------------------------------------------

np.random.seed(42)  # For reproducibility

# Generate 30 sorted random values between 0 and 1
X = np.sort(np.random.rand(30))

# Create a linear relationship with added Gaussian noise
y = 2 * X + 1 + np.random.normal(0, 0.1, len(X))

# Reshape X to 2D array for model compatibility
X = X.reshape(-1, 1)

# --------------------------------------------------
# Step 2: Fit a Linear Regression Model (Underfitting)
# --------------------------------------------------

linear_model = LinearRegression()
linear_model.fit(X, y)
y_linear_pred = linear_model.predict(X)

# --------------------------------------------------
# Step 3: Fit a Polynomial Regression Model (Overfitting)
# --------------------------------------------------

# Transform features to polynomial terms up to degree 15
polynomial_features = PolynomialFeatures(degree=15)
X_poly = polynomial_features.fit_transform(X)

# Fit linear regression on polynomial-transformed features
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
y_poly_pred = poly_model.predict(X_poly)

# --------------------------------------------------
# Step 4: Visualize Underfitting vs. Overfitting
# --------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 📉 Underfitting: Linear Regression
axes[0].scatter(X, y, color='black', label='Data', s=50)
axes[0].plot(X, y_linear_pred, color='blue', linewidth=2, label='Linear Fit (Underfitting)')
axes[0].set_title('Underfitting: Linear Regression', fontsize=14)
axes[0].set_xlabel('X', fontsize=12)
axes[0].set_ylabel('y', fontsize=12)
axes[0].grid(True, linestyle='--', color='gray', alpha=0.6)
axes[0].legend()

# 📈 Overfitting: Polynomial Regression (Degree 15)
axes[1].scatter(X, y, color='black', label='Data', s=50)
axes[1].plot(X, y_poly_pred, color='red', linewidth=2, label='Polynomial Fit (Overfitting)')
axes[1].set_title('Overfitting: Polynomial Regression', fontsize=14)
axes[1].set_xlabel('X', fontsize=12)
axes[1].set_ylabel('y', fontsize=12)
axes[1].grid(True, linestyle='--', color='gray', alpha=0.6)
axes[1].legend()

# Adjust layout and display
plt.tight_layout()
plt.show()
