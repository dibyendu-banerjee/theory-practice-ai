"""#Chapter 3: overfitting (polynomial regression) and underfitting (linear regression)."""

import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import mean_squared_error



# Create a sample dataset with some noise

np.random.seed(42)

X = np.sort(np.random.rand(30))  # 30 random values between 0 and 1

y = 2 * X + 1 + np.random.normal(0, 0.1, len(X))  # Linear relationship with noise



# Reshape X for compatibility with models

X = X.reshape(-1, 1)



# Linear regression model (underfitting)

linear_model = LinearRegression()

linear_model.fit(X, y)

y_linear_pred = linear_model.predict(X)



# Polynomial regression model (degree 15 for overfitting)

polynomial_features = PolynomialFeatures(degree=15)

X_poly = polynomial_features.fit_transform(X)

poly_model = LinearRegression()

poly_model.fit(X_poly, y)

y_poly_pred = poly_model.predict(X_poly)



# Create figure for both graphs

fig, axes = plt.subplots(1, 2, figsize=(16, 6))



# 1. Underfitting (Linear Regression)

axes[0].scatter(X, y, color='black', label='Data', s=50)

axes[0].plot(X, y_linear_pred, label='Linear Fit (Underfitting)', color='blue', linewidth=2)

axes[0].set_title('Underfitting: Linear Regression', fontsize=14)

axes[0].set_xlabel('X', fontsize=12)

axes[0].set_ylabel('y', fontsize=12)

axes[0].grid(True, linestyle='--', color='gray', alpha=0.6)

axes[0].legend()



# 2. Overfitting (Polynomial Regression)

axes[1].scatter(X, y, color='black', label='Data', s=50)

axes[1].plot(X, y_poly_pred, label='Polynomial Fit (Overfitting)', color='red', linewidth=2)

axes[1].set_title('Overfitting: Polynomial Regression', fontsize=14)

axes[1].set_xlabel('X', fontsize=12)

axes[1].set_ylabel('y', fontsize=12)

axes[1].grid(True, linestyle='--', color='gray', alpha=0.6)

axes[1].legend()



# Adjust layout for better spacing

plt.tight_layout()



# Display the plot

plt.show()