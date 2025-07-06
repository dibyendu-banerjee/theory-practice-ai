# ================================================================
# Chapter 3: Comparing Linear, Ridge, Lasso, and ElasticNet Regression
# Description: Demonstrates how different regression techniques
# handle multivariate data and regularization using synthetic data.
# ================================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error

# --------------------------------------------------
# Step 1: Generate Synthetic Multivariate Dataset
# --------------------------------------------------

np.random.seed(42)  # For reproducibility
n_samples = 100

# Generate 5 independent features scaled between 0 and 100
X = np.random.rand(n_samples, 5) * 100

# Construct target variable with known coefficients and noise
# y = 3*x1 - 2*x2 + 1.5*x3 + Gaussian noise
y = 3 * X[:, 0] - 2 * X[:, 1] + 1.5 * X[:, 2] + np.random.randn(n_samples) * 10

# --------------------------------------------------
# Step 2: Split Data into Training and Testing Sets
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------------------------
# Step 3: Initialize Regression Models
# --------------------------------------------------

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),            # L2 regularization
    "Lasso Regression": Lasso(alpha=0.1),            # L1 regularization
    "Elastic Net": ElasticNet(alpha=0.1, l1_ratio=0.5)  # Combination of L1 and L2
}

# --------------------------------------------------
# Step 4: Train Models and Evaluate Performance
# --------------------------------------------------

results = []

for name, model in models.items():
    model.fit(X_train, y_train)                     # Train the model
    y_pred = model.predict(X_test)                  # Predict on test set
    mse = mean_squared_error(y_test, y_pred)        # Compute Mean Squared Error
    results.append((name, mse, model.coef_))        # Store results

# --------------------------------------------------
# Step 5: Display Model Performance and Coefficients
# --------------------------------------------------

print("📊 Model Performance (Mean Squared Error):\n")
for name, mse, coeffs in results:
    print(f"{name}:")
    print(f"  MSE = {mse:.2f}")
    print(f"  Coefficients = {np.round(coeffs, 2)}\n")
