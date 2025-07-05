"""#Chapter 3: Use Case 1: Demonstrating LR, Ridge, Lasso, and Elastic Net"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error

# Generate synthetic dataset
np.random.seed(42)
n_samples = 100
X = np.random.rand(n_samples, 5) * 100  # 5 predictors
y = 3 * X[:, 0] - 2 * X[:, 1] + 1.5 * X[:, 2] + np.random.randn(n_samples) * 10  # target variable

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1),
    "Elastic Net": ElasticNet(alpha=0.1, l1_ratio=0.5)
}

# Fit models and evaluate
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    results.append((name, mse, model.coef_))

# Display results
print("Model Performance (Mean Squared Error):")
for name, mse, coeffs in results:
    print(f"{name}: MSE = {mse:.2f}")
    print(f" Coefficients: {coeffs}\n")