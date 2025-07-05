"""##Chapter 3: Example of Python Code to implement Polynomial Regression Predictions"""

from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split  # Fix: missing import

# Generate synthetic data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1) + 5 * (X ** 2)  # Quadratic relationship

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Calculate MSE
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)

print(f"MSE on training data: {mse_train:.2f}")
print(f"MSE on test data: {mse_test:.2f}")

# Plot data and predictions
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X, model.predict(X), color='red', label='Linear Regression Line')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

# Polynomial regression
poly = PolynomialFeatures(degree=2)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

model_poly = LinearRegression()
model_poly.fit(X_poly_train, y_train)
y_pred_train_poly = model_poly.predict(X_poly_train)
y_pred_test_poly = model_poly.predict(X_poly_test)

# Calculate MSE for polynomial regression
mse_train_poly = mean_squared_error(y_train, y_pred_train_poly)
mse_test_poly = mean_squared_error(y_test, y_pred_test_poly)

print(f"MSE on training data (Polynomial Regression): {mse_train_poly:.2f}")
print(f"MSE on test data (Polynomial Regression): {mse_test_poly:.2f}")

# Plot polynomial regression predictions
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X, model_poly.predict(poly.transform(X)), color='green', label='Polynomial Regression Line')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()