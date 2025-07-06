# ================================================================
# File: model_tuning_metrics.py
# Description: This script demonstrates evaluation metrics for both 
# classification and regression tasks. It includes ROC-AUC for binary 
# classification and MSE/MAE for regression using synthetic datasets.
#
# Author: Dibyendu Banerjee
# ================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, roc_auc_score,
    mean_squared_error, mean_absolute_error
)
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

# -------------------------------
# Part 1: ROC-AUC for Classification
# -------------------------------

# Generate synthetic classification data
X_class, y_class = make_classification(
    n_samples=1000, n_features=20, n_informative=2,
    n_classes=2, random_state=42
)

# Train-test split
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_class, y_class, test_size=0.3, random_state=42
)

# Train a Random Forest classifier
clf = RandomForestClassifier()
clf.fit(X_train_c, y_train_c)

# Predict probabilities for class 1
y_probs = clf.predict_proba(X_test_c)[:, 1]

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test_c, y_probs)
auc_score = roc_auc_score(y_test_c, y_probs)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})', color='purple')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

print(f"AUC Score: {auc_score:.2f}")

# -------------------------------
# Part 2: MSE and MAE for Regression
# -------------------------------

# Generate synthetic regression data
X_reg, y_reg = make_regression(n_samples=1000, n_features=1, noise=20, random_state=42)

# Train-test split
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

# Train a linear regression model
reg_model = LinearRegression()
reg_model.fit(X_train_r, y_train_r)

# Predict on test set
y_pred_r = reg_model.predict(X_test_r)

# Compute metrics
mse = mean_squared_error(y_test_r, y_pred_r)
mae = mean_absolute_error(y_test_r, y_pred_r)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")

# Plot actual vs predicted
plt.scatter(X_test_r, y_test_r, color='blue', label='Actual')
plt.scatter(X_test_r, y_pred_r, color='red', label='Predicted')
plt.plot(X_test_r, y_pred_r, color='green', linewidth=2, label='Regression Line')
plt.title('Actual vs Predicted')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.legend()
plt.grid(True)
plt.show()
