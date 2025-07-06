# ================================================================
# File: model_cross_validation.py
# Description: This script demonstrates how to perform k-fold 
# cross-validation using scikit-learn. It uses the Iris dataset 
# and a Random Forest classifier to evaluate model performance 
# across multiple folds.
#
# Author: Dibyendu Banerjee
# ================================================================

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# -------------------------------
# Step 1: Load the dataset
# -------------------------------

X, y = load_iris(return_X_y=True)

# -------------------------------
# Step 2: Define the model
# -------------------------------

model = RandomForestClassifier()

# -------------------------------
# Step 3: Perform k-fold cross-validation
# -------------------------------

cv_scores = cross_val_score(model, X, y, cv=5)

# -------------------------------
# Step 4: Display results
# -------------------------------

print("Cross-Validation Scores:", cv_scores)
print("Mean Score:", cv_scores.mean())
