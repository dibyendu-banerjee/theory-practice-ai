# ================================================================
# Chapter 4: Predicting Customer Churn with Logistic Regression
# Description: Demonstrates a full pipeline for churn prediction
# using synthetic data, preprocessing, and logistic regression.
# ================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# --------------------------------------------------
# Step 1: Generate Synthetic Customer Churn Dataset
# --------------------------------------------------

np.random.seed(42)  # For reproducibility
n_samples = 500

# Simulate customer features
MonthlySpend = np.random.uniform(20, 150, size=n_samples)  # Monthly bill
ContractType = np.random.choice(['Month-to-Month', 'One-Year', 'Two-Year'], size=n_samples)
CustomerTenure = np.random.randint(1, 73, size=n_samples)  # Tenure in months
Churn = np.random.choice([0, 1], size=n_samples)  # Binary churn label

# Create DataFrame
data = pd.DataFrame({
    'MonthlySpend': MonthlySpend,
    'ContractType': ContractType,
    'CustomerTenure': CustomerTenure,
    'Churn': Churn
})

# Preview the dataset
print("📋 Sample of the dataset:")
print(data.head())

# --------------------------------------------------
# Step 2: Validate and Clean Target Variable
# --------------------------------------------------

# Ensure 'Churn' column exists
if 'Churn' not in data.columns:
    raise ValueError("The target variable 'Churn' is missing from the dataset.")

# Handle missing values in target (if any)
if data['Churn'].isnull().any():
    print("⚠️ Missing values found in 'Churn'. Filling with mode.")
    data['Churn'].fillna(data['Churn'].mode()[0], inplace=True)

# --------------------------------------------------
# Step 3: Define Features and Target
# --------------------------------------------------

X = data[['MonthlySpend', 'ContractType', 'CustomerTenure']]
y = data['Churn']

# --------------------------------------------------
# Step 4: Build Preprocessing Pipeline
# --------------------------------------------------

# Scale numerical features and one-hot encode categorical ones
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), ['MonthlySpend', 'CustomerTenure']),
    ('cat', OneHotEncoder(), ['ContractType'])
])

# Combine preprocessing and model into a pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# --------------------------------------------------
# Step 5: Train-Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# --------------------------------------------------
# Step 6: Train the Logistic Regression Model
# --------------------------------------------------

pipeline.fit(X_train, y_train)

# --------------------------------------------------
# Step 7: Make Predictions and Evaluate
# --------------------------------------------------

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# --------------------------------------------------
# Step 8: Display Evaluation Results
# --------------------------------------------------

print(f"\n✅ Model Accuracy: {accuracy:.2f}")
print("\n📊 Classification Report:")
print(report)
