"""# **Chapter 4: Supervised Learning: Logistic Regression**

# Chapter 4 - Use Case: Predicting Customer Churn with Logistic Regression
"""

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, classification_report



# Generate synthetic dataset

np.random.seed(42)



# Create a sample dataset

n_samples = 500

MonthlySpend = np.random.uniform(20, 150, size=n_samples)  # Random monthly spend

ContractType = np.random.choice(['Month-to-Month', 'One-Year', 'Two-Year'], size=n_samples)  # Random contract type

CustomerTenure = np.random.randint(1, 73, size=n_samples)  # Customer tenure in months (1-72 months)

Churn = np.random.choice([0, 1], size=n_samples)  # Random churn (0 = No, 1 = Yes)



# Create DataFrame

data = pd.DataFrame({

    'MonthlySpend': MonthlySpend,

    'ContractType': ContractType,

    'CustomerTenure': CustomerTenure,

    'Churn': Churn

})



# Display the first few rows of the dataset

print(data.head())



# Check for missing values in the 'Churn' column

if 'Churn' not in data.columns:

    raise ValueError("The target variable 'Churn' is missing from the dataset.")



# Check for missing values

if data['Churn'].isnull().any():

    print("Missing values found in the 'Churn' column. Filling with mode.")

    data['Churn'].fillna(data['Churn'].mode()[0], inplace=True)



# Features and target variable

X = data[['MonthlySpend', 'ContractType', 'CustomerTenure']]

y = data['Churn']



# Preprocessing

preprocessor = ColumnTransformer(

    transformers=[

        ('num', StandardScaler(), ['MonthlySpend', 'CustomerTenure']),

        ('cat', OneHotEncoder(), ['ContractType'])

    ]

)



# Create pipeline

pipeline = Pipeline(steps=[

    ('preprocessor', preprocessor),

    ('classifier', LogisticRegression())

])



# Split the data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



# Train the model

pipeline.fit(X_train, y_train)



# Make predictions

y_pred = pipeline.predict(X_test)



# Evaluate the model

accuracy = accuracy_score(y_test, y_pred)

report = classification_report(y_test, y_pred)



# Print results

print(f"Accuracy: {accuracy:.2f}")

print("Classification Report:")

print(report)