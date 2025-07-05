"""# Chapter 4 - Use Case: Classifying Deepfake and Authentic Images Using Logistic Regression"""

# Import necessary libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, classification_report



# Step 1: Create a synthetic dataset

np.random.seed(42)  # For reproducibility



# Define the size of the dataset

n_samples = 1000



# Create synthetic data

data = {

    'pixel_inconsistency': np.random.uniform(0.0, 1.0, n_samples),  # Random value representing pixel inconsistency

    'facial_distortion': np.random.uniform(0.0, 1.0, n_samples),     # Random value representing facial distortion

    'lighting_shadows': np.random.uniform(0.0, 1.0, n_samples),       # Random value for lighting and shadows inconsistencies

    'eye_reflection': np.random.uniform(0.0, 1.0, n_samples),         # Random value for abnormal eye reflections

    'motion_artifacts': np.random.uniform(0.0, 1.0, n_samples),       # Random value for motion artifacts

}



# Generate a target variable: 'is_deepfake' (0 = Real, 1 = Deepfake)

# Assuming deepfakes have higher pixel inconsistency, facial distortion, motion artifacts, and eye reflection

data['is_deepfake'] = np.where(

    (data['pixel_inconsistency'] > 0.5) &

    (data['facial_distortion'] > 0.5) &

    (data['lighting_shadows'] > 0.5) &

    (data['eye_reflection'] > 0.5) &

    (data['motion_artifacts'] > 0.5),

    1, 0

)



# Convert to DataFrame

df = pd.DataFrame(data)



# Step 2: Preprocessing

# Define features (X) and target (y)

X = df.drop('is_deepfake', axis=1)

y = df['is_deepfake']



# Step 3: Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Step 4: Standardize the feature data

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



# Step 5: Train a Logistic Regression model

model = LogisticRegression()

model.fit(X_train, y_train)



# Step 6: Predict and evaluate the model

y_pred = model.predict(X_test)



# Step 7: Evaluate the model

accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy: {accuracy:.4f}')



# Classification Report

print('Classification Report:')

print(classification_report(y_test, y_pred))



# Step 8: Sample predictions (Print results for a few sample images)

sample_data = {

    'pixel_inconsistency': [0.7, 0.3, 0.8],

    'facial_distortion': [0.8, 0.2, 0.9],

    'lighting_shadows': [0.7, 0.4, 0.8],

    'eye_reflection': [0.8, 0.1, 0.9],

    'motion_artifacts': [0.7, 0.2, 0.8]

}



# Create a DataFrame for sample predictions

sample_df = pd.DataFrame(sample_data)



# Standardize the sample data (important for Logistic Regression)

sample_df_scaled = scaler.transform(sample_df)



# Predicting for the sample data

sample_predictions = model.predict(sample_df_scaled)



# Printing the results for the sample images

for idx, prediction in enumerate(sample_predictions):

    if prediction == 1:

        print(f"Sample Image {idx+1}: This is a Deepfake (Yes).")

    else:

        print(f"Sample Image {idx+1}: This is a Real Image (No).")