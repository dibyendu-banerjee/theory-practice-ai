# ================================================================
# Chapter 4: Detecting Deepfakes Using Logistic Regression
# Description: Simulates a binary classification task to detect
# deepfakes based on synthetic visual inconsistency features.
# ================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# --------------------------------------------------
# Step 1: Generate Synthetic Deepfake Detection Dataset
# --------------------------------------------------

np.random.seed(42)  # For reproducibility
n_samples = 1000

# Simulate visual inconsistency features
data = {
    'pixel_inconsistency': np.random.uniform(0.0, 1.0, n_samples),
    'facial_distortion': np.random.uniform(0.0, 1.0, n_samples),
    'lighting_shadows': np.random.uniform(0.0, 1.0, n_samples),
    'eye_reflection': np.random.uniform(0.0, 1.0, n_samples),
    'motion_artifacts': np.random.uniform(0.0, 1.0, n_samples),
}

# Define target: 1 = Deepfake, 0 = Real
# A sample is labeled as deepfake if all features exceed 0.5
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

# --------------------------------------------------
# Step 2: Preprocessing - Define Features and Target
# --------------------------------------------------

X = df.drop('is_deepfake', axis=1)
y = df['is_deepfake']

# --------------------------------------------------
# Step 3: Train-Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------------------------
# Step 4: Feature Scaling
# --------------------------------------------------

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------------
# Step 5: Train Logistic Regression Model
# --------------------------------------------------

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# --------------------------------------------------
# Step 6: Predict and Evaluate
# --------------------------------------------------

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy:.4f}")

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# --------------------------------------------------
# Step 7: Predict on Sample Inputs
# --------------------------------------------------

sample_data = {
    'pixel_inconsistency': [0.7, 0.3, 0.8],
    'facial_distortion': [0.8, 0.2, 0.9],
    'lighting_shadows': [0.7, 0.4, 0.8],
    'eye_reflection': [0.8, 0.1, 0.9],
    'motion_artifacts': [0.7, 0.2, 0.8]
}

sample_df = pd.DataFrame(sample_data)
sample_scaled = scaler.transform(sample_df)
sample_predictions = model.predict(sample_scaled)

# Display predictions
print("\n🔍 Sample Predictions:")
for idx, pred in enumerate(sample_predictions):
    label = "Deepfake (Yes)" if pred == 1 else "Real Image (No)"
    print(f"Sample Image {idx+1}: This is a {label}.")
