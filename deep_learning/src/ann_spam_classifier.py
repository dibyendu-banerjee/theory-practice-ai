# ================================================================
# File: ann_spam_classifier.py
# Description: This script demonstrates how to build a simple 
# Artificial Neural Network (ANN) using scikit-learn to classify 
# emails as spam or not spam based on basic features. It also 
# includes a conceptual example of an artificial neuron.
#
# Author: Dibyendu Banerjee
# ================================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# Part 1: ANN for Spam Detection
# -------------------------------

# Simulated dataset with basic email features
data = {
    "has_spam_words": [1, 0, 1, 0, 1, 0, 0, 1],
    "email_size": [300, 120, 450, 200, 500, 180, 150, 400],
    "link_count": [5, 0, 7, 1, 6, 1, 0, 8],
    "spam_label": [1, 0, 1, 0, 1, 0, 0, 1]  # 1 = spam, 0 = not spam
}

# Convert to DataFrame
email_data = pd.DataFrame(data)

# Split features and target
X = email_data.drop("spam_label", axis=1)
y = email_data["spam_label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define and train the ANN model
model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42, verbose=True)
model.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_scaled)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nDetailed Classification Report:\n", classification_report(y_test, y_pred))

# Plot training loss
plt.figure(figsize=(8, 6))
plt.plot(model.loss_curve_, color="blue", label="Training Loss")
plt.title("Training Loss Curve of Neural Network")
plt.xlabel("Iterations")
plt.ylabel("Loss Value")
plt.legend()
plt.grid(True)
plt.show()

# --------------------------------------------
# Part 2: Conceptual Example of a Single Neuron
# --------------------------------------------

class ArtificialNeuron:
    def __init__(self, weights, bias):
        self.weights = np.array(weights)
        self.bias = bias

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, inputs):
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        return self.sigmoid(weighted_sum)

# Example input: [has_spam_words, email_size, link_count]
email_features = [1, 450, 5]
weights = [0.7, 0.05, 0.1]
bias = -0.3

neuron = ArtificialNeuron(weights, bias)
prediction = neuron.predict(email_features)

if prediction > 0.5:
    print("The email is classified as Spam.")
else:
    print("The email is classified as Not Spam.")
