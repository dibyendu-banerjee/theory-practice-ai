"""# **Chapter 9: Machine Learning Model Evaluation**

# Evaluating Machine Learning Models: Metrics and Visualizations for Comprehensive Analysis
"""

import numpy as np

import pandas as pd

from sklearn.datasets import make_classification

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (

    accuracy_score, precision_score, recall_score, f1_score,

    confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay

)

import matplotlib.pyplot as plt

import seaborn as sns



# Set the plot style for a more stunning look

sns.set(style="whitegrid")



# Generate a synthetic dataset

X, y = make_classification(

    n_samples=1000, n_features=20, n_informative=15, n_redundant=5,

    n_classes=2, random_state=42

)



# Split the dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



# Train a Random Forest classifier

model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)



# Predictions

y_pred = model.predict(X_test)

y_pred_proba = model.predict_proba(X_test)[:, 1]



# Calculate evaluation metrics

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)



# Create a DataFrame to display the metrics in a table

metrics = {

    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],

    'Score': [accuracy, precision, recall, f1]

}



metrics_df = pd.DataFrame(metrics)



# Plot the metrics table

fig, ax = plt.subplots(figsize=(6, 2))  # Define size of the table plot

ax.axis('tight')

ax.axis('off')

ax.table(cellText=metrics_df.values, colLabels=metrics_df.columns, cellLoc='center', loc='center', colColours=["#f4f4f4"]*2)



plt.show()



# Print raw metrics

print("Accuracy:", accuracy)

print("Precision:", precision)

print("Recall:", recall)

print("F1 Score:", f1)

print("Confusion Matrix:\n", cm)



# Plot Confusion Matrix with Seaborn heatmap

fig, ax = plt.subplots(figsize=(6, 5))

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)

ax.set_title('Confusion Matrix')

ax.set_xlabel('Predicted')

ax.set_ylabel('True')

plt.show()



# Plot ROC Curve

fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

roc_auc = auc(fpr, tpr)



fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')

ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

ax.set_xlabel('False Positive Rate')

ax.set_ylabel('True Positive Rate')

ax.set_title('ROC Curve')

ax.legend(loc="lower right")

plt.show()



# Overfitting and Underfitting Check

train_score = model.score(X_train, y_train)

test_score = model.score(X_test, y_test)



print("\nModel Diagnostics:")

if train_score - test_score > 0.1:

    print("Potential Overfitting Detected.")

elif test_score < 0.6:

    print("Potential Underfitting Detected.")

else:

    print("Model is balanced.")