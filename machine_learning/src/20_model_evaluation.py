# ================================================================
# Chapter 9: Machine Learning Model Evaluation
# Use Case: Evaluating Classifier Performance with Metrics & Visuals
# ================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)

# --------------------------------------------------
# Step 1: Generate Synthetic Classification Dataset
# --------------------------------------------------

X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=15, n_redundant=5,
    n_classes=2, random_state=42
)

# --------------------------------------------------
# Step 2: Split Data into Training and Testing Sets
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# --------------------------------------------------
# Step 3: Train a Random Forest Classifier
# --------------------------------------------------

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# --------------------------------------------------
# Step 4: Make Predictions and Compute Probabilities
# --------------------------------------------------

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# --------------------------------------------------
# Step 5: Calculate Evaluation Metrics
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# --------------------------------------------------
# Step 6: Display Metrics in Tabular Format
# --------------------------------------------------

metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'Score': [accuracy, precision, recall, f1]
})

# Plot metrics as a table
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(6, 2))
ax.axis('tight')
ax.axis('off')
ax.table(
    cellText=metrics_df.values,
    colLabels=metrics_df.columns,
    cellLoc='center',
    loc='center',
    colColours=["#f4f4f4"] * 2
)
plt.title("📊 Evaluation Metrics Summary")
plt.tight_layout()
plt.show()

# --------------------------------------------------
# Step 7: Print Raw Metrics and Confusion Matrix
# --------------------------------------------------

print("✅ Accuracy:", accuracy)
print("✅ Precision:", precision)
print("✅ Recall:", recall)
print("✅ F1 Score:", f1)
print("\n🧮 Confusion Matrix:\n", cm)

# --------------------------------------------------
# Step 8: Visualize Confusion Matrix
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
ax.set_title('Confusion Matrix')
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
plt.tight_layout()
plt.show()

# --------------------------------------------------
# Step 9: Plot ROC Curve
# --------------------------------------------------

fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.legend(loc="lower right")
plt.tight_layout()
plt.show()

# --------------------------------------------------
# Step 10: Check for Overfitting or Underfitting
# --------------------------------------------------

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print("\n🧪 Model Diagnostics:")
print(f"Training Accuracy: {train_score:.2f}")
print(f"Testing Accuracy: {test_score:.2f}")

if train_score - test_score > 0.1:
    print("⚠️ Potential Overfitting Detected.")
elif test_score < 0.6:
    print("⚠️ Potential Underfitting Detected.")
else:
    print("✅ Model appears well-balanced.")
