# ================================================================
# Chapter 6: Support Vector Machines (SVM)
# Use Case: Linear SVM for Binary Classification on Synthetic Data
# ================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# --------------------------------------------------
# Step 1: Generate Synthetic Binary Classification Dataset
# --------------------------------------------------

np.random.seed(0)  # For reproducibility

# Create 2D dataset with 2 informative features and no redundancy
X, y = datasets.make_classification(
    n_samples=100, n_features=2, n_informative=2,
    n_redundant=0, random_state=4
)

# --------------------------------------------------
# Step 2: Split Data into Training and Testing Sets
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# --------------------------------------------------
# Step 3: Train a Linear SVM Classifier
# --------------------------------------------------

model = SVC(kernel='linear', C=1.0)
model.fit(X_train, y_train)

# --------------------------------------------------
# Step 4: Make Predictions and Evaluate
# --------------------------------------------------

y_pred = model.predict(X_test)

print(f"\n✅ Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred))

# --------------------------------------------------
# Step 5: Visualize the Decision Boundary
# --------------------------------------------------

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='autumn', edgecolor='k', s=30)
plt.title('🧭 Linear SVM Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Create mesh grid for decision boundary
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx, yy = np.meshgrid(
    np.linspace(xlim[0], xlim[1], 500),
    np.linspace(ylim[0], ylim[1], 500)
)

# Compute decision function for each point in the grid
Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundary and margins
plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1],
            linestyles=['--', '-', '--'], alpha=0.7)

plt.tight_layout()
plt.show()
