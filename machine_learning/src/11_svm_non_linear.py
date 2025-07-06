# ================================================================
# Chapter 6: Support Vector Machines (SVM)
# Use Case: Non-Linear SVM with RBF Kernel on Iris Dataset
# ================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# --------------------------------------------------
# Step 1: Load and Prepare the Iris Dataset
# --------------------------------------------------

# Load Iris dataset and use only the first two classes and features
iris = datasets.load_iris()
X = iris.data[:, :2]  # Use only first two features for 2D visualization
y = iris.target

# Filter to binary classification (class 0 and 1 only)
X = X[y != 2]
y = y[y != 2]

# --------------------------------------------------
# Step 2: Split Data into Training and Testing Sets
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# --------------------------------------------------
# Step 3: Train a Non-Linear SVM with RBF Kernel
# --------------------------------------------------

model = SVC(kernel='rbf', C=1.0, gamma='auto')
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
plt.title('🌐 Non-Linear SVM Decision Boundary (RBF Kernel)')
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
