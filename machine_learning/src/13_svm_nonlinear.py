# ================================================================
# Chapter 6: Support Vector Machines (SVM)
# Use Case: Identifying Malicious Web Traffic Using Non-Linear SVM
# ================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# Step 1: Generate Synthetic Web Traffic Dataset
# --------------------------------------------------

# Simulate 2D data with non-linear class boundaries (moons shape)
X, y = datasets.make_moons(n_samples=300, noise=0.2, random_state=42)

# --------------------------------------------------
# Step 2: Standardize Features
# --------------------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------------------------------
# Step 3: Create DataFrame for Inspection
# --------------------------------------------------

df = pd.DataFrame(X_scaled, columns=['Request Frequency', 'Request Size'])
df['Class'] = y  # 0 = Normal, 1 = Malicious

print("📋 Sample of the dataset:")
print(df.head())

# --------------------------------------------------
# Step 4: Train Non-Linear SVM with RBF Kernel
# --------------------------------------------------

clf = SVC(kernel='rbf', C=1.0, gamma='auto')
clf.fit(X_scaled, y)

# --------------------------------------------------
# Step 5: Create Mesh Grid for Decision Boundary
# --------------------------------------------------

h = 0.02  # Step size for mesh grid resolution
x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, h),
    np.arange(y_min, y_max, h)
)

# --------------------------------------------------
# Step 6: Predict Class Labels for Mesh Grid
# --------------------------------------------------

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# --------------------------------------------------
# Step 7: Visualize Decision Boundary and Data Points
# --------------------------------------------------

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, edgecolors='k', s=50, cmap=plt.cm.coolwarm)
plt.title('🛡️ SVM Decision Boundary for Malicious Web Traffic Detection')
plt.xlabel('Request Frequency')
plt.ylabel('Request Size')
plt.colorbar(label='Class (0 = Normal, 1 = Malicious)')
plt.tight_layout()
plt.show()
