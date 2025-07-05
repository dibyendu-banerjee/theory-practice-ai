"""# Chapter 6 - Use Case: Identifying Malicious Web Traffic using SVM"""

import numpy as np

import matplotlib.pyplot as plt

from sklearn import datasets

from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler

import pandas as pd



# Step 1: Generate a synthetic 2D dataset (representing web traffic features)

X, y = datasets.make_moons(n_samples=300, noise=0.2, random_state=42)



# Step 2: Standardize the features (important for SVM performance)

scaler = StandardScaler()

X = scaler.fit_transform(X)



# Step 3: Create a DataFrame to display the dataset in tabular format

df = pd.DataFrame(X, columns=['Request Frequency', 'Request Size'])

df['Class'] = y  # Add the target class (0 = normal, 1 = malicious)



# Optional: Print the first few rows of the dataset to understand the structure

print(df.head())



# Step 4: Train a non-linear SVM with RBF kernel

clf = SVC(kernel='rbf', C=1, gamma='auto')

clf.fit(X, y)



# Step 5: Create a mesh grid for plotting decision boundary

h = .02  # Step size in the mesh grid

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1

y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))



# Step 6: Predict the class labels for each point in the mesh grid

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)



# Step 7: Plot decision boundary and data points

plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)  # Decision boundary with shading

plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=50, cmap=plt.cm.coolwarm)  # Data points



# Step 8: Add labels and title to the plot

plt.title('Non-Linear SVM Decision Boundary (RBF Kernel)')

plt.xlabel('Request Frequency')

plt.ylabel('Request Size')

plt.colorbar()  # To add a color bar indicating the class labels



# Step 9: Show the plot

plt.show()