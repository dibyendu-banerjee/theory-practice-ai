"""# Chapter 6: Code Example of Non Linear SVM"""

import numpy as np

import matplotlib.pyplot as plt

from sklearn import datasets

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn.metrics import classification_report, accuracy_score



# Load the Iris dataset

iris = datasets.load_iris()

X = iris.data[:, :2]  # Use only the first two features for simplicity

y = iris.target

# We will classify only two classes for simplicity

X = X[y != 2]

y = y[y != 2]



# Split the dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



# Create a Non-Linear SVM model with RBF kernel

model = SVC(kernel='rbf', C=1.0, gamma='auto')

model.fit(X_train, y_train)



# Make predictions

y_pred = model.predict(X_test)



# Print accuracy and classification report

print("Accuracy:", accuracy_score(y_test, y_pred))

print("Classification Report:\n", classification_report(y_test, y_pred))



# Plot the decision boundary

plt.figure(figsize=(8, 6))

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='autumn', edgecolor='k', s=20)

ax = plt.gca()

xlim = ax.get_xlim()

ylim = ax.get_ylim()



xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 500),

                     np.linspace(ylim[0], ylim[1], 500))

Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)



plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,

             linestyles=['--', '-', '--'])

plt.title('Non-Linear SVM Decision Boundary with RBF Kernel')

plt.xlabel('Feature 1')

plt.ylabel('Feature 2')

plt.show()