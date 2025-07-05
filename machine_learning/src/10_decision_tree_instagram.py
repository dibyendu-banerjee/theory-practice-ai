"""# **Chapter 6: Support Vector Machines**

# Chapter 6: Code Example of Linear SVM
"""

import numpy as np

import matplotlib.pyplot as plt

from sklearn import datasets

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn.metrics import classification_report, accuracy_score



# Create a synthetic dataset

np.random.seed(0)

X, y = datasets.make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=4)



# Split the data into training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



# Train a Linear SVM model

model = SVC(kernel='linear', C=1.0)

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

plt.title('Linear SVM Decision Boundary')

plt.xlabel('Feature 1')

plt.ylabel('Feature 2')

plt.show()