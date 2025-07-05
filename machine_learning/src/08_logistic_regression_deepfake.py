"""# **Chapter 5: Decision Tree**

# Use Case: Fraud Detection using Decision Tree
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Step 1: Create synthetic dataset (no file used)
np.random.seed(42)
n_samples = 200

data = pd.DataFrame({
    'Amount': np.random.uniform(10, 1000, n_samples),
    'TransactionType': np.random.choice(['Online', 'POS', 'ATM'], size=n_samples),
    'Location': np.random.choice(['Urban', 'Rural', 'Suburban'], size=n_samples),
    'Category': np.random.choice(['Groceries', 'Electronics', 'Clothing'], size=n_samples),
    'Fraud': np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15])  # 15% fraud
})

# Step 2: Preprocessing
label_encoder = LabelEncoder()
for col in ['TransactionType', 'Location', 'Category']:
    data[col] = label_encoder.fit_transform(data[col])

X = data.drop(columns=['Fraud'])
y = data['Fraud']

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train and evaluate base Decision Tree model
model = DecisionTreeClassifier(criterion='gini', random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("=== Base Decision Tree ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Step 5: Train and evaluate pruned Decision Tree (max_depth=5)
model_pruned = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
model_pruned.fit(X_train, y_train)
y_pred_pruned = model_pruned.predict(X_test)

print("\n=== Pruned Decision Tree (max_depth=5) ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_pruned) * 100:.2f}%")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_pruned))
print("Classification Report:")
print(classification_report(y_test, y_pred_pruned))