# ================================================================
# File: model_regularization.py
# Description: This script demonstrates two regularization techniques 
# in deep learning using Keras: L2 regularization and Dropout. 
# It uses synthetic classification data to show how these methods 
# help prevent overfitting.
#
# Author: Dibyendu Banerjee
# ================================================================

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2

# -------------------------------
# Part 1: L2 Regularization
# -------------------------------

# Generate synthetic classification data
X_l2, y_l2 = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train_l2, X_test_l2, y_train_l2, y_test_l2 = train_test_split(X_l2, y_l2, test_size=0.2)

# Build a neural network with L2 regularization
model_l2 = Sequential()
model_l2.add(Dense(64, input_dim=10, activation='relu', kernel_regularizer=l2(0.01)))
model_l2.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01)))
model_l2.add(Dense(1, activation='sigmoid'))

# Compile and train
model_l2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_l2.fit(X_train_l2, y_train_l2, epochs=10, batch_size=32)

# Evaluate
loss_l2, acc_l2 = model_l2.evaluate(X_test_l2, y_test_l2)
print(f"[L2] Test Loss: {loss_l2:.4f}, Test Accuracy: {acc_l2:.4f}")

# -------------------------------
# Part 2: Dropout Regularization
# -------------------------------

# Generate new synthetic data
X_do, y_do = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train_do, X_test_do, y_train_do, y_test_do = train_test_split(X_do, y_do, test_size=0.2)

# Build a neural network with Dropout
model_do = Sequential()
model_do.add(Dense(64, input_dim=10, activation='relu'))
model_do.add(Dropout(0.5))  # Drop 50% of neurons during training
model_do.add(Dense(32, activation='relu'))
model_do.add(Dense(1, activation='sigmoid'))

# Compile and train
model_do.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_do.fit(X_train_do, y_train_do, epochs=10, batch_size=32)

# Evaluate
loss_do, acc_do = model_do.evaluate(X_test_do, y_test_do)
print(f"[Dropout] Test Loss: {loss_do:.4f}, Test Accuracy: {acc_do:.4f}")
