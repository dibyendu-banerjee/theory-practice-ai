# ================================================================
# File: cnn_image_classification.py
# Description: This script demonstrates how to build a simple 
# Convolutional Neural Network (CNN) from scratch using TensorFlow/Keras 
# to classify images from the CIFAR-10 dataset. It includes data 
# preprocessing, model training, and evaluation.
#
# Author: Dibyendu Banerjee, Sourav Kairi
# ================================================================

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

# -------------------------------
# Step 1: Load and preprocess data
# -------------------------------

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Display dataset shape
print(f"Training data shape: {x_train.shape}")
print(f"Testing data shape: {x_test.shape}")

# Visualize the first training image
plt.imshow(x_train[0])
plt.title(f"Label: {y_train[0].argmax()}")
plt.axis('off')
plt.show()

# -------------------------------
# Step 2: Build the CNN model
# -------------------------------

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')  # 10 classes for CIFAR-10
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# -------------------------------
# Step 3: Train the model
# -------------------------------

model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

# -------------------------------
# Step 4: Evaluate the model
# -------------------------------

test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
