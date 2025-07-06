# ================================================================
# File: cnn_transfer_learning.py
# Description: This script demonstrates transfer learning using 
# MobileNetV2 pre-trained on ImageNet, fine-tuned on the CIFAR-10 dataset. 
# It includes data preprocessing, model customization, training, and evaluation.
#
# Author: Dibyendu Banerjee, Sourav Kairi
# ================================================================

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# -------------------------------
# Step 1: Load and preprocess CIFAR-10 data
# -------------------------------

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# -------------------------------
# Step 2: Load pre-trained MobileNetV2
# -------------------------------

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# -------------------------------
# Step 3: Build the transfer learning model
# -------------------------------

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')  # 10 classes for CIFAR-10
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# -------------------------------
# Step 4: Train the model
# -------------------------------

history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

# -------------------------------
# Step 5: Evaluate the model
# -------------------------------

test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
