# ================================================================
# File: knowledge_distillation.py
# Description: This script demonstrates knowledge distillation, where 
# a smaller student model learns from a larger pre-trained teacher model. 
# The teacher is based on ResNet50, and the student is a shallow dense network.
#
# Author: Dibyendu Banerjee, Sourav Kairi
# ================================================================

from keras.models import Sequential, Model
from keras.layers import Dense, Input, GlobalAveragePooling2D
from keras.applications import ResNet50
from keras.optimizers import Adam
from keras.utils import to_categorical
import numpy as np

# -------------------------------
# Step 1: Define the teacher model
# -------------------------------

# Load pre-trained ResNet50 without top layers
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
teacher_output = Dense(10, activation='softmax')(x)

# Build the teacher model
teacher_model = Model(inputs=base_model.input, outputs=teacher_output)

# Compile the teacher model
teacher_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# -------------------------------
# Step 2: Generate synthetic training data
# -------------------------------

X_train = np.random.random((100, 224, 224, 3))  # 100 RGB images
y_train = np.random.randint(0, 10, 100)         # 10 classes
y_train = to_categorical(y_train, num_classes=10)

# -------------------------------
# Step 3: Train the teacher model
# -------------------------------

teacher_model.fit(X_train, y_train, epochs=10)

# -------------------------------
# Step 4: Get soft targets from teacher
# -------------------------------

soft_targets = teacher_model.predict(X_train)

# -------------------------------
# Step 5: Define the student model
# -------------------------------

student_model = Sequential()
student_model.add(Dense(64, input_dim=224*224*3, activation='relu'))  # Flattened input
student_model.add(Dense(10, activation='softmax'))

# Compile the student model
student_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# -------------------------------
# Step 6: Train the student model on soft targets
# -------------------------------

X_train_flat = X_train.reshape(-1, 224*224*3)
student_model.fit(X_train_flat, soft_targets, epochs=10)
