# ================================================================
# File: model_transfer_learning.py
# Description: This script demonstrates transfer learning using 
# the ResNet50 model pre-trained on ImageNet. It adds custom 
# classification layers for a new dataset and performs training 
# and evaluation using synthetic image data.
#
# Author: Dibyendu Banerjee
# ================================================================

from keras.applications import ResNet50
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.applications.resnet50 import preprocess_input
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np

# -------------------------------
# Step 1: Prepare synthetic image data
# -------------------------------

# Simulate 100 RGB images of size 224x224 and 10 classes
X_data = np.random.rand(100, 224, 224, 3)
y_data = np.random.randint(0, 10, 100)

# Preprocess input for ResNet50
X_data = preprocess_input(X_data)

# One-hot encode labels
y_data = to_categorical(y_data, num_classes=10)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2)

# -------------------------------
# Step 2: Load pre-trained ResNet50
# -------------------------------

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

# -------------------------------
# Step 3: Add custom classification layers
# -------------------------------

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

# Define the full model
model = Model(inputs=base_model.input, outputs=predictions)

# -------------------------------
# Step 4: Compile and train the model
# -------------------------------

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32)

# -------------------------------
# Step 5: Evaluate the model
# -------------------------------

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
