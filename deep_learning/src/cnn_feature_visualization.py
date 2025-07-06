# ================================================================
# File: cnn_feature_visualization.py
# Description: This script demonstrates how to visualize feature maps 
# from convolutional layers in a CNN trained on the CIFAR-10 dataset. 
# It helps understand what the network learns at each layer.
#
# Author: Dibyendu Banerjee
# ================================================================

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# Step 1: Load and preprocess data
# -------------------------------

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32') / 255