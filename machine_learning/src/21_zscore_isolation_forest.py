# ================================================================
# Chapter 10: Anomaly Detection
# Use Case: Isolation Forest for Detecting Anomalous Network Traffic
# ================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest

# --------------------------------------------------
# Step 1: Generate Synthetic Network Traffic Data
# --------------------------------------------------

np.random.seed(42)

# Simulate normal traffic (mean=100, std=20) and anomalous spikes (mean=500, std=50)
normal_traffic = np.random.normal(loc=100, scale=20, size=1000).reshape(-1, 1)
anomalous_traffic = np.random.normal(loc=500, scale=50, size=50).reshape(-1, 1)

# Combine into one dataset
traffic_data = np.vstack([normal_traffic, anomalous_traffic])
labels = np.concatenate([np.zeros(1000), np.ones(50)])  # 0 = normal, 1 = anomaly (ground truth)

# --------------------------------------------------
# Step 2: Train Isolation Forest Model
# --------------------------------------------------

model = IsolationForest(contamination=0.05, random_state=42)  # Assume ~5% anomalies
model.fit(traffic_data)

# Predict anomalies: -1 = anomaly, 1 = normal
predictions = model.predict(traffic_data)

# --------------------------------------------------
# Step 3: Create DataFrame for Visualization
# --------------------------------------------------

df_traffic = pd.DataFrame({
    'Traffic': traffic_data.flatten(),
    'TrueLabel': labels,
    'Anomaly': predictions
})

# --------------------------------------------------
# Step 4: Visualize Anomaly Detection Results
# --------------------------------------------------

plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=np.arange(len(df_traffic)),
    y='Traffic',
    hue='Anomaly',
    data=df_traffic,
    palette={1: 'blue', -1: 'red'},
    s=50
)
plt.title('🚨 Network Traffic Anomaly Detection (Isolation Forest)')
plt.xlabel('Sample Index')
plt.ylabel('Traffic Value')
plt.legend(title="Prediction", labels=["Normal", "Anomalous"])
plt.tight_layout()
plt.show()

# --------------------------------------------------
# Step 5: Display Sample Output
# --------------------------------------------------

print("📋 Sample of Detected Anomalies:")
print(df_traffic.head())
