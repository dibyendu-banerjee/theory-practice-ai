"""# Use Case: Isolation Forest: Anomaly Detection in Network Traffic"""

from sklearn.ensemble import IsolationForest

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



# Generate synthetic network traffic data

np.random.seed(42)

normal_traffic = np.random.normal(100, 20, 1000).reshape(-1, 1)  # Normal network traffic

anomalous_traffic = np.random.normal(500, 50, 50).reshape(-1, 1)  # Anomalous traffic spikes



# Combine normal and anomalous traffic

traffic_data = np.vstack([normal_traffic, anomalous_traffic])

labels = np.concatenate([np.zeros(1000), np.ones(50)])  # 0: normal, 1: anomalous



# Fit Isolation Forest model

model = IsolationForest(contamination=0.05)  # 5% of the data is anomalous

model.fit(traffic_data)



# Predict anomalies

df_traffic = pd.DataFrame({'Traffic': traffic_data.flatten(), 'Label': labels})

df_traffic['Anomaly'] = model.predict(traffic_data)



# Plotting the results

plt.figure(figsize=(10,6))

sns.scatterplot(data=df_traffic, x=np.arange(len(df_traffic)), y='Traffic', hue='Anomaly', palette={1: 'blue', -1: 'red'}, s=50)

plt.title('Network Traffic Anomaly Detection (Isolation Forest)')

plt.xlabel('Sample Index')

plt.ylabel('Traffic Value')

plt.legend(title="Anomaly", labels=["Normal", "Anomalous"])

plt.show()



# Show the first few rows with detected anomalies

print(df_traffic.head())