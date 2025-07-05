"""# Use Case: Fraud Detection and Anomaly Detection Using Z-Score and Isolation Forest in Python"""

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats



# Generate synthetic transaction data

np.random.seed(42)

normal_transactions = np.random.normal(100, 20, 1000)  # Normal transactions with mean=100, std=20

fraudulent_transactions = np.random.normal(500, 50, 50)  # Fraudulent transactions with a much higher amount



# Combine normal and fraudulent transactions

all_transactions = np.concatenate([normal_transactions, fraudulent_transactions])

labels = np.concatenate([np.zeros(1000), np.ones(50)])  # 0: normal, 1: fraudulent



# Create a DataFrame

df = pd.DataFrame({'Transaction Amount': all_transactions, 'Label': labels})



# Calculate Z-Score

df['Z-Score'] = (df['Transaction Amount'] - df['Transaction Amount'].mean()) / df['Transaction Amount'].std()



# Flag fraudulent transactions using a Z-Score threshold

threshold = 3

df['Fraudulent'] = df['Z-Score'].apply(lambda x: 1 if x > threshold else 0)



# Plotting the results

plt.figure(figsize=(10,6))

sns.histplot(df[df['Fraudulent'] == 1]['Transaction Amount'], color='red', kde=True, label='Fraudulent', bins=30)

sns.histplot(df[df['Fraudulent'] == 0]['Transaction Amount'], color='blue', kde=True, label='Normal', bins=30)

plt.title('Fraudulent Transactions vs Normal Transactions (Z-Score)')

plt.xlabel('Transaction Amount')

plt.ylabel('Frequency')

plt.legend()

plt.show()



# Show the first few rows with detected fraudulent transactions

print(df.head())