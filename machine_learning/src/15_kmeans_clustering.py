"""# Use Case: Hierarchical Clustering for Employee Performance Analysis"""

import pandas as pd

import numpy as np

from sklearn.preprocessing import StandardScaler

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

import matplotlib.pyplot as plt



# Step 1: Create sample employee performance data with updated Employee_ID format

data = {

    'Employee_ID': [f'EMP{str(i).zfill(5)}' for i in range(1, 11)],  # Generates EMP00001, EMP00002, ..., EMP00010

    'Task_Completion_Rate': [90, 70, 85, 60, 95, 50, 80, 55, 88, 65],

    'Quality_Score': [4.8, 4.0, 4.5, 3.5, 5.0, 3.0, 4.2, 3.2, 4.6, 3.8],

    'Hours_Worked': [40, 38, 42, 35, 45, 30, 39, 33, 41, 36],

    'Peer_Review_Score': [9, 7, 8, 6, 10, 5, 8, 5, 9, 6]

}

df = pd.DataFrame(data)



# Step 2: Preprocess the data

features = df[['Task_Completion_Rate', 'Quality_Score', 'Hours_Worked', 'Peer_Review_Score']]

scaler = StandardScaler()

scaled_features = scaler.fit_transform(features)



# Step 3: Apply hierarchical clustering

linkage_matrix = linkage(scaled_features, method='ward')



# Step 4: Plot dendrogram to visualize clustering

plt.figure(figsize=(10, 7))

dendrogram(linkage_matrix, labels=df['Employee_ID'].values, leaf_rotation=90, leaf_font_size=10)

plt.title("Dendrogram for Employee Clustering")

plt.xlabel("Employees")

plt.ylabel("Euclidean Distance")

plt.show()



# Step 5: Cut the dendrogram to form clusters

optimal_clusters = 3  # Number of clusters determined based on dendrogram

df['Cluster'] = fcluster(linkage_matrix, optimal_clusters, criterion='maxclust')



# Step 6: Analyze the clusters

for cluster in sorted(df['Cluster'].unique()):

    print(f"Cluster {cluster}:")

    print(df[df['Cluster'] == cluster])

    print()