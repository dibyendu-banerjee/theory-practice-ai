"""# **Chapter 8: Unsupervised Learning Algorithms**

# Real-Life Example of K-Means Clustering in Python
"""

# Importing required libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler



# Sample dataset creation

data = {

    "CustomerID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],

    "Annual Income (k$)": [15, 16, 17, 28, 29, 70, 71, 72, 80, 81],

    "Spending Score": [39, 81, 6, 77, 40, 76, 6, 94, 3, 72]

}

df = pd.DataFrame(data)



# Selecting features for clustering

X = df[["Annual Income (k$)", "Spending Score"]]



# Standardizing the data (important for K-Means)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)



# Applying the Elbow Method to determine the optimal number of clusters

inertia = []

for k in range(1, 10):

    kmeans = KMeans(n_clusters=k, random_state=42)

    kmeans.fit(X_scaled)

    inertia.append(kmeans.inertia_)



# Plotting the Elbow Method

plt.figure(figsize=(8, 5))

plt.plot(range(1, 10), inertia, marker='o')

plt.title('Elbow Method for Optimal K')

plt.xlabel('Number of Clusters (k)')

plt.ylabel('Inertia')

plt.grid(True)

plt.show()



# Based on the elbow plot, we choose k=3 (assume it looks optimal)

# Applying K-Means clustering

kmeans = KMeans(n_clusters=3, random_state=42)

df["Cluster"] = kmeans.fit_predict(X_scaled)



# Visualizing the clusters

plt.figure(figsize=(8, 5))

for cluster in range(3):

    plt.scatter(

        df.loc[df["Cluster"] == cluster, "Annual Income (k$)"],

        df.loc[df["Cluster"] == cluster, "Spending Score"],

        label=f"Cluster {cluster}"

    )



# Plotting cluster centroids

centroids = scaler.inverse_transform(kmeans.cluster_centers_)

plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', label='Centroids', marker='X')

plt.title("Customer Segmentation with K-Means Clustering")

plt.xlabel("Annual Income (k$)")

plt.ylabel("Spending Score")

plt.legend()

plt.grid(True)

plt.show()



# Displaying the clustered dataset

print(df)