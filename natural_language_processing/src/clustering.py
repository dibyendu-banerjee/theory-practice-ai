# Chapter 20: Clustering

import matplotlib.pyplot as plt

import numpy as np

from sklearn.decomposition import TruncatedSVD

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score



# Sample Documents about India

documents = [

    "New Delhi is the capital of India, known for its historical landmarks like India Gate, Red Fort, and Qutub Minar.",

    "Mumbai, the financial capital of India, is famous for Bollywood, the Gateway of India, and Marine Drive.",

    "Kolkata, formerly known as Calcutta, is known for its colonial architecture, cultural festivals, and literary history.",

    "Chennai, the cultural capital of South India, is famous for its temples, classical music, and delicious South Indian cuisine.",

    "Bangalore is the tech hub of India, known for its IT parks, the Silicon Valley of India, and vibrant startup culture.",

    "Hyderabad, known for its rich history, the Charminar, and delicious Hyderabadi biryani, is also a growing IT center.",

    "Jaipur, the Pink City, is known for its majestic palaces, forts, and vibrant Rajasthani culture.",

    "Varanasi, one of the oldest cities in the world, is a spiritual hub, famous for its ghats is) using Truncated SVD

lsa = TruncatedSVD(n_components=2, random_state=42)  # Reduce to 2 dimensions for visualization

lsa_topic_matrix = lsa.fit_transform(X)



# Step 3: Perform KMeans clustering on the LSA topic matrix

kmeans = KMeans(n_clusters=5, random_state=42)  # Increased number of clusters for diversity

kmeans.fit(lsa_topic_matrix)



# Step 4: Visualize the results using a scatter plot

plt.figure(figsize=(12, 8))



# Plot each document as a point in the 2D LSA space

scatter = plt.scatter(lsa_topic_matrix[:, 0], lsa_topic_matrix[:, 1], c=kmeans.labels_, cmap='viridis', s=100, edgecolor='black')



# Plot cluster centers

centers = kmeans.cluster_centers_

plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Cluster Centers')



# Annotate points with document index

for i, txt in enumerate(range(len(documents))):

    plt.annotate(txt, (lsa_topic_matrix[i, 0], lsa_topic_matrix[i, 1]), fontsize=12, color='black', weight='bold')



# Title and Labels

plt.title("LSA for Document Clustering of Indian Cities and Landmarks", fontsize=16)

plt.xlabel("LSA Component 1")

plt.ylabel("LSA Component 2")



# Display colorbar

plt.colorbar(scatter, label='Cluster Label')



# Show the plot

plt.legend()

plt.grid(True)

plt.show()



# Step 5: Print out the cluster assignments and the top words for each cluster

# 5a: Print the cluster assignments for each document

print("\nCluster Assignments:")

for i, label in enumerate(kmeans.labels_):

    print(f"Document {i}: Cluster {label}")



# 5b: Print the top words for each cluster

print("\nTop Words for Each Cluster:")

n_top_words = 10  # Number of top words to display

feature_names = vectorizer.get_feature_names_out()

for cluster_idx in range(kmeans.n_clusters):

    print(f"\nCluster {cluster_idx}:")

    cluster_center = kmeans.cluster_centers_[cluster_idx]

    top_indices = cluster_center.argsort()[-n_top_words:][::-1]

    top_words = [feature_names[i] for i in top_indices]

    print("Top words:", ", ".join(top_words))



# Step 6: Calculate Silhouette Score (optional)

sil_score = silhouette_score(lsa_topic_matrix, kmeans.labels_)

print(f"\nSilhouette Score: {sil_score:.2f}")