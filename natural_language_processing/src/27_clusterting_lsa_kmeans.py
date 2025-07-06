# ================================================================
# Document Clustering with LSA and KMeans
# Use Case: Clustering Descriptions of Indian Cities and Landmarks
# ================================================================

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# --------------------------------------------------
# Step 1: Define Sample Documents
# --------------------------------------------------

documents = [
    "New Delhi is the capital of India, known for its historical landmarks like India Gate, Red Fort, and Qutub Minar.",
    "Mumbai, the financial capital of India, is famous for Bollywood, the Gateway of India, and Marine Drive.",
    "Kolkata, formerly known as Calcutta, is known for its colonial architecture, cultural festivals, and literary history.",
    "Chennai, the cultural capital of South India, is famous for its temples, classical music, and delicious South Indian cuisine.",
    "Bangalore is the tech hub of India, known for its IT parks, the Silicon Valley of India, and vibrant startup culture.",
    "Hyderabad, known for its rich history, the Charminar, and delicious Hyderabadi biryani, is also a growing IT center.",
    "Jaipur, the Pink City, is known for its majestic palaces, forts, and vibrant Rajasthani culture.",
    "Varanasi, one of the oldest cities in the world, is a spiritual hub, famous for its ghats and rituals on the Ganges River."
]

# --------------------------------------------------
# Step 2: Vectorize Text Using TF-IDF and Apply LSA
# --------------------------------------------------

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

# Reduce dimensionality using Truncated SVD (LSA)
lsa = TruncatedSVD(n_components=2, random_state=42)
lsa_topic_matrix = lsa.fit_transform(X)

# --------------------------------------------------
# Step 3: Perform KMeans Clustering
# --------------------------------------------------

kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(lsa_topic_matrix)

# --------------------------------------------------
# Step 4: Visualize Clusters in 2D LSA Space
# --------------------------------------------------

plt.figure(figsize=(12, 8))
scatter = plt.scatter(
    lsa_topic_matrix[:, 0], lsa_topic_matrix[:, 1],
    c=kmeans.labels_, cmap='viridis', s=100, edgecolor='black'
)

# Plot cluster centers
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Cluster Centers')

# Annotate each point with its document index
for i in range(len(documents)):
    plt.annotate(str(i), (lsa_topic_matrix[i, 0], lsa_topic_matrix[i, 1]), fontsize=12, color='black', weight='bold')

plt.title("LSA for Document Clustering of Indian Cities and Landmarks", fontsize=16)
plt.xlabel("LSA Component 1")
plt.ylabel("LSA Component 2")
plt.colorbar(scatter, label='Cluster Label')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --------------------------------------------------
# Step 5: Print Cluster Assignments and Top Words
# --------------------------------------------------

print("\n📌 Cluster Assignments:")
for i, label in enumerate(kmeans.labels_):
    print(f"Document {i}: Cluster {label}")

print("\n🧠 Top Words for Each Cluster:")
n_top_words = 10
feature_names = vectorizer.get_feature_names_out()

# Note: Top words are based on original TF-IDF space, not LSA
for cluster_idx in range(kmeans.n_clusters):
    print(f"\nCluster {cluster_idx}:")
    cluster_center = kmeans.cluster_centers_[cluster_idx]
    top_indices = cluster_center.argsort()[-n_top_words:][::-1]
    top_words = [feature_names[i] for i in top_indices if i < len(feature_names)]
    print("Top words:", ", ".join(top_words))

# --------------------------------------------------
# Step 6: Evaluate Clustering with Silhouette Score
# --------------------------------------------------

sil_score = silhouette_score(lsa_topic_matrix, kmeans.labels_)
print(f"\n📈 Silhouette Score: {sil_score:.2f}")
