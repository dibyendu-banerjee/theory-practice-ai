"""
Clustering with KMeans
This script demonstrates: Clustering with KMeans
"""

# Sample data: A set of articles on different topics

articles = [

    "The government is planning new policies for healthcare reform.",

    "Basketball is a popular sport, and the NBA finals are coming soon.",

    "The economy is growing rapidly in the US due to new policies.",

    "Artificial Intelligence is changing the tech landscape.",

    "Football is a sport loved by millions around the world.",

    "Robotics and AI are closely related fields in technology.",

    "The latest elections were held in the US, with major debates.",

    "Technology companies are investing heavily in AI and robotics."

]



# Step 1: Vectorization - Convert text data into numerical format using TF-IDF

# This will transform the text into a matrix of numerical features based on the frequency of words in each article

vectorizer = TfidfVectorizer(stop_words='english')  # Remove common English stop words

X = vectorizer.fit_transform(articles)



# Step 2: Clustering - Apply KMeans clustering to group similar articles together

# We will specify 3 clusters, as we expect 3 topics: Politics, Sports, and Technology

kmeans = KMeans(n_clusters=3, random_state=42)

kmeans.fit(X)



# Step 3: Extracting the cluster labels

# These are the cluster labels assigned to each article

labels = kmeans.labels_



# Step 4: Displaying the articles and their assigned cluster labels

print("Clustering Results:")

for i, label in enumerate(labels):

    print(f"Article: '{articles[i]}' -> Cluster: {label}")



# Step 5: Display the top terms (words) contributing to each cluster

terms = vectorizer.get_feature_names_out()



print("\nTop terms for each cluster:")

for i in range(3):  # We have 3 clusters

    print(f"\nCluster {i}:")

    cluster_center = kmeans.cluster_centers_[i]  # The center of the cluster (centroid)

    sorted_term_indices = cluster_center.argsort()[-5:][::-1]  # Get the top 5 terms

    top_terms = [terms[index] for index in sorted_term_indices]

    print("Top terms: