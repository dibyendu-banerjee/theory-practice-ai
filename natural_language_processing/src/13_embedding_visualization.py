"""
Embedding Visualization
This script demonstrates: Embedding Visualization
"""

# Reduce the dimensionality of the embeddings to 2D using t-SNE

tsne = TSNE(n_components=2, random_state=42)

reduced_embeddings = tsne.fit_transform(embeddings[1:])  # Skipping index 0 for padding



# Plot the embeddings

plt.figure(figsize=(10, 10))

for i, word in enumerate(word_index):

    plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1])

    plt.text(reduced_embeddings[i, 0] + 0.02, reduced_embeddings[i, 1] + 0.02, word, fontsize=12)



plt.title("Word Embeddings Visualization (Kolkata Data)")

plt.show()