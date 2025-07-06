"""
CBOW 3D Visualization
This script demonstrates: CBOW 3D Visualization
"""

# Extract all word vectors (excluding the padding token)

word_vectors = cbow_model.wv[cbow_model.wv.index_to_key]



# Set a reasonable value for perplexity, keeping it smaller than the number of samples (words)

perplexity_value = min(len(word_vectors) - 1, 30)  # Perplexity should be < number of words



# Reduce dimensionality of word vectors to 3D using t-SNE with the correct perplexity

tsne_model = TSNE(n_components=3, random_state=42, perplexity=perplexity_value)

word_vectors_3d = tsne_model.fit_transform(word_vectors)



# Create a 3D plot

fig = plt.figure(figsize=(14, 12))

ax = fig.add_subplot(111, projection='3d')



# Scatter plot of the word vectors

ax.scatter(word_vectors_3d[:, 0], word_vectors_3d[:, 1], word_vectors_3d[:, 2], edgecolors='k', c='b', alpha=0.7)



# Annotate each point with the corresponding word

for i, word in enumerate(cbow_model.wv.index_to_key):

    ax.text(word_vectors_3d[i, 0] + 0.02, word_vectors_3d[i, 1] + 0.02, word_vectors_3d[i, 2] + 0.02, word, fontsize=12)



ax.set_title("3D Visualization of Word Embeddings (CBOW Model) for Kolkata City", fontsize=15)

plt.show()