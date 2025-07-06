"""
GloVe Visualization
This script demonstrates: GloVe Visualization
"""

# Load the pre-trained word embeddings model

model = api.load("glove-wiki-gigaword-50")  # 50-dimensional GloVe model



# Define categories and words for visualization

categories = {

    'Animals': ['cat', 'dog', 'wolf', 'lion', 'tiger', 'bear', 'elephant', 'giraffe', 'zebra', 'horse', 'rabbit', 'fox', 'deer'],

    'Colors': ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown', 'black', 'white', 'gray', 'cyan', 'magenta'],

    'Countries': ['USA', 'Canada', 'Germany', 'France', 'Japan', 'China', 'Brazil', 'India', 'Australia', 'Russia', 'Mexico', 'Italy', 'South Africa'],

    'Foods': ['pizza', 'burger', 'sushi', 'pasta', 'salad', 'apple', 'banana', 'cherry', 'grape', 'steak', 'bread', 'cheese', 'chocolate'],

    'Emotions': ['happy', 'sad', 'angry', 'excited', 'bored', 'fearful', 'surprised', 'nervous', 'relieved', 'disgusted', 'joyful', 'anxious', 'content']

}



# Flatten the list of words and create a category list

words = [word for category in categories.values() for word in category]

categories_flat = [cat for cat, words_list in categories.items() for _ in words_list]



# Filter words to include only those present in the model

words = [word for word in words if word in model]

categories_flat = [cat for word, cat in zip(words, categories_flat) if word in model]



# Get word vectors

word_vectors = np.array([model[word] for word in words])



# Check if there are enough words for t-SNE

if len(word_vectors) < 2:

    raise ValueError("Not enough words to perform t-SNE. Please use more words.")



# Apply t-SNE to reduce dimensionality to 2D

perplexity = min(30, len(word_vectors) - 1)  # Adjusted for a larger number of samples

tsne = TSNE(n_components=2, perplexity=perplexity, random_state=0)

word_vectors_2d = tsne.fit_transform(word_vectors)



# Create a 2D scatter plot

plt.figure(figsize=(18, 14))

ax = plt.gca()



# Define colors for each category

colors = {

    'Animals': 'red',

    'Colors': 'blue',

    'Countries': 'green',

    'Foods': 'orange',

    'Emotions': 'purple'

}

color_map = [colors[cat] for cat in categories_flat]



# Plot each category with different colors and larger markers

scatter = plt.scatter(word_vectors_2d[:, 0], word_vectors_2d[:, 1], c=color_map, marker='o', s=100, alpha=0.8, edgecolors='w', linewidth=0.5)



# Add legend with a background color

handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[cat], markersize=15, label=cat) for cat in colors]

plt.legend(handles=handles, title='Categories', loc='upper right', frameon=True, facecolor='lightgrey')



# Annotate each point with the corresponding word using a bold font

for i, word in enumerate(words):

    plt.annotate(word, (word_vectors_2d[i, 0], word_vectors_2d[i, 1]), fontsize=12, weight='bold', alpha=0.9)



plt.title('2D Visualization of Word Embeddings using t-SNE', fontsize=16, weight='bold')

plt.xlabel('Dimension 1', fontsize=14, weight='bold')

plt.ylabel('Dimension 2', fontsize=14, weight='bold')

plt.grid(True, linestyle='--', alpha=0.6)

plt.show()