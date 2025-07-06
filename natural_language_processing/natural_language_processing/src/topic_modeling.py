# Chapter 20: Topic Modeling

# **Chapter 23: Topic Modeling**

# Chapter 23: Topic Extraction Using LDA

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.decomposition import LatentDirichletAllocation

from wordcloud import WordCloud



# Extended and diversified sample text data

corpus = [

    "Programming in Python is incredibly flexible, it can be used for various tasks such as web development, data analysis, and machine learning.",

    "Natural Language Processing (NLP) enables machines to understand human language, transforming the way we interact with computers and technology.",

    "Topic modeling, especially through methods like LDA, helps to extract hidden thematic structures from large volumes of text.",

    "Machine learning is part of artificial intelligence, allowing computers to learn from data and make predictions or decisions without being explicitly programmed.",

    "Deep learning models, particularly neural networks, have revolutionized fields such as image recognition and natural language understanding.",

    "Data science blends mathematics, programming, and domain knowledge to extract valuable insights from structured and unstructured data.",

    "Cloud computing is a crucial technology, offering scalable resources over the internet and enabling services like AWS and Google Cloud to flourish.",

    "Artificial intelligence, with deep learning and reinforcement learning at its core, is rapidly changing industries like healthcare, finance, and automotive.",

    "Blockchain technology underpins cryptocurrencies, offering decentralized security and transparency that could disrupt sectors like finance and supply chain management.",

    "Quantum computing is pushing the limits of computational power, with the potential to revolutionize industries like cryptography, material science, and optimization."

]



# Step 1: Convert text data into a term-document matrix using CountVectorizer

vectorizer = CountVectorizer(stop_words='english')

doc_term_matrix = vectorizer.fit_transform(corpus)



# Step 2: Perform LDA (Latent Dirichlet Allocation) to extract topics

lda_extractor = LatentDirichletAllocation(n_components=3, random_state=42)  # 3 topics for example

lda_extractor.fit(doc_term_matrix)



# Step 3: Displaying the most significant words in each topic

def display_topics_from_lda(model, feature_list, top_n_words=10):

    for topic_idx, topic in enumerate(model.components_):

        print(f"Topic #{topic_idx + 1}:")

        print(" ".join([feature_list[i] for i in topic.argsort()[:-top_n_words - 1:-1]]))

    print()



# Get feature names from vectorizer

feature_names = vectorizer.get_feature_names_out()

display_topics_from_lda(lda_extractor, feature_names)



# Step 4: Visualizing topics with word clouds

def visualize_word_clouds(lda_model, feature_list, num_topics=3):

    plt.figure(figsize=(18, 10))

    for t in range(num_topics):

        plt.subplot(1, num_topics, t + 1)

        topic_words = [feature_list[i] for i in lda_model.components_[t].argsort()[:-11:-1]]

        word_frequencies = {word: lda_model.components_[t][i] for i, word in enumerate(topic_words)}

        wordcloud = WordCloud(width=400, height=400, background_color='white').generate_from_frequencies(word_frequencies)

        plt.imshow(wordcloud, interpolation="bilinear")

        plt.axis('off')

        plt.title(f"Topic {t + 1}")

    plt.show()



# Visualize the word clouds for each topic

visualize_word_clouds(lda_extractor, feature_names)



# Step 5: Visualizing topic distribution per document (Bar Plot)

topic_distribution = lda_extractor.transform(doc_term_matrix)  # Get topic distribution for each document



# Prepare data for plotting

topic_distribution_df = pd.DataFrame(topic_distribution, columns=[f"Topic {i + 1}" for i in range(topic_distribution.shape[1])])



# Plotting the topic distribution for each document

plt.figure(figsize=(12, 7))

topic_distribution_df.plot(kind='bar', stacked=True, cmap='tab20', ax=plt.gca())

plt.title("Topic Distribution Across Documents")

plt.xlabel("Documents")

plt.ylabel("Proportion of Topics")

plt.xticks(range(len(corpus)), [f"Doc {i + 1}" for i in range(len(corpus))], rotation=0)

plt.legend(title="Topics", loc="upper left")

plt.tight_layout()

plt.show()

# Chapter 23: Code Example: Topic Modeling Using NMF

# Chapter 23: Code to Demonstrate using Evaluation techniques in Topic Modeling