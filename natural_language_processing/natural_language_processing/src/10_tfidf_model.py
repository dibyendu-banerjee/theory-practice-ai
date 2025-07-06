"""
TF-IDF Model
This script demonstrates: TF-IDF Model
"""

# Sample documents

documents = [

    "I love machine learning",

    "Machine learning is great",

    "I love coding"

]



# Initialize the CountVectorizer

# CountVectorizer is a tool in NLP that converts a collection of text documents into a matrix of token counts.

vectorizer = CountVectorizer()



# Fit the vectorizer to the documents and transform them into vectors

# This process involves two main steps:

# 1. It 'learns' the vocabulary from the documents.

# 2. It transforms the documents into vectors (where each document is represented by a vector of word counts).

X = vectorizer.fit_transform(documents)



# Convert the sparse matrix to a dense matrix and print the result

# The result is stored as a sparse matrix, but we convert it to a dense matrix (array) for easy reading.

bow_matrix = X.toarray()



# Display the vocabulary (the list of words found in the documents)

# The 'vocabulary_' attribute contains a dictionary where the keys are the words in the documents,

# and the values are the indices that represent the position of those words in the feature vector.

print("Vocabulary:", vectorizer.vocabulary_)



# Display the Bag-of-Words Matrix (the vector representation of each document)

# This matrix shows how many times each word in the vocabulary appears in each document.

# Each row corresponds to a document, and each column corresponds to a word in the vocabulary.

print("Bag-of-Words Matrix:\n", bow_matrix)

"""# Chapter 19: implement the TF-IDF model using Python and the Scikit-learn library."""

from sklearn.feature_extraction.text import TfidfVectorizer



# Sample documents

my_documents = [

    "I love machine learning",

    "Machine learning is great",

    "I love coding"

]



# Initialize the TfidfVectorizer

vectorizer = TfidfVectorizer()



# Fit the vectorizer to the documents and transform them into TF-IDF vectors

X = vectorizer.fit_transform(my_documents)

my_documents

# Convert the sparse matrix to a dense matrix and print the result

tfidf_matrix = X.toarray()



print("Here is  Vocabulary:", vectorizer.vocabulary_)

print("Here is the TF-IDF Matrix