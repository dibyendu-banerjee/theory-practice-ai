# Chapter 18: TF-IDF-Based Filtering

from sklearn.feature_extraction.text import TfidfVectorizer



# Sample text

text = ["The quick brown fox jumps over the lazy dog.",

        "The dog is lazy and loves to sleep."]



# Create a TF-IDF vectorizer

vectorizer = TfidfVectorizer()



# Fit and transform the text

tfidf_matrix = vectorizer.fit_transform(text)



# Get the vocabulary and TF-IDF scores

vocabulary = vectorizer.get_feature_names_out()

tfidf_scores = tfidf_matrix.toarray()



# Set threshold for TF-IDF scores

threshold = 0.1



# Filter words based on TF-IDF scores

filtered_tokens = []

for i in range(tfidf_scores.shape[0]):

    filtered_tokens.extend([word for word, score in zip(vocabulary, tfidf_scores[i]) if score > threshold])



print("Filtered Tokens (TF-IDF-Based Filtering):", set(filtered_tokens))