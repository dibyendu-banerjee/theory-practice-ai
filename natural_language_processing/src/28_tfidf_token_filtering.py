# ================================================================
# Chapter 18: TF-IDF-Based Filtering
# Use Case: Extracting Informative Words from Text Using TF-IDF Scores
# ================================================================

from sklearn.feature_extraction.text import TfidfVectorizer

# --------------------------------------------------
# Step 1: Define Sample Text Corpus
# --------------------------------------------------

documents = [
    "The quick brown fox jumps over the lazy dog.",
    "The dog is lazy and loves to sleep."
]

# --------------------------------------------------
# Step 2: Create and Fit TF-IDF Vectorizer
# --------------------------------------------------

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# --------------------------------------------------
# Step 3: Extract Vocabulary and TF-IDF Scores
# --------------------------------------------------

vocabulary = vectorizer.get_feature_names_out()
tfidf_scores = tfidf_matrix.toarray()

# --------------------------------------------------
# Step 4: Filter Words Based on TF-IDF Threshold
# --------------------------------------------------

threshold = 0.1  # Minimum score to consider a word informative
filtered_tokens = []

for i in range(tfidf_scores.shape[0]):
    for word, score in zip(vocabulary, tfidf_scores[i]):
        if score > threshold:
            filtered_tokens.append(word)

# --------------------------------------------------
# Step 5: Display Filtered Tokens
# --------------------------------------------------

print("🧠 Filtered Tokens (TF-IDF > 0.1):")
print(set(filtered_tokens))
