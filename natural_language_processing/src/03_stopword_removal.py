# ================================================================
# Chapter 11: Text Preprocessing
# Use Case: Stopword Removal Using NLTK
# ================================================================

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# --------------------------------------------------
# Step 1: Download Required NLTK Resources
# --------------------------------------------------

nltk.download('punkt')
nltk.download('stopwords')

# --------------------------------------------------
# Step 2: Load English Stopwords
# --------------------------------------------------

stop_words = set(stopwords.words('english'))

# --------------------------------------------------
# Step 3: Define Sample Text
# --------------------------------------------------

text = "The quick brown fox jumps over the lazy dog."

# --------------------------------------------------
# Step 4: Tokenize Text
# --------------------------------------------------

tokens = word_tokenize(text)

# --------------------------------------------------
# Step 5: Remove Stopwords
# --------------------------------------------------

filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

# --------------------------------------------------
# Step 6: Display Results
# --------------------------------------------------

print("📝 Original Tokens:")
print(tokens)

print("\n🚫 Filtered Tokens (Stopwords Removed):")
print(filtered_tokens)
