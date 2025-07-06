"""
Stopword Removal
This script demonstrates: Stopword Removal
"""

# Load stopwords

stop_words = set(stopwords.words('english'))



# Sample text

text = "The quick brown fox jumps over the lazy dog."



# Tokenize text

tokens = word_tokenize(text)



# Remove stopwords

filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

print("Filtered Tokens (Predefined Lists)