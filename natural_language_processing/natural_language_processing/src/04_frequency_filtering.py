"""
Frequency Filtering
This script demonstrates: Frequency Filtering
"""

# Tokenization

tokens = word_tokenize(text)



# Count word frequencies

word_counts = Counter(tokens)



# Set frequency threshold

threshold = 1  # Words appearing more than once



# Remove words with frequency below the threshold

filtered_tokens = [word for word, count in word_counts.items() if count > threshold]



print("Filtered Tokens (Frequency-Based Filtering)