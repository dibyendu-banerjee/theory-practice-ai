"""
Tokenization using NLTK
This script demonstrates: Tokenization using NLTK
"""

# Sample text

text = "My name is Dibyendu Banerjee. I am exploring Python."



# Tokenize text into words

word_tokens = word_tokenize(text)

print("Word Tokens:", word_tokens)



# Tokenize text into sentences

sent_tokens = sent_tokenize(text)

print("Sentence Tokens