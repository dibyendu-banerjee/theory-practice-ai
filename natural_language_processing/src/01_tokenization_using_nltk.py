# ================================================================
# Chapter 11: Text Preprocessing
# Use Case: Tokenizing Text into Words and Sentences using NLTK
# ================================================================

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# Download required NLTK resources (only needed once)
nltk.download('punkt')

# --------------------------------------------------
# Step 1: Define Sample Text
# --------------------------------------------------

text = "My name is Dibyendu Banerjee. I am exploring Python."

# --------------------------------------------------
# Step 2: Tokenize Text into Words
# --------------------------------------------------

word_tokens = word_tokenize(text)
print("📝 Word Tokens:")
print(word_tokens)

# --------------------------------------------------
# Step 3: Tokenize Text into Sentences
# --------------------------------------------------

sent_tokens = sent_tokenize(text)
print("\n📚 Sentence Tokens:")
print(sent_tokens)
