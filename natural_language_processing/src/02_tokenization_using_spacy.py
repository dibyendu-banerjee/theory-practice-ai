"""
Tokenization using SpaCy
This script demonstrates: Tokenization using SpaCy
"""

# Load the SpaCy model

nlp = spacy.load('en_core_web_sm')



# Sample text

text = "My name is Dibyendu Banerjee. I am exploring Python."



# Process the text

doc = nlp(text)



# Extract tokens

tokens = [token.text for token in doc]

print("Tokens