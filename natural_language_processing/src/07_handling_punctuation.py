"""
Handling Punctuation
This script demonstrates: Handling Punctuation
"""

# Sample text with punctuation and special characters

text = "Hello! How are you today Sourav? 😊 Let's clean this text: #NLP #DataScience."



# Process the text with SpaCy

doc = nlp(text)



# Extract tokens and filter out punctuation

cleaned_words = [token.text for token in doc if not token.is_punct and not token.is_space]



# Output the cleaned words

print("Cleaned Words