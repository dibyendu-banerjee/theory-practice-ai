"""
Text Normalization
This script demonstrates: Text Normalization
"""

text = "HeLLo! I can't beleive it's already 2moro. I'm excited but a little nevous abt the trip!"



# Convert text to lowercase

text = text.lower()

pip install nltk autocorrect

import nltk

nltk.download('punkt')  # For tokenization

nltk.download('words')  # For English word corpus

# Expand contractions (manual implementation or use external libraries)

contractions = {

    "i'm": "i am", "it's": "it is", "can't": "cannot", "don't": "do not", "i’ve": "i have", "he'll": "he will",

    "2moro": "tomorrow", "abt": "about", "nevous": "nervous"

}

tokens = word_tokenize(text)



# Expand contractions

expanded_tokens = [contractions.get(word, word) for word in tokens]



# Correct spelling using Autocorrect

spell = Speller(lang='en')

corrected_tokens = [spell(token) for token in expanded_tokens]



# Join tokens back into text

normalized_text = ' '.join(corrected_tokens)

print("Normalized Text