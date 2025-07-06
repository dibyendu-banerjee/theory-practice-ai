"""
Stemming and Lemmatization
This script demonstrates: Stemming and Lemmatization
"""

# Load the SpaCy English model

nlp = spacy.load("en_core_web_sm")



# Sample text for demonstration

text = "The runner was running faster than before. The geese were flying happily in the sky."



# Tokenize and process the text using SpaCy

doc = nlp(text)



# Lemmatization using SpaCy

lemmas = [token.lemma_ for token in doc]

tokens = [token.text for token in doc]



# Stemming using NLTK

porter_stemmer = PorterStemmer()

snowball_stemmer = SnowballStemmer("english")

lancaster_stemmer = LancasterStemmer()



porter_stems = [porter_stemmer.stem(token) for token in tokens]

snowball_stems = [snowball_stemmer.stem(token) for token in tokens]

lancaster_stems = [lancaster_stemmer.stem(token) for token in tokens]



# Output Results

print("Original Tokens:", tokens)

print("Lemmatized Tokens (SpaCy):", lemmas)

print("Porter Stemmer (NLTK):", porter_stems)

print("Snowball Stemmer (NLTK):", snowball_stems)

print("Lancaster Stemmer