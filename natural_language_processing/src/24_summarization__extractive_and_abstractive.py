"""
Summarization - Extractive & Abstractive
This script demonstrates: Summarization - Extractive & Abstractive
"""

# Load pre-trained BERT model for NER

model = BertForTokenClassification.from_pretrained('dbmdz/bert-large-cased-finetuned-conll03-english')

tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-large-cased-finetuned-conll03-english')



nlp = pipeline('ner', model=model, tokenizer=tokenizer)



# Example text

text = "Rabindranath Tagore was born in West Bengal."



# Perform NER

ner_results = nlp(text)

print("NER results:", ner_results)

"""# Chapter 21: SpaCy"""

import spacy



def extract_entities_from_text(text):

    # Load the pre-trained SpaCy model for Named Entity Recognition

    nlp = spacy.load("en_core_web_sm")



    # Process the input text through the NLP pipeline

    doc = nlp(text)



    # Extract and return named entities along with their labels

    return [(entity.text, entity.label_) for entity in doc.ents]



# Example sentence for NER

text = "Tesla announced plans to build a new factory in Berlin in 2023."



# Extract and display the named entities

entities = extract_entities_from_text(text)



# Output the results

for entity in entities:

    print(f"Entity: {entity[0]}, Label: {entity[1]}"))

"""# Chapter 21: NLTK (Natural Language Toolkit)"""

import nltk
from nltk.chunk import ne_chunk
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# Example text
text = " "Albert Einstein developed the theory of relativity in the early 20th century.""

# Tokenize and tag parts of speech
tokens = word_tokenize(text)
tagged = pos_tag(tokens)

# Named Entity Recognition
entities = ne_chunk(tagged)
print(entities)

"""# Chapter 21:Stanford NER"""

from stanfordnlp.server import CoreNLPClient

client = CoreNLPClient(annotators=['ner'], timeout=30000, memory='16G')

# Annotate text
ann = client.annotate("Kolkata is the capital of West Bengal, India.")
for sentence in ann['sentences']:
    for token in sentence['tokens']:
        print(token['word'], token['ner'])

"""# Chapter 21: Hugging Face Transformers"""

from transformers import pipeline

# Load pre-trained model for NER
nlp = pipeline('ner', model='dbmdz/bert-large-cased-finetuned-conll03-english')

# Process text
result = nlp("Durgapur is an industrial city in the state of West Bengal, India.")
print(result)

"""# Chapter 21: TextBlob"""

from textblob import TextBlob

# Example text
text = "Netaji Subhash Chandra Bose was born in Cuttack."

# Create TextBlob object
blob = TextBlob(text)

# Extract noun phrases (not exact NER, but can be useful)
print(blob.noun_phrases)

"""# Chapter 21: Scikit-learn"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn_crfsuite import CRF

# Example data
X_train = [['Barack', 'Obama', 'was', 'born', 'in', 'Honolulu']]
y_train = [['B-PER', 'I-PER', 'O', 'O', 'O', 'LOC']]

# Feature extraction
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform([' '.join(sentence) for sentence in X_train])

# Train CRF model
crf = CRF()
crf.fit(X_train, y_train)

"""# Chapter 21: Flair  """

Code

from flair.data import Sentence
from flair.models import SequenceTagger

# Load pre-trained NER model
tagger = SequenceTagger.load('ner')

# Process text
sentence = Sentence("Tagore Obama was born in India")
tagger.predict(sentence)

# Print named entities
for entity in sentence.get_spans('ner'):
    print(entity.text, entity.get_label('ner').value)

"""# Chapter 21: Visualizing NER in Text"""

# Function to add background color to named entities

def highlight_entities_with_bg(text):

    doc = nlp(text)

    # Define background colors for different entity types

    colors = {

        "PERSON": "\033[41m",  # Red background

        "ORG": "\033[42m",     # Green background

        "GPE": "\033[46m",     # Cyan background

        "DATE": "\033[43m",    # Yellow background

        "TIME": "\033[44m",    # Blue background

        "MONEY": "\033[45m",   # Magenta background

        "LOC": "\033[47m",     # White background

        "RESET": "\033[0m"     # Reset to default

        # Add more entity types and colors as needed

    }

    highlighted_text = ""



    for token in doc:

        if token.ent_type_ in colors:

            highlighted_text += colors[token.ent_type_] + token.text + colors["RESET"] + " "

        else:

            highlighted_text += token.text + " "



    return highlighted_text



# Sample text

text = "India, officially the Republic of India (ISO: Bhārat Gaṇarājya),[21] is a country in South Asia. It is the seventh-largest country by area; the most populous country with effect from June 2023;[22][23] and from the time of its independence in 1947, the world's most populous democracy.[24][25][26] Bounded by the Indian Ocean on the south, the Arabian Sea on the southwest, and the Bay of Bengal on the southeast, it shares land borders with Pakistan to the west;[j] China, Nepal, and Bhutan to the north; and Bangladesh and Myanmar to the east. In the Indian Ocean, India is in the vicinity of Sri Lanka and the Maldives; its Andaman and Nicobar Islands share a maritime border with Thailand, Myanmar, and Indonesia. Narendra Modi, a former chief minister of Gujarat, is serving as the 14th Prime Minister of India in his third term since May 26, 2014."



# Highlight the entities in the text with background colors

highlighted_text = highlight_entities_with_bg(text)

print(highlighted_text)

"""# **Chapter 22 Text Summarization **

# Chapter 22: NLTK (Natural Language Toolkit)
"""

from nltk.tokenize import sent_tokenize



text = "Your text here"

sentences = sent_tokenize(text)

print(sentences)

"""# Chapter 22: Gensim"""

from gensim.summarization import summarize



text = "Your text here"

summary = summarize(text)

print(summary)