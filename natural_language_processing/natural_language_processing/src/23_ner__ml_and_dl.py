"""
NER - ML & DL
This script demonstrates: NER - ML & DL
"""

# Example data

texts = ["Barack Obama was born in Honolulu.", "New York is a great city."]

labels = ['PER', 'LOC']



# Convert text to feature vectors

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(texts)



# Train SVM classifier

clf = SVC(kernel='linear')

clf.fit(X, labels)



# Predict on new data

test_texts = ["I visited London last year."]

X_test = vectorizer.transform(test_texts)

predictions = clf.predict(X_test)



print("Predicted labels:", predictions)

"""# Chapter 21: Deep Learning-Based Method of NER"""

from transformers import BertTokenizer, BertForTokenClassification

from transformers import pipeline



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
    print(entity.text