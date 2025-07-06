"""
Topic Modeling - LDA
This script demonstrates: Topic Modeling - LDA
"""

# Sample data

data = {

    'text': [

        'I love programming',

        'Python is great for data science',

        'I hate bugs and errors',

        'Debugging is a fun process',

        'I enjoy learning new languages',

        'This code is terrible',

        'Machine learning is fascinating',

        'I dislike poorly written code',

        'Collaboration is essential in programming',

        'I feel great when my code works'

    ],

    'label': [

        'positive',

        'positive',

        'negative',

        'positive',

        'positive',

        'negative',

        'positive',

        'negative',

        'positive',

        'positive'

    ]

}



# Create DataFrame

df = pd.DataFrame(data)



# Split the dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.3, random_state=42)



# Convert text to feature vectors

vectorizer = CountVectorizer(stop_words='english')  # Remove English stop words

X_train_counts = vectorizer.fit_transform(X_train)

X_test_counts = vectorizer.transform(X_test)



# Initialize the Naive Bayes classifier

classifier = MultinomialNB()



# Train the classifier

classifier.fit(X_train_counts, y_train)



# Make predictions

y_pred = classifier.predict(X_test_counts)



# Evaluate the classifier

accuracy = accuracy_score(y_test, y_pred)

report = classification_report(y_test, y_pred)

conf_matrix = confusion_matrix(y_test, y_pred)



# Print predicted vs actual labels

results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

print("Predicted vs Actual Labels:\n", results)



print(f'\nAccuracy: {accuracy:.2f}')

print('\nClassification Report:\n', report)

"""# Chapter 20: Using Support Vector Machines (SVM) for Text Classification"""

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt

import seaborn as sns



# Sample dataset with detailed email content

data = {

    'email_content': [

        "Subject: Congratulations Sir! You've won a $1000 gift card!\n\nDear winner, ee are thrilled to announce that you have been randomly selected to receive a $45000 gift card! Click the link below to claim your prize now. Don’t miss this exciting opportunity!",

        "Subject: Important update regarding your account\n\nDear user, we wanted to inform you about some important changes to your account. Please log in to your account to review the updates. Thank you for your attention.",

        "Subject: Click here to claim your free prize!\n\nHi there! You have an exclusive chance to claim a free gift. Act fast and click the link to see your surprise! This offer is only available for a limited time.",

        "Subject: Your latest invoice is attached\n\nHello, please find your invoice attached to this email. If you have any questions regarding the details, feel free to reach out to us. Thank you!",

        "Subject: You've been selected for a special promotion!\n\nDear valued customer, you have been selected for an amazing promotional sale! Get 85% off your next purchase. Click here to find out more!",

        "Subject: Meeting agenda for tomorrow\n\nHi team, please find attached the agenda for our meeting scheduled for tomorrow. Make sure to review it and come prepared. Looking forward to seeing everyone!"

    ],

    'label': ['spam', 'not spam', 'spam', 'not spam', 'spam', 'not spam']

}



# Create a DataFrame

df = pd.DataFrame(data)



# Split the dataset into features and labels

X = df['email_content']

y = df['label']



# Split into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



# Convert text to feature vectors using TF-IDF

vectorizer = TfidfVectorizer(stop_words='english')

X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)



# Initialize the SVM classifier

classifier = SVC(kernel='linear')



# Train the classifier

classifier.fit(X_train_tfidf, y_train)



# Make predictions

y_pred = classifier.predict(X_test_tfidf)



# Evaluate the classifier

accuracy = accuracy_score(y_test, y_pred)

report = classification_report(y_test, y_pred)

conf_matrix = confusion_matrix(y_test, y_pred)



# Print accuracy

print(f'Accuracy: {accuracy:.2f}\n')



# Print detailed predictions

print("Predictions on Test Emails:\n")

for email, prediction in zip(X_test, y_pred):

    print(f"Email Content:\n{email}\nPredicted Label: {prediction}\n")



# Print classification report

print('Classification Report:\n', report)



# Visualizing the confusion matrix

plt.figure(figsize=(6, 5))

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',

            xticklabels=['Not Spam', 'Spam'],

            yticklabels=['Not Spam', 'Spam'])

plt.ylabel('Actual')

plt.xlabel('Predicted')

plt.title('Confusion Matrix')

plt.show()

"""# Chapter 20: Clustering"""

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.cluster import KMeans



# Sample data: A set of articles on different topics

articles = [

    "The government is planning new policies for healthcare reform.",

    "Basketball is a popular sport, and the NBA finals are coming soon.",

    "The economy is growing rapidly in the US due to new policies.",

    "Artificial Intelligence is changing the tech landscape.",

    "Football is a sport loved by millions around the world.",

    "Robotics and AI are closely related fields in technology.",

    "The latest elections were held in the US, with major debates.",

    "Technology companies are investing heavily in AI and robotics."

]



# Step 1: Vectorization - Convert text data into numerical format using TF-IDF

# This will transform the text into a matrix of numerical features based on the frequency of words in each article

vectorizer = TfidfVectorizer(stop_words='english')  # Remove common English stop words

X = vectorizer.fit_transform(articles)



# Step 2: Clustering - Apply KMeans clustering to group similar articles together

# We will specify 3 clusters, as we expect 3 topics: Politics, Sports, and Technology

kmeans = KMeans(n_clusters=3, random_state=42)

kmeans.fit(X)



# Step 3: Extracting the cluster labels

# These are the cluster labels assigned to each article

labels = kmeans.labels_



# Step 4: Displaying the articles and their assigned cluster labels

print("Clustering Results:")

for i, label in enumerate(labels):

    print(f"Article: '{articles[i]}' -> Cluster: {label}")



# Step 5: Display the top terms (words) contributing to each cluster

terms = vectorizer.get_feature_names_out()



print("\nTop terms for each cluster:")

for i in range(3):  # We have 3 clusters

    print(f"\nCluster {i}:")

    cluster_center = kmeans.cluster_centers_[i]  # The center of the cluster (centroid)

    sorted_term_indices = cluster_center.argsort()[-5:][::-1]  # Get the top 5 terms

    top_terms = [terms[index] for index in sorted_term_indices]

    print("Top terms:", top_terms)

"""# Chapter 20: Topic Modeling"""

import gensim

from gensim import corpora



# Sample data

texts = [['I', 'love', 'baseball'], ['Hockey', 'is', 'my', 'favorite', 'sport']]



# Create a dictionary and corpus

dictionary = corpora.Dictionary(texts)

corpus = [dictionary.doc2bow(text) for text in texts]



# Train LDA model

lda_model = gensim.models.LdaModel(corpus, num_topics=2, id2word=dictionary, passes=15)



# Print topics

for idx, topic in lda_model.print_topics(-1):

    print(f'Topic: {idx} \nWords: {topic}')

"""# **Chapter 21: Named Entity Recognition(NER) **

# Regular Expressions
"""

import re



text = "My phone number is 704-408-5490 and my email is banerjee.dibyendu@gmail.com"



# Regular expression patterns for phone numbers and emails

phone_pattern = r'\d{3}-\d{3}-\d{4}'

email_pattern = r'\S+@\S+'



# Find matches

phone_matches = re.findall(phone_pattern, text)

email_matches = re.findall(email_pattern, text)



print("Phone numbers:", phone_matches)

print("Emails:", email_matches)

"""# Chapter 21: Dictionary Lookups  """

from collections import defaultdict



# Example dictionary of named entities

entity_dict = {

    'locations': {'Kolkata', 'Delhi', 'Chennai'},

    'persons': {'Sachin', 'Saurav', 'Virat'}

}



text = "Virat and Sachin are visiting Kolkata and Chennai."



def lookup_entities(text, entity_dict):

    found_entities = defaultdict(list)

    for entity_type, entities in entity_dict.items():

        for entity in entities:

            if entity in text:

                found_entities[entity_type].append(entity)

    return found_entities



entities_found = lookup_entities(text, entity_dict)

print("Entities found:", dict(entities_found))

"""# Chapter 21: Pattern-Based Rules"""

import re



text = "Rabindranath Tagore was born in India."



# Define a pattern for capitalized words (potential named entities)

pattern = r'\b[A-Z][a-z]*\b'



# Find all capitalized words

capitalized_words = re.findall(pattern, text)



print("Capitalized words:", capitalized_words)

"""# Chapter 21: Support Vector Machines (SVM)"""

from sklearn.svm import SVC

from sklearn.feature_extraction.text import CountVectorizer



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