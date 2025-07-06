# **Chapter 19: Feature Extraction and Representation **

# Chapter 19: Implement the BoW model using Pthon and the popular Scikit-learn library.

from sklearn.feature_extraction.text import CountVectorizer



# Sample documents

documents = [

    "I love machine learning",

    "Machine learning is great",

    "I love coding"

]



# Initialize the CountVectorizer

# CountVectorizer is a tool in NLP that converts a collection of text documents into a matrix of token counts.

vectorizer = CountVectorizer()



# Fit the vectorizer to the documents and transform them into vectors

# This process involves two main steps:

# 1. It 'learns' the vocabulary from the documents.

# 2. It transforms the documents into vectors (where each document is represented by a vector of word counts).

X = vectorizer.fit_transform(documents)



# Convert the sparse matrix to a dense matrix and print the result

# The result is stored as a sparse matrix, but we convert it to a dense matrix (array) for easy reading.

bow_matrix = X.toarray()



# Display the vocabulary (the list of words found in the documents)

# The 'vocabulary_' attribute contains a dictionary where the keys are the words in the documents,

# and the values are the indices that represent the position of those words in the feature vector.

print("Vocabulary:", vectorizer.vocabulary_)



# Display the Bag-of-Words Matrix (the vector representation of each document)

# This matrix shows how many times each word in the vocabulary appears in each document.

# Each row corresponds to a document, and each column corresponds to a word in the vocabulary.

print("Bag-of-Words Matrix:\n", bow_matrix)

# Chapter 19: implement the TF-IDF model using Python and the Scikit-learn library.

from sklearn.feature_extraction.text import TfidfVectorizer



# Sample documents

my_documents = [

    "I love machine learning",

    "Machine learning is great",

    "I love coding"

]



# Initialize the TfidfVectorizer

vectorizer = TfidfVectorizer()



# Fit the vectorizer to the documents and transform them into TF-IDF vectors

X = vectorizer.fit_transform(my_documents)

my_documents

# Convert the sparse matrix to a dense matrix and print the result

tfidf_matrix = X.toarray()



print("Here is  Vocabulary:", vectorizer.vocabulary_)

print("Here is the TF-IDF Matrix:\n", tfidf_matrix)

from sklearn.feature_extraction.text import CountVectorizer



# Initialize the CountVectorizer to build the bag-of-words representation

word_vectorizer = CountVectorizer()

bow_matrix = word_vectorizer.fit_transform(cleaned_texts)



# Display the words (features) and the corresponding Bag-of-Words matrix

print("Vocabulary:", word_vectorizer.get_feature_names_out())

print("Bag-of-Words Matrix:\n", bow_matrix.toarray())

# Chapter 19: Implementing TF-IDF with Scikit-Learn

from sklearn.feature_extraction.text import TfidfVectorizer



# Initialize the TfidfVectorizer to build the TF-IDF representation

tfidf_extractor = TfidfVectorizer()

tfidf_matrix = tfidf_extractor.fit_transform(cleaned_texts)



# Display the vocabulary and the resulting TF-IDF matrix

print("Vocabulary:", tfidf_extractor.get_feature_names_out())

print("TF-IDF Matrix:\n", tfidf_matrix.toarray())

from sklearn.feature_extraction.text import TfidfVectorizer



# Initialize the TfidfVectorizer to build the TF-IDF representation

tfidf_extractor = TfidfVectorizer()

tfidf_matrix = tfidf_extractor.fit_transform(cleaned_texts)



# Display the vocabulary and the resulting TF-IDF matrix

print("Vocabulary:", tfidf_extractor.get_feature_names_out())

print("TF-IDF Matrix:\n", tfidf_matrix.toarray())

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

import matplotlib.pyplot as plt

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import NMF



# Sample documents related to Bankura and Kolkata

docuMents = [

    "Bankura is a district in West Bengal known for its rich cultural heritage and historical temples.",

    "Kolkata, the capital of West Bengal, is a hub for political activities and business in Eastern India.",

    "Bankura is facing challenges in infrastructure development, with poor road connectivity in rural areas.",

    "Kolkata's economy is growing with new industries, including IT and real estate, expanding rapidly.",

    "The cultural festivals in Bankura attract thousands of visitors every year, with art exhibitions and performances.",

    "The Kolkata Metro, one of the oldest in India, is undergoing significant upgrades to expand its reach.",

    "Many rural areas in Bankura still struggle with access to basic services such as healthcare and education.",

    "Kolkata is home to some of India's most iconic landmarks, such as the Howrah Bridge and Victoria Memorial.",

    "Bankura’s agricultural economy is important, with rice, vegetables, and fruits being major crops.",

    "Kolkata’s political landscape is shaped by historical movements, and it is a key center for political activism in India."

]



# Vectorize the documents using TF-IDF

talkBooster = TfidfVectorizer(stop_words='english')

tfidfMatrix = talkBooster.fit_transform(docuMents)



# Apply NMF for topic extraction

thingyMachine = NMF(n_components=2, random_state=1)

W_matrix = thingyMachine.fit_transform(tfidfMatrix)

H_matrix = thingyMachine.components_



# Visualize the topics with a bar chart

featureNames = talkBooster.get_feature_names_out()



for topic_idx, topic in enumerate(H_matrix):

    print(f"Topic {topic_idx}:")

    top_words = [featureNames[i] for i in topic.argsort()[:-6:-1]]

    print(" ".join(top_words))



    # Plotting the top words for the topic with thinner bars and dual-tone grayscale

    top_word_indices = topic.argsort()[:-6:-1]

    top_word_scores = topic[top_word_indices]



    # Split the top words into two categories: higher and lower importance

    mid_point = len(top_word_scores) // 2

    higher_scores = top_word_scores[:mid_point]

    lower_scores = top_word_scores[mid_point:]



    # Assign different colors for the higher and lower importance words

    higher_color = plt.cm.Greys(0.6)  # Darker shade

    lower_color = plt.cm.Greys(0.3)   # Lighter shade



    # Create an alternating color pattern

    bar_colors = [higher_color] * len(higher_scores) + [lower_color] * len(lower_scores)



    plt.figure(figsize=(8, 4))

    plt.barh(top_words[:mid_point], higher_scores, color=higher_color, height=0.4)

    plt.barh(top_words[mid_point:], lower_scores, color=lower_color, height=0.4)



    plt.xlabel('Importance')

    plt.title(f'Topic {topic_idx} Word Importance')

    plt.gca().invert_yaxis()

    plt.show()