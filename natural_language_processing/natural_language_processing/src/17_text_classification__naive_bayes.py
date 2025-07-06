"""
Text Classification - Naive Bayes
This script demonstrates: Text Classification - Naive Bayes
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

print("Predicted vs Actual Labels