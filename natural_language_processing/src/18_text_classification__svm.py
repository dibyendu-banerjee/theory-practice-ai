"""
Text Classification - SVM
This script demonstrates: Text Classification - SVM
"""

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