"""# Use Case: Predicting Instagram User Engagement Based on Hashtags"""

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor, plot_tree

from sklearn.metrics import mean_squared_error



# Step 1: Sample Data

data = {

    'hashtags': ['#food', '#travel', '#fashion', '#nature', '#fitness', '#food', '#travel', '#fashion', '#nature', '#fitness'],

    'likes': [250, 400, 300, 500, 350, 270, 410, 310, 520, 360],

    'comments': [30, 50, 40, 80, 60, 35, 55, 45, 85, 65],

    'shares': [10, 15, 12, 18, 14, 11, 16, 13, 20, 15],

    'engagement': [300, 465, 352, 598, 424, 316, 481, 368, 625, 440]

}



df = pd.DataFrame(data)



# Step 2: Encode hashtags

hashtag_encoder = LabelEncoder()

df['hashtag_encoded'] = hashtag_encoder.fit_transform(df['hashtags'])



# Step 3: Prepare data for training

X = df[['hashtag_encoded', 'likes', 'comments', 'shares']]

y = df['engagement']



# Step 4: Split the data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Step 5: Train Decision Tree Regressor Model

model = DecisionTreeRegressor(random_state=42)

model.fit(X_train, y_train)



# Step 6: Make Predictions

y_pred = model.predict(X_test)



# Step 7: Visualize the Decision Tree

plt.figure(figsize=(12, 8))

plot_tree(model, feature_names=X.columns, filled=True, rounded=True)

plt.title('Decision Tree for Predicting Instagram Engagement')

plt.show()



# Step 8: Evaluate the Model

mse = mean_squared_error(y_test, y_pred)

print(f"Mean Squared Error: {mse}")



# Step 9: Feature Importance

feature_importance = model.feature_importances_

sns.barplot(x=X.columns, y=feature_importance)

plt.title("Feature Importance")

plt.ylabel('Importance')

plt.show()



# Step 10: Predictions on the Test Set

test_predictions = pd.DataFrame({

    'Hashtag': hashtag_encoder.inverse_transform(X_test['hashtag_encoded']),

    'Predicted Engagement': y_pred

})

print(test_predictions)