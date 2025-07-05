"""# Use Case: PCA in Personalized music recommendation system"""

# Importing necessary libraries

import pandas as pd

import numpy as np

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.metrics.pairwise import cosine_similarity



# Sample dataset with real song names and some made-up features

data = {

    'Song': ['Blinding Lights', 'Shape of You', 'Levitating', 'Uptown Funk', 'Watermelon Sugar', 'Stay', 'Good 4 U'],

    'Artist': ['The Weeknd', 'Ed Sheeran', 'Dua Lipa', 'Mark Ronson ft. Bruno Mars', 'Harry Styles', 'The Kid LAROI & Justin Bieber', 'Olivia Rodrigo'],

    'Tempo': [85, 96, 103, 115, 95, 105, 138],

    'Energy': [0.8, 0.85, 0.9, 0.88, 0.92, 0.85, 0.88],

    'Danceability': [0.75, 0.85, 0.9, 0.87, 0.91, 0.88, 0.92],

    'Mood': [0.8, 0.7, 0.85, 0.85, 0.9, 0.75, 0.88]

}



# Convert the data to a DataFrame

df = pd.DataFrame(data)



# Display the dataset

print("Original Music Dataset:")

print(df)



# Feature Selection (excluding 'Song' and 'Artist' columns for PCA)

X = df[['Tempo', 'Energy', 'Danceability', 'Mood']]



# Standardizing the features

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)



# Applying PCA

pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization

X_pca = pca.fit_transform(X_scaled)



# Add the PCA results to the original dataset for visualization

df['PCA_1'] = X_pca[:, 0]

df['PCA_2'] = X_pca[:, 1]



# Visualize the PCA result

plt.figure(figsize=(8, 6))

plt.scatter(df['PCA_1'], df['PCA_2'], c='blue', marker='o')

for i, txt in enumerate(df['Song']):

    plt.annotate(f"{df['Song'][i]} - {df['Artist'][i]}", (df['PCA_1'][i], df['PCA_2'][i]), fontsize=10)

plt.title('PCA of Music Features')

plt.xlabel('Principal Component 1')

plt.ylabel('Principal Component 2')

plt.grid(True)

plt.show()



# Recommending songs based on PCA

def recommend_song(song_name, n_recommendations=2):

    # Get the index of the song selected for recommendation

    song_idx = df[df['Song'] == song_name].index[0]



    # Compute cosine similarity between the selected song and all other songs

    song_pca_values = df[['PCA_1', 'PCA_2']].iloc[song_idx].values.reshape(1, -1)

    similarities = cosine_similarity(song_pca_values, df[['PCA_1', 'PCA_2']])



    # Get the indices of the most similar songs

    similar_songs_idx = np.argsort(similarities[0])[::-1][1:n_recommendations+1]

    recommended_songs = df.iloc[similar_songs_idx]['Song'].values

    return recommended_songs



# Example: Recommending songs similar to 'Blinding Lights'

recommended = recommend_song('Blinding Lights')

print("\nRecommended Songs for 'Blinding Lights':")

print(recommended)