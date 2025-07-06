"""
CBOW Embedding
This script demonstrates: CBOW Embedding
"""

# Sample sentences about Kolkata City

kolkata_text = [

    "Kolkata is the capital of West Bengal",

    "The Howrah Bridge is one of Kolkata's most iconic landmarks",

    "Kolkata is known for its colonial architecture and cultural heritage",

    "Durga Puja is the most celebrated festival in Kolkata",

    "The Victoria Memorial is a famous historical monument in Kolkata",

    "Kolkata is a major center for literature, arts, and education in India",

    "The city of Kolkata is also called the City of Joy",

    "Kolkata has a rich history, with important figures like Rabindranath Tagore",

    "The Indian Museum in Kolkata is one of the oldest and largest museums in India"

]



# Tokenize each sentence and convert to lowercase

tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in kolkata_text]



print("Tokenized Sentences:", tokenized_sentences)



# Train the CBOW model (sg=0 for CBOW, vector size=100, window size=2)

cbow_model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=2, min_count=1, sg=0)



# Check the model's vocabulary size

print("\nVocabulary Size:", len(cbow_model.wv))



# Get the word vector for 'kolkata'

kolkata_vector = cbow_model.wv['kolkata']

print("\nEmbedding for 'kolkata':", kolkata_vector[:5])  # Displaying a part of the vector



# Retrieve similar words to 'kolkata'

similar_words_list = cbow_model.wv.most_similar('kolkata', topn=5)