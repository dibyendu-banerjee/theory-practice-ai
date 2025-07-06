# Chapter 19: Implementing the Skip-Gram Model in Python

# Chapter 19: Skip-Gram Model Definition

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Embedding, Dense, Reshape

from tensorflow.keras.optimizers import Adam



# Hyperparameters

embedding_dim = 10  # Dimensionality of the word embeddings



# Define the Skip-Gram model

model = Sequential()



# Embedding layer for the target words

model.add(Embedding(input_dim=len(word_index) + 1,

                    output_dim=embedding_dim,

                    input_length=1,

                    name='embedding_layer'))



# Reshape to match the Dense layer's output

model.add(Reshape((embedding_dim,)))



# Dense layer to predict the context words (softmax for classification)

model.add(Dense(len(word_index) + 1, activation='softmax'))



# Compile the model

model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])



# Summarize the model architecture

model.summary()

# Chapter 19: Implementing CBOW using Python

# Chapter 19: Implementing Word2Vec with Gensim