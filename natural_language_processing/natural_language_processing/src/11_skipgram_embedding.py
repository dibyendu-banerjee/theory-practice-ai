"""
Skip-Gram Embedding
This script demonstrates: Skip-Gram Embedding
"""

# Sample corpus related to Kolkata

corpus = [

    "Kolkata is known for its rich history",

    "Howrah Bridge is a famous landmark in Kolkata",

    "Street food in Kolkata is delicious",

    "Victoria Memorial is a historical landmark in Kolkata",

    "Kolkata has a vibrant cultural scene",

    "Durga Puja is the biggest festival in Kolkata",

    "The Sundarbans is a beautiful mangrove forest near Kolkata",

    "Park Street is a popular road in Kolkata"

]



# Tokenizing the corpus

tokenizer = Tokenizer()

tokenizer.fit_on_texts(corpus)

word_index = tokenizer.word_index

index_word = {i: word for word, i in word_index.items()}



print("Word Index