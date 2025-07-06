import spacy

# Load the SpaCy model

nlp = spacy.load('en_core_web_sm')



# Sample text

text = "My name is Dibyendu Banerjee. I am exploring Python."



# Process the text

doc = nlp(text)



# Extract tokens

tokens = [token.text for token in doc]

print("Tokens:", tokens)

import numpy as np

import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

import gensim.downloader as api



# Load the pre-trained word embeddings model

model = api.load("glove-wiki-gigaword-50")  # 50-dimensional GloVe model



# Define categories and words for visualization

categories = {

    'Animals': ['cat', 'dog', 'wolf', 'lion', 'tiger', 'bear', 'elephant', 'giraffe', 'zebra', 'horse', 'rabbit', 'fox', 'deer'],

    'Colors': ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown', 'black', 'white', 'gray', 'cyan', 'magenta'],

    'Countries': ['USA', 'Canada', 'Germany', 'France', 'Japan', 'China', 'Brazil', 'India', 'Australia', 'Russia', 'Mexico', 'Italy', 'South Africa'],

    'Foods': ['pizza', 'burger', 'sushi', 'pasta', 'salad', 'apple', 'banana', 'cherry', 'grape', 'steak', 'bread', 'cheese', 'chocolate'],

    'Emotions': ['happy', 'sad', 'angry', 'excited', 'bored', 'fearful', 'surprised', 'nervous', 'relieved', 'disgusted', 'joyful', 'anxious', 'content']

}



# Flatten the list of words and create a category list

words = [word for category in categories.values() for word in category]

categories_flat = [cat for cat, words_list in categories.items() for _ in words_list]



# Filter words to include only those present in the model

words = [word for word in words if word in model]

categories_flat = [cat for word, cat in zip(words, categories_flat) if word in model]



# Get word vectors

word_vectors = np.array([model[word] for word in words])



# Check if there are enough words for t-SNE

if len(word_vectors) < 2:

    raise ValueError("Not enough words to perform t-SNE. Please use more words.")



# Apply t-SNE to reduce dimensionality to 2D

perplexity = min(30, len(word_vectors) - 1)  # Adjusted for a larger number of samples

tsne = TSNE(n_components=2, perplexity=perplexity, random_state=0)

word_vectors_2d = tsne.fit_transform(word_vectors)



# Create a 2D scatter plot

plt.figure(figsize=(18, 14))

ax = plt.gca()



# Define colors for each category

colors = {

    'Animals': 'red',

    'Colors': 'blue',

    'Countries': 'green',

    'Foods': 'orange',

    'Emotions': 'purple'

}

color_map = [colors[cat] for cat in categories_flat]



# Plot each category with different colors and larger markers

scatter = plt.scatter(word_vectors_2d[:, 0], word_vectors_2d[:, 1], c=color_map, marker='o', s=100, alpha=0.8, edgecolors='w', linewidth=0.5)



# Add legend with a background color

handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[cat], markersize=15, label=cat) for cat in colors]

plt.legend(handles=handles, title='Categories', loc='upper right', frameon=True, facecolor='lightgrey')



# Annotate each point with the corresponding word using a bold font

for i, word in enumerate(words):

    plt.annotate(word, (word_vectors_2d[i, 0], word_vectors_2d[i, 1]), fontsize=12, weight='bold', alpha=0.9)



plt.title('2D Visualization of Word Embeddings using t-SNE', fontsize=16, weight='bold')

plt.xlabel('Dimension 1', fontsize=14, weight='bold')

plt.ylabel('Dimension 2', fontsize=14, weight='bold')

plt.grid(True, linestyle='--', alpha=0.6)

plt.show()

import nltk

from wordcloud import WordCloud

import matplotlib.pyplot as plt



# Download and load NLTK text sample

nltk.download('gutenberg')

from nltk.corpus import gutenberg

text = gutenberg.raw('austen-emma.txt')



# Generate and display word cloud

def generate_word_cloud(text):

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)



    plt.figure(figsize=(12, 6))

    plt.imshow(wordcloud, interpolation='bilinear')

    plt.axis('off')  # Hide axes

    plt.title('Word Cloud from NLTK Text')

    plt.show()



generate_word_cloud(text)

# **Chapter 21: Named Entity Recognition(NER) **

# Regular Expressions

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

# Chapter 21: Deep Learning-Based Method of NER

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

# Chapter 21:Stanford NER

from stanfordnlp.server import CoreNLPClient

client = CoreNLPClient(annotators=['ner'], timeout=30000, memory='16G')

# Annotate text
ann = client.annotate("Kolkata is the capital of West Bengal, India.")
for sentence in ann['sentences']:
    for token in sentence['tokens']:
        print(token['word'], token['ner'])

from transformers import pipeline

# Load pre-trained model for NER
nlp = pipeline('ner', model='dbmdz/bert-large-cased-finetuned-conll03-english')

# Process text
result = nlp("Durgapur is an industrial city in the state of West Bengal, India.")
print(result)

from textblob import TextBlob

# Example text
text = "Netaji Subhash Chandra Bose was born in Cuttack."

# Create TextBlob object
blob = TextBlob(text)

# Extract noun phrases (not exact NER, but can be useful)
print(blob.noun_phrases)

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

# Chapter 21: Visualizing NER in Text

Using GPT-3 (Generative Pre-trained Transformer 3)

Generative AI refers to algorithms that can generate new content, including text, images, and sounds,

by learning patterns from existing data. These models have gained attention due to their ability to

create human-like content and their applications in various fields such as entertainment, education, and

marketing. As technology evolves, generative AI systems are becoming increasingly sophisticated,

capable of producing highly realistic outputs that can challenge our perceptions of creativity and authorship.

# Summarize using the first model

summary_1 = summarizer_1(text, max_length=50, min_length=25, do_sample=False)[0]['summary_text']

print("Summary from Model 1 (BART):")

print(summary_1)



# Summarize using the second model

summary_2 = summarizer_2(text, max_length=50, min_length=25, do_sample=False)[0]['summary_text']

print("\nSummary from Model 2 (PEGASUS):")

print(summary_2)



# Reference summary (ground truth)

reference_summary = "Generative AI refers to algorithms that can generate new content. These models have gained attention due to their ability to create human-like content."



# Function to calculate ROUGE scores

def calculate_rouge(reference, generated):

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    scores = scorer.score(reference, generated)

    return scores



# Calculate ROUGE scores

rouge_scores_1 = calculate_rouge(reference_summary, summary_1)

rouge_scores_2 = calculate_rouge(reference_summary, summary_2)



print("\nROUGE Scores for Model 1 (BART):")

print(rouge_scores_1)



print("\nROUGE Scores for Model 2 (PEGASUS):")

from nltk.translate.bleu_score import sentence_bleu



# Reference summary (ground truth)

reference = [

    "Generative AI refers to algorithms that can generate new content, including text, images, and sounds."

]



# Model outputs

summary_bart = "Generative AI refers to algorithms that can generate new content, including text, images, and sounds."

summary_pegasus = "Researchers at the University of California, Los Angeles (UCLA), have developed a new type of artificial intelligence (AI) called generative AI."



# Calculate BLEU scores

bleu_bart = sentence_bleu([reference[0].split()], summary_bart.split())

bleu_pegasus = sentence_bleu([reference[0].split()], summary_pegasus.split())



# Print BLEU scores

print(f"BLEU Score for BART: {bleu_bart:.4f}")

print(f"BLEU Score for PEGASUS: {bleu_pegasus:.4f}")

# Chapter 24: GPT (Generative Pre-trained Transformer)

# Chapter 24: Example Code - Text Generation with GPT-2

# Chapter 24: Practical Example : Advanced Text Generation with GPT-3

import openai



# Set up API key

openai.api_key = 'your-api-key'



# Generate text

response = openai.Completion.create(

    engine="text-davinci-003",

    prompt="In a distant future, humans have discovered a new form of energy that",

    max_tokens=200,

    temperature=0.6,

    top_p=0.9,

    frequency_penalty=0,

    presence_penalty=0

)



generated_text = response.choices[0].text.strip()

print(f"Generated Text: {generated_text}")

import openai



# Set up API key

openai.api_key = 'your-api-key'



# Define a conversation prompt

conversation_prompt_example = "User: What is the capital of India?\nAI:"



# Generate a response

response = openai.Completion.create(

    engine="text-davinci-003",

    prompt=conversation_prompt_example,

    max_tokens=50

)



ai_response = response.choices[0].text.strip()

print(f"AI Response: {ai_response}")

# Chapter 24: Practical Example : Poetry Generation with GPT-3

import openai



# Set up API key

openai.api_key = 'your-api-key'



# Generate poetry

response = openai.Completion.create(

    engine="text-davinci-003",

    prompt="Compose a poem about the Quiet Sunset Moments",

    max_tokens=100

)



poem = response.choices[0].text.strip()

print(f"Generated Poem: {poem}")

from transformers import Trainer, TrainingArguments



# Initialize the Trainer with distributed training arguments

training_args = TrainingArguments(

    output_dir='./results',

    per_device_train_batch_size=16,

    per_device_eval_batch_size=16,

    num_train_epochs=3,

    fp16=True,  # Enable mixed precision training for large datasets

    gradient_accumulation_steps=8,

    deepspeed='./ds_config.json'  # Enables distributed training

)



trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=eval_dataset

)



# Start training in a distributed fashion

trainer.train()