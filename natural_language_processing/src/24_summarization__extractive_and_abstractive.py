# ================================================================
# Named Entity Recognition and Text Summarization
#
# This script demonstrates:
# 1. Named Entity Recognition (NER) using SpaCy, NLTK, Hugging Face, Flair, TextBlob
# 2. Visualization of Named Entities with background highlighting
# 3. Extractive Text Summarization using NLTK and Gensim
# ================================================================

# -------------------------------
# Section 1: Named Entity Recognition (NER)
# -------------------------------

# Hugging Face Transformers (BERT)
from transformers import BertTokenizer, BertForTokenClassification, pipeline
from flair.data import Sentence
from flair.models import SequenceTagger
from textblob import TextBlob
import spacy
import nltk
from nltk import pos_tag, ne_chunk
from nltk.tokenize import word_tokenize
from nltk.chunk import tree2conlltags

# Download required NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Hugging Face NER
print("\n🔍 Hugging Face Transformers NER:")
hf_model = BertForTokenClassification.from_pretrained('dbmdz/bert-large-cased-finetuned-conll03-english')
hf_tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-large-cased-finetuned-conll03-english')
hf_pipeline = pipeline('ner', model=hf_model, tokenizer=hf_tokenizer)
print(hf_pipeline("Rabindranath Tagore was born in West Bengal."))

# SpaCy NER
print("\n🧠 SpaCy NER:")
spacy_nlp = spacy.load("en_core_web_sm")
doc = spacy_nlp("Tesla announced plans to build a new factory in Berlin in 2023.")
print([(ent.text, ent.label_) for ent in doc.ents])

# NLTK NER
print("\n📚 NLTK NER:")
text_nltk = "Albert Einstein developed the theory of relativity in the early 20th century."
tokens = word_tokenize(text_nltk)
tagged = pos_tag(tokens)
entities = ne_chunk(tagged)
print(entities)

# Flair NER
print("\n✨ Flair NER:")
flair_tagger = SequenceTagger.load('ner')
sentence = Sentence("Tagore Obama was born in India")
flair_tagger.predict(sentence)
for entity in sentence.get_spans('ner'):
    print(entity.text, entity.get_label('ner').value)

# TextBlob Noun Phrase Extraction
print("\n🧾 TextBlob Noun Phrases:")
blob = TextBlob("Netaji Subhash Chandra Bose was born in Cuttack.")
print(blob.noun_phrases)

# -------------------------------
# Section 2: Visualizing Named Entities
# -------------------------------

def highlight_entities_with_bg(text):
    doc = spacy_nlp(text)
    colors = {
        "PERSON": "\033[41m", "ORG": "\033[42m", "GPE": "\033[46m",
        "DATE": "\033[43m", "TIME": "\033[44m", "MONEY": "\033[45m",
        "LOC": "\033[47m", "RESET": "\033[0m"
    }
    highlighted = ""
    for token in doc:
        if token.ent_type_ in colors:
            highlighted += f"{colors[token.ent_type_]}{token.text}{colors['RESET']} "
        else:
            highlighted += token.text + " "
    return highlighted

print("\n🎨 Highlighted Entities in Text:")
sample_text = (
    "India, officially the Republic of India, is a country in South Asia. "
    "Narendra Modi, a former chief minister of Gujarat, is serving as the 14th Prime Minister of India."
)
print(highlight_entities_with_bg(sample_text))

# -------------------------------
# Section 3: Text Summarization
# -------------------------------

print("\n📝 Text Summarization:")

# NLTK Sentence Tokenization
from nltk.tokenize import sent_tokenize
text_summary = (
    "India is a vast country with diverse cultures. "
    "It has a rich history and a growing economy. "
    "Many global companies are investing in India due to its market potential. "
    "The country is also known for its technological advancements and skilled workforce."
)
sentences = sent_tokenize(text_summary)
print("\n📌 Sentences:")
print(sentences)

# Gensim Summarization
try:
    from gensim.summarization import summarize
    summary = summarize(text_summary)
    print("\n📚 Gensim Summary:")
    print(summary)
except Exception as e:
    print("\n⚠️ Gensim could not summarize the text:", e)
