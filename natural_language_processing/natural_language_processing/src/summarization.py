# **Chapter 22 Text Summarization **

# Chapter 22: NLTK (Natural Language Toolkit)

from gensim.summarization import summarize



text = "Your text here"

summary = summarize(text)

print(summary)

from transformers import pipeline



summarizer = pipeline("summarization")

text = "Your text here"

summary = summarizer(text, max_length=150, min_length=30, do_sample=False)

# Extractive Summarization with TextRank

from gensim.summarization import summarize



def extractive_summary(text, ratio=0.2):

    return summarize(text, ratio=ratio)



# Example text

article_text =

# Chapter 22: Abstractive Summarization with GPT-3 or BART

# Chapter 22: Extractive Summarization Using Graph-Based Techniques

# Chapter 22: Abstractive Summarization with Large Language Models (GPT-3, T5)

pip install transformers rouge-score

from transformers import pipeline

from rouge_score import rouge_scorer



# Initialize summarizers

summarizer_1 = pipeline("summarization", model="facebook/bart-large-cnn")

summarizer_2 = pipeline("summarization", model="google/pegasus-xsum")



# Sample text for summarization

text =