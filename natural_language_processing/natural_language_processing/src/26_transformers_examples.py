"""
Transformers Examples
This script demonstrates: Transformers Examples
"""

# Load pre-trained Transformer models (BERT for sentiment & GPT-2 for text generation)

bert_alias = 'bert-base-uncased'

gpt2_alias = 'gpt2'



# Initializing tokenizers and models

bert_parser = BertTokenizer.from_pretrained(bert_alias)

bert_analyzer = BertForSequenceClassification.from_pretrained(bert_alias)

gpt2_parser = GPT2Tokenizer.from_pretrained(gpt2_alias)

gpt2_writer = GPT2LMHeadModel.from_pretrained(gpt2_alias)



# BERT-based Sentiment Analysis Pipeline

analyzer_pipeline = pipeline('sentiment-analysis', model=bert_analyzer, tokenizer=bert_parser)



# Sample inputs for sentiment analysis

statements = [

    "Language models are revolutionizing AI applications.",

    "I dislike the way some AI models work."

]

sentiment_results = analyzer_pipeline(statements)



# GPT-2 Text Generation

starter_text = "Artificial intelligence is transforming the world of"

input_encodings = gpt2_parser(starter_text, return_tensors='pt')

generated_output = gpt2_writer.generate(input_encodings['input_ids'], max_length=50, num_return_sequences=1)

generated_sentence = gpt2_parser.decode(generated_output[0], skip_special_tokens=True)



# Display BERT sentiment analysis results

print("BERT Sentiment Analysis Results:")

for stmt, outcome in zip(statements, sentiment_results):

    print(f"Statement: {stmt}\nSentiment: {outcome['label']} (Confidence: {outcome['score']:.2f})\n")



# Display GPT-2 generated text

print("GPT-2 Generated Text:")

print(f"Input Prompt: {starter_text}\nGenerated Text: {generated_sentence}")

"""# Chapter 24: Practical Example: Predicting Missing Words with BERT Masked Language Model"""

from transformers import BertTokenizer, BertForMaskedLM

import torch



# Load pre-trained BERT model and tokenizer for masked word prediction

bert_variant = 'bert-base-uncased'

token_parser = BertTokenizer.from_pretrained(bert_variant)

mask_filler = BertForMaskedLM.from_pretrained(bert_variant)



# Example sentence with a missing word

sample_text = "The intelligent machine [MASK] learns from experience."

encoded_inputs = token_parser(sample_text, return_tensors='pt')



# Creating label clone and identifying mask position

label_clone = encoded_inputs.input_ids.detach().clone()

masked_position = torch.where(encoded_inputs.input_ids == token_parser.mask_token_id)[1].tolist()



# Predicting the missing word using BERT

with torch.no_grad():

    output_scores = mask_filler(**encoded_inputs, labels=label_clone)

predicted_scores = output_scores.logits

predicted_word_id = torch.argmax(predicted_scores[0, masked_position, :], dim=-1)

predicted_word = token_parser.decode(predicted_word_id)



# Display the predicted word

print(f"Predicted word: {predicted_word}")

"""# Chapter 24: Applications of BERT: Sentiment Analysis, and Question Answering"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Define the model identifier

pretrained_model = "bert-base-uncased"

# Load tokenizer and model using Auto classestext_tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

sentiment_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model)



# Construct an NLP pipeline for sentiment analysis

sentiment_pipeline = pipeline("sentiment-analysis", model=sentiment_model, tokenizer=text_tokenizer)



# Input text for analysis

sample_text = "BERT is a groundbreaking model in NLP."



# Perform sentiment analysis

analysis_result = sentiment_pipeline(sample_text)



# Display the outcome

print(f"Input Text: {sample_text}")

print(f"Predicted Sentiment: {analysis_result[0]['label']} (Confidence Score: {analysis_result[0]['score']:.2f})")

"""# Chapter 24: Question Answering"""

from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline



# Define the model identifier

pretrained_model = "bert-large-uncased-whole-word-masking-finetuned-squad"



# Load tokenizer and model for question answering

question_answer_tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

question_answer_model = AutoModelForQuestionAnswering.from_pretrained(pretrained_model)



# Construct an NLP pipeline for question answering

qa_pipeline = pipeline("question-answering", model=question_answer_model, tokenizer=question_answer_tokenizer)



# Input context and question

context = "BERT is a transformer-based model developed by Google. It is used for various NLP tasks, including question answering."

question = "What is BERT used for?"

# Perform question answering

qa_result = qa_pipeline(question=question, context=context)

# Display the answer

print(f"Question: {question}")

print(f"Answer: {qa_result['answer']}")

"""# Chapter 24: GPT (Generative Pre-trained Transformer)"""

from transformers import GPT2LMHeadModel, GPT2Tokenizer



# Load pre-trained model and tokenizer

model_name = 'gpt2'

tokenizer = GPT2Tokenizer.from_pretrained(model_name)

model = GPT2LMHeadModel.from_pretrained(model_name)



# Generate text

input_text = "Once upon a time"

inputs = tokenizer(input_text, return_tensors='pt')

outputs = model.generate(inputs['input_ids'], max_length=50, num_return_sequences=1)



generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Generated Text