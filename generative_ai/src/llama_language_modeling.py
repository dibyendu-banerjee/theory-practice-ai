# ================================================================
# File: llama_language_modeling.py
# Description: This script demonstrates basic language modeling 
# using Facebook's LLaMA model via HuggingFace Transformers. It 
# generates a continuation for a given prompt.
#
# Author: Dibyendu Banerjee
# ================================================================

from transformers import AutoTokenizer, AutoModelForCausalLM

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('facebook/llama-7b')
model = AutoModelForCausalLM.from_pretrained('facebook/llama-7b')

# Define input prompt
prompt = "Explain LLMs."
inputs = tokenizer(prompt, return_tensors="pt")

# Generate output
outputs = model.generate(**inputs, max_length=100)

# Decode and print the result
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated Text:\n", generated_text)
