# ================================================================
# File: llama_text_generation.py
# Description: This script demonstrates text generation using 
# Meta's LLaMA-2 model via HuggingFace Transformers. It uses 
# beam search to generate coherent continuations of a prompt.
#
# Author: Dibyendu Banerjee
# ================================================================

from transformers import LlamaTokenizer, LlamaForCausalLM

# Load tokenizer and model from HuggingFace
tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b')
model = LlamaForCausalLM.from_pretrained('meta-llama/Llama-2-7b').to('cuda')

# Define input prompt
input_text = "In the realm of generative AI, LLaMA offers"
inputs = tokenizer(input_text, return_tensors='pt').to('cuda')

# Generate text using beam search
outputs = model.generate(
    **inputs,
    max_length=100,
    num_beams=5,
    early_stopping=True
)

# Decode and print the generated output
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Generated Text:\n{generated_text}")
