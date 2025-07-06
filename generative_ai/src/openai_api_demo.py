# ================================================================
# File: openai_api_demo.py
# Description: This script demonstrates how to use the OpenAI API 
# (GPT-4) to generate text completions. It sends a prompt and 
# receives a concise explanation from the model.
#
# Author: Dibyendu Banerjee
# ================================================================

import openai

# Set your OpenAI API key
openai.api_key = 'your-api-key'  # Replace with your actual key

# Define the prompt
prompt = "Explain the concept of Large Language Models (LLMs)."

# Call the OpenAI API
response = openai.Completion.create(
    model="gpt-4",
    prompt=prompt,
    max_tokens=100,
    temperature=0.7
)

# Print the generated response
print("Response from GPT-4:")
print(response.choices[0].text.strip())
