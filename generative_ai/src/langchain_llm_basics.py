# ================================================================
# File: langchain_llm_basics.py
# Description: This script demonstrates how to use a basic LLM 
# with LangChain to generate text. It initializes a model and 
# performs a simple prediction.
#
# Author: Dibyendu Banerjee
# ================================================================

from langchain import LLM

# Initialize a pre-trained language model
model = LLM(model_name="gpt-3.5-turbo")  # Replace with your preferred model

# Define a prompt
prompt = "Write a short story about a space adventure."

# Generate a response
response = model.predict(prompt)

# Print the result
print("Generated Response:\n", response)
