# ================================================================
# File: gemini_api_demo.py
# Description: This script demonstrates how to use Google Cloud's 
# Vertex AI to access a Gemini model for text generation. It sends 
# a prompt and receives a prediction from the model.
#
# Author: Dibyendu Banerjee
# ================================================================

from google.cloud import aiplatform

# Initialize the Vertex AI platform
aiplatform.init(
    project="your-project-id",        # Replace with your GCP project ID
    location="us-central1"            # Replace with your region
)

# Load the deployed Gemini model
model = aiplatform.Model(
    model_name="projects/your-project-id/locations/us-central1/models/your-model-id"
)

# Send a prompt to the model
response = model.predict(instances=["Explain the concept of LLMs."])

# Print the model's prediction
print("Gemini Model Prediction:")
print(response)
