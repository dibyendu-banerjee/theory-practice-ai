# ================================================================
# File: blip_image_captioning.py
# Description: This script demonstrates image captioning using 
# BLIP (Bootstrapped Language Image Pretraining) from Salesforce. 
# It loads an image and generates a descriptive caption.
#
# Author: Dibyendu Banerjee
# ================================================================

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Load the BLIP processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load an image from local path
image_path = "path/to/image.jpg"  # Replace with your image path
image = Image.open(image_path).convert("RGB")

# Preprocess the image
inputs = processor(images=image, return_tensors="pt")

# Generate a caption
output = model.generate(**inputs)

# Decode and print the caption
caption = processor.decode(output[0], skip_special_tokens=True)
print("Generated Caption:", caption)
