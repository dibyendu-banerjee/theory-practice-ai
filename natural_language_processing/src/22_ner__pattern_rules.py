"""
NER - Pattern Rules
This script demonstrates: NER - Pattern Rules
"""

# Define a pattern for capitalized words (potential named entities)

pattern = r'\b[A-Z][a-z]*\b'



# Find all capitalized words

capitalized_words = re.findall(pattern, text)



print("Capitalized words