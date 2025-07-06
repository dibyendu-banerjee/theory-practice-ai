# ================================================================
# Chapter 11: Text Preprocessing with SpaCy
# Use Case: Tokenizing Text Using SpaCy NLP Pipeline
# ================================================================

import spacy

# --------------------------------------------------
# Step 1: Load SpaCy English Language Model
# --------------------------------------------------

# Make sure to run: python -m spacy download en_core_web_sm (if not already installed)
nlp = spacy.load('en_core_web_sm')

# --------------------------------------------------
# Step 2: Define Sample Text
# --------------------------------------------------

text = "My name is Dibyendu Banerjee. I am exploring Python."

# --------------------------------------------------
# Step 3: Process Text with SpaCy
# --------------------------------------------------

doc = nlp(text)

# --------------------------------------------------
# Step 4: Extract Tokens
# --------------------------------------------------

tokens = [token.text for token in doc]
print("📝 Tokens:")
print(tokens)

# --------------------------------------------------
# Optional: Extract Lemmas, POS Tags, and Named Entities
# --------------------------------------------------

lemmas = [token.lemma_ for token in doc]
pos_tags = [(token.text, token.pos_) for token in doc]
entities = [(ent.text, ent.label_) for ent in doc.ents]

print("\n🔤 Lemmas:")
print(lemmas)

print("\n🔠 Part-of-Speech Tags:")
print(pos_tags)

print("\n🏷️ Named Entities:")
print(entities)
