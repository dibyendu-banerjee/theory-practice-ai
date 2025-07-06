# ================================================================
# File: langchain_usecase_content.py
# Description: Use Case 2 — Generating dynamic content using 
# LangChain. Supports article generation, summarization, and 
# creative writing based on user input.
#
# Author: Dibyendu Banerjee
# ================================================================

from langchain.llms import OpenAI
from langchain.prompts import SimplePromptTemplate
from langchain import Chain

# -------------------------------
# Step 1: Initialize the LLM
# -------------------------------

model = OpenAI(api_key="your_openai_api_key", model_name="gpt-3.5-turbo")

# -------------------------------
# Step 2: Define prompt templates
# -------------------------------

article_template = SimplePromptTemplate(
    "Write a comprehensive article about {topic}, highlighting the latest trends and offering valuable insights."
)

summary_template = SimplePromptTemplate(
    "Please summarize the following text: {text}"
)

creative_writing_template = SimplePromptTemplate(
    "Create an imaginative short story inspired by this prompt: {prompt}"
)

# -------------------------------
# Step 3: Define content generation chains
# -------------------------------

class ArticleChain(Chain):
    def __init__(self, model, template):
        self.model = model
        self.template = template

    def run(self, topic):
        prompt = self.template.format(topic=topic)
        return self.model.predict(prompt)

class SummaryChain(Chain):
    def __init__(self, model, template):
        self.model = model
        self.template = template

    def run(self, text):
        prompt = self.template.format(text=text)
        return self.model.predict(prompt)

class CreativeWritingChain(Chain):
    def __init__(self, model, template):
        self.model = model
        self.template = template

    def run(self, prompt_text):
        prompt = self.template.format(prompt=prompt_text)
        return self.model.predict(prompt)

# -------------------------------
# Step 4: Run the content generator
# -------------------------------

def main():
    print("📝 Dynamic Content Generator")
    article_chain = ArticleChain(model, article_template)
    summary_chain = SummaryChain(model, summary_template)
    creative_chain = CreativeWritingChain(model, creative_writing_template)

    while True:
        print("\nSelect content type:")
        print("1. Article")
        print("2. Summary")
        print("3. Creative Writing")
        print("4. Exit")

        choice = input("Enter choice (1-4): ")

        if choice == "1":
            topic = input("Enter topic for the article: ")
            result = article_chain.run(topic)
            print("\n📰 Generated Article:\n", result)
        elif choice == "2":
            text = input("Enter text to summarize: ")
            result = summary_chain.run(text)
            print("\n📄 Summary:\n", result)
        elif choice == "3":
            prompt = input("Enter prompt for creative writing: ")
            result = creative_chain.run(prompt)
            print("\n🎨 Creative Writing:\n", result)
        elif choice == "4":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
