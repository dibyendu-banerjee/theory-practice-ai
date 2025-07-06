# ================================================================
# File: langchain_usecase_convo.py
# Description: Use Case 1 — Building a conversational agent using 
# LangChain with context management and prompt templating.
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
# Step 2: Define a prompt template
# -------------------------------

prompt_template = SimplePromptTemplate(
    "You are a helpful assistant. Answer the following query based on your knowledge: {query}"
)

# -------------------------------
# Step 3: Define a basic conversation chain
# -------------------------------

class CustomerServiceChain(Chain):
    def __init__(self, model, prompt_template):
        self.model = model
        self.prompt_template = prompt_template

    def run(self, user_query):
        prompt = self.prompt_template.format(query=user_query)
        return self.model.predict(prompt)

# -------------------------------
# Step 4: Add context management
# -------------------------------

class ContextualCustomerServiceChain(CustomerServiceChain):
    def __init__(self, model, prompt_template):
        super().__init__(model, prompt_template)
        self.context = ""

    def run(self, user_query):
        self.context += f"User: {user_query}\n"
        prompt = self.prompt_template.format(query=self.context)
        response = self.model.predict(prompt)
        self.context += f"Agent: {response}\n"
        return response

# -------------------------------
# Step 5: Run the conversational agent
# -------------------------------

def main():
    print("🤖 Welcome to the Customer Service Agent!")
    chain = ContextualCustomerServiceChain(model=model, prompt_template=prompt_template)

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Agent: Goodbye!")
            break
        response = chain.run(user_input)
        print("Agent:", response)

if __name__ == "__main__":
    main()
