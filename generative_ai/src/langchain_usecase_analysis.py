# ================================================================
# File: langchain_usecase_analysis.py
# Description: Use Case 3 — Data extraction and analysis using 
# LangChain. Extracts key information from text and analyzes it 
# for insights, with optional validation and visualization.
#
# Author: Dibyendu Banerjee
# ================================================================

from langchain.llms import OpenAI
from langchain.prompts import SimplePromptTemplate
from langchain import Chain
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# Step 1: Initialize the LLM
# -------------------------------

model = OpenAI(api_key="your_openai_api_key", model_name="gpt-3.5-turbo")

# -------------------------------
# Step 2: Define prompt templates
# -------------------------------

extraction_template = SimplePromptTemplate(
    "Extract the key information from the following text: {text}"
)

analysis_template = SimplePromptTemplate(
    "Analyze the following data and provide insights: {data}"
)

# -------------------------------
# Step 3: Define extraction and analysis chains
# -------------------------------

class DataExtractionChain(Chain):
    def __init__(self, model, template):
        self.model = model
        self.template = template

    def run(self, text):
        prompt = self.template.format(text=text)
        return self.model.predict(prompt)

class DataAnalysisChain(Chain):
    def __init__(self, model, template):
        self.model = model
        self.template = template

    def run(self, data):
        prompt = self.template.format(data=data)
        return self.model.predict(prompt)

# -------------------------------
# Step 4: Optional — Data validation
# -------------------------------

def validate_data(data: str) -> str:
    try:
        df = pd.read_json(data)
        if df.empty:
            return "The data is empty. Please provide valid data."
        return "✅ Data is valid and ready for analysis."
    except Exception as e:
        return f"❌ Data validation failed: {e}"

# -------------------------------
# Step 5: Optional — Visualization
# -------------------------------

def visualize_data(data: str):
    try:
        df = pd.read_json(data)
        if df.empty:
            print("The data is empty. Please provide valid data for visualization.")
            return
        ax = df.plot(kind='bar', x='name', y='value', legend=False)
        ax.set_xlabel('Categories')
        ax.set_ylabel('Values')
        ax.set_title('Bar Chart Visualization')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error occurred while visualizing the data: {e}")

# -------------------------------
# Step 6: Run the system
# -------------------------------

def main():
    print("📊 Data Extraction and Analysis System")
    extraction_chain = DataExtractionChain(model, extraction_template)
    analysis_chain = DataAnalysisChain(model, analysis_template)

    while True:
        print("\nSelect operation:")
        print("1. Extract Data")
        print("2. Analyze Data")
        print("3. Validate Data")
        print("4. Visualize Data")
        print("5. Exit")

        choice = input("Enter choice (1-5): ")

        if choice == "1":
            text = input("Enter text for data extraction: ")
            result = extraction_chain.run(text)
            print("\n🔍 Extracted Data:\n", result)
        elif choice == "2":
            data = input("Enter JSON data for analysis: ")
            result = analysis_chain.run(data)
            print("\n📈 Analysis Insights:\n", result)
        elif choice == "3":
            data = input("Enter JSON data to validate: ")
            print(validate_data(data))
        elif choice == "4":
            data = input("Enter JSON data to visualize: ")
            visualize_data(data)
        elif choice == "5":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
