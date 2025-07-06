# ================================================================
# File: langchain_agents.py
# Description: This script demonstrates different types of agents 
# in LangChain, including Action Agents, Conversational Agents, 
# Decision-Making Agents, Tool-Using Agents, and Integrated Agents.
#
# Author: Dibyendu Banerjee
# ================================================================

from langchain.agents import BaseAgent, ActionAgent, ConversationalAgent
from langchain.llms import OpenAI
from langchain.prompts import SimplePromptTemplate

# -------------------------------
# Action Agent
# -------------------------------

class SimpleActionAgent(BaseAgent):
    def perform_action(self, action, *args, **kwargs):
        if action == "print_message":
            print(kwargs.get("message", "No message provided"))
        elif action == "add_numbers":
            numbers = kwargs.get("numbers", [])
            print(f"Sum: {sum(numbers)}")
        else:
            print("Unknown action")

# -------------------------------
# Conversational Agent
# -------------------------------

class MyConversationalAgent(ConversationalAgent):
    def __init__(self, llm, prompt_template):
        super().__init__(llm=llm, prompt_template=prompt_template)

    def respond(self, user_input):
        return self.llm.predict(self.prompt_template.format(user_input=user_input))

# -------------------------------
# Decision-Making Agent
# -------------------------------

class DecisionMakingAgent(BaseAgent):
    def __init__(self, llm, prompt_template):
        self.llm = llm
        self.prompt_template = prompt_template

    def make_decision(self, options, criteria):
        prompt = self.prompt_template.format(options=options, criteria=criteria)
        return self.llm.predict(prompt)

# -------------------------------
# Tool-Using Agent
# -------------------------------

import requests

class ToolUsingAgent(BaseAgent):
    def perform_tool_action(self, tool_name, *args, **kwargs):
        if tool_name == "weather_api":
            location = kwargs.get("location", "New York")
            api_key = kwargs.get("api_key", "your_api_key")
            response = requests.get(f"https://api.weatherapi.com/v1/current.json?key={api_key}&q={location}")
            weather_data = response.json()
            return f"Current temperature in {location} is {weather_data['current']['temp_c']}°C."
        else:
            return "Unknown tool"

# -------------------------------
# Integrated Agent
# -------------------------------

class IntegratedAgent(BaseAgent):
    def __init__(self, conversation_agent, action_agent):
        self.conversation_agent = conversation_agent
        self.action_agent = action_agent

    def handle_interaction(self, user_input):
        response = self.conversation_agent.respond(user_input)
        if "action" in response.lower():
            self.action_agent.perform_action("print_message", message="Executing action based on user input")
        return response

# -------------------------------
# Example Initialization
# -------------------------------

# Replace with your actual OpenAI API key
model = OpenAI(api_key="your_openai_api_key", model_name="gpt-3.5-turbo")

conversation_prompt = SimplePromptTemplate("User: {user_input}\nAgent:")

# Instantiate agents
conversation_agent = MyConversationalAgent(llm=model, prompt_template=conversation_prompt)
action_agent = SimpleActionAgent()
decision_prompt = SimplePromptTemplate(
    "Given the following options: {options}, which one is the best choice based on the criteria: {criteria}?"
)
decision_agent = DecisionMakingAgent(llm=model, prompt_template=decision_prompt)

# Integrated agent usage
integrated_agent = IntegratedAgent(conversation_agent=conversation_agent, action_agent=action_agent)
response = integrated_agent.handle_interaction("Trigger an action")
print("Integrated Agent Response:", response)
