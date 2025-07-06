# ================================================================
# File: langchain_templates.py
# Description: This script demonstrates prompt templating in 
# LangChain using SimpleTemplate, VariableTemplate, ConditionalTemplate, 
# MultiStepTemplate, and CompositeTemplate.
#
# Author: Dibyendu Banerjee
# ================================================================

from langchain import LLM
from langchain.templating import (
    SimpleTemplate,
    VariableTemplate,
    ConditionalTemplate,
    MultiStepTemplate,
    CompositeTemplate
)

# Initialize the LLM
model = LLM(model_name="gpt-3.5-turbo")

# -------------------------------
# SimpleTemplate
# -------------------------------

simple_template = SimpleTemplate(template="Translate the following text into {language}: {text}")

class SimpleTemplatedChain:
    def __init__(self, model: LLM, template: SimpleTemplate):
        self.model = model
        self.template = template

    def run(self, text: str, language: str) -> str:
        prompt = self.template.format(language=language, text=text)
        return self.model.predict(prompt)

# -------------------------------
# VariableTemplate
# -------------------------------

variable_template = VariableTemplate(
    template="For a {genre} book written by {author}, provide a summary of the plot."
)

class VariableTemplatedChain:
    def __init__(self, model: LLM, template: VariableTemplate):
        self.model = model
        self.template = template

    def run(self, genre: str, author: str) -> str:
        prompt = self.template.format(genre=genre, author=author)
        return self.model.predict(prompt)

# -------------------------------
# ConditionalTemplate
# -------------------------------

conditional_template = ConditionalTemplate(
    condition="if {condition} then provide a brief explanation of {topic}",
    else_prompt="Explain {topic} in detail."
)

class ConditionalTemplatedChain:
    def __init__(self, model: LLM, template: ConditionalTemplate):
        self.model = model
        self.template = template

    def run(self, topic: str, condition: bool) -> str:
        prompt = self.template.format(condition="yes" if condition else "no", topic=topic)
        return self.model.predict(prompt)

# -------------------------------
# MultiStepTemplate
# -------------------------------

multi_step_template = MultiStepTemplate(
    steps=[
        "Step 1: Describe the general concept of {topic}.",
        "Step 2: Provide examples related to {topic}.",
        "Step 3: Explain the significance of {topic} in today's context."
    ]
)

class MultiStepTemplatedChain:
    def __init__(self, model: LLM, template: MultiStepTemplate):
        self.model = model
        self.template = template

    def run(self, topic: str) -> str:
        prompt = self.template.format(topic=topic)
        return self.model.predict(prompt)

# -------------------------------
# CompositeTemplate
# -------------------------------

intro_template = SimpleTemplate(template="Provide an introduction to {topic}.")
details_template = VariableTemplate(template="Discuss the main aspects of {topic}, focusing on {aspect}.")

composite_template = CompositeTemplate(
    templates=[intro_template, details_template],
    separator="\n\n---\n\n"
)

class CompositeTemplatedChain:
    def __init__(self, model: LLM, template: CompositeTemplate):
        self.model = model
        self.template = template

    def run(self, topic: str, aspect: str) -> str:
        prompt = self.template.format(topic=topic, aspect=aspect)
        return self.model.predict(prompt)
