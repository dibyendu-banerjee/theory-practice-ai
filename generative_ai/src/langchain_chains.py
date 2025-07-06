# ================================================================
# File: langchain_chains.py
# Description: This script demonstrates various chain types in 
# LangChain, including Simple, Sequential, Conditional, Parallel, 
# Recursive, and Composite chains.
#
# Author: Dibyendu Banerjee
# ================================================================

from langchain import Chain, LLM

# -------------------------------
# Simple Chain
# -------------------------------

class SimpleChain(Chain):
    def __init__(self, model: LLM):
        self.model = model

    def run(self, input_text: str) -> str:
        return self.model.predict(input_text)

# -------------------------------
# Sequential Chain
# -------------------------------

class SequentialChain(Chain):
    def __init__(self, pre: LLM, core: LLM, post: LLM):
        self.pre = pre
        self.core = core
        self.post = post

    def run(self, input_text: str) -> str:
        step1 = self.pre.predict(f"Preprocess: {input_text}")
        step2 = self.core.predict(f"Process: {step1}")
        return self.post.predict(f"Postprocess: {step2}")

# -------------------------------
# Conditional Chain
# -------------------------------

class ConditionalChain(Chain):
    def __init__(self, condition_model: LLM, true_model: LLM, false_model: LLM):
        self.condition_model = condition_model
        self.true_model = true_model
        self.false_model = false_model

    def run(self, input_text: str) -> str:
        condition = self.condition_model.predict(f"Condition for: {input_text}")
        if "true" in condition.lower():
            return self.true_model.predict(f"True branch: {input_text}")
        else:
            return self.false_model.predict(f"False branch: {input_text}")

# -------------------------------
# Parallel Chain
# -------------------------------

class ParallelChain(Chain):
    def __init__(self, task1_model: LLM, task2_model: LLM):
        self.task1_model = task1_model
        self.task2_model = task2_model

    def run(self, input_text: str) -> dict:
        return {
            "task1": self.task1_model.predict(f"Task 1: {input_text}"),
            "task2": self.task2_model.predict(f"Task 2: {input_text}")
        }

# -------------------------------
# Recursive Chain
# -------------------------------

class RecursiveChain(Chain):
    def __init__(self, model: LLM, max_depth: int):
        self.model = model
        self.max_depth = max_depth

    def run(self, input_text: str, depth: int = 0) -> str:
        if depth >= self.max_depth:
            return input_text
        refined = self.model.predict(f"Refine: {input_text}")
        return self.run(refined, depth + 1)

# -------------------------------
# Composite Chain
# -------------------------------

class CompositeChain(Chain):
    def __init__(self, sequential_chain: Chain, parallel_chain: Chain):
        self.sequential_chain = sequential_chain
        self.parallel_chain = parallel_chain

    def run(self, input_text: str) -> dict:
        seq_result = self.sequential_chain.run(input_text)
