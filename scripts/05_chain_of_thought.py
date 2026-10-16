"""Lab section: Chain-of-thought (CoT) prompt (+ Exercise 4).

CoT = ask the model to reason step by step before answering. This helps on
math, logic and decisions with several steps.

Run with:  uv run python scripts/05_chain_of_thought.py
"""

from promptlab.llm import llm_model
from promptlab.utils import show

params = {"max_new_tokens": 512, "temperature": 0.5}

problem = (
    "A store had 22 apples. They sold 15 apples today and got a new delivery "
    "of 8 apples. How many apples are there now?"
)

# --- Small addition: the same question WITHOUT chain-of-thought -----------
# Comparing both versions makes the effect of CoT visible.
direct_prompt = f"{problem}\nAnswer with a number only."
show("Without CoT (direct answer)", direct_prompt, llm_model(direct_prompt, params))

# --- Lab example: with chain-of-thought -------------------------------------
cot_prompt = f"""Consider the problem: '{problem}'

Break down each step of your calculation."""
show("With CoT", cot_prompt, llm_model(cot_prompt, params))

# --- Exercise 4 ----------------------------------------------------------------

# 1. Decision-making, step by step
decision_making_prompt = """A student has a test in two days. Tonight, friends invited them
to a movie. Should the student study tonight or go to the movie?

Think through this step by step:
1. List the pros and cons of each option.
2. Consider how much time is left before the test.
3. Give a final recommendation and explain it in one sentence."""

# 2. Explaining a process, step by step
sandwich_making_prompt = """Explain how to make a peanut butter and jelly sandwich.

Break the process into numbered steps, from gathering the ingredients to
serving the sandwich."""

exercises = {
    "Exercise 4.1: decision making": decision_making_prompt,
    "Exercise 4.2: step-by-step process": sandwich_making_prompt,
}

for title, prompt in exercises.items():
    show(title, prompt, llm_model(prompt, params))
