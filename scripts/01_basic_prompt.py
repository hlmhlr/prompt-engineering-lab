"""Lab section: Basic prompt (+ Exercise 1).

A basic prompt is a short phrase with no instructions. The model simply
continues or responds based on what it learned during training.

Run with:  uv run python scripts/01_basic_prompt.py
"""

from promptlab.llm import llm_model
from promptlab.utils import show

# Same generation settings as the lab.
params = {
    "max_new_tokens": 128,
    "temperature": 0.5,
    "top_p": 0.2,
    "top_k": 1,
}

# --- Lab example -------------------------------------------------------------
prompt = "The wind is "
show("Basic prompt", prompt, llm_model(prompt, params))

# --- Exercise 1: compare several basic prompts ----------------------------
prompts = [
    "The future of artificial intelligence is",
    "Once upon a time in a distant galaxy",
    "The benefits of sustainable energy include",
]

for prompt in prompts:
    show("Exercise 1", prompt, llm_model(prompt, params))
