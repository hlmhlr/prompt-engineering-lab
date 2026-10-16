"""Lab section: Self-consistency.

Self-consistency = solve the same problem several independent times and keep
the answer that comes up most often.

Part 1 is the lab's version: one prompt asks for three solutions.
Part 2 is how it's done in practice: separate model calls + majority vote.

Run with:  uv run python scripts/06_self_consistency.py
"""

import re
from collections import Counter

from promptlab.llm import get_llm, llm_model
from promptlab.utils import show

question = "When I was 6, my sister was half of my age. Now I am 70, what age is my sister?"

# --- Part 1: lab version (all three solutions in a single answer) ----------
lab_prompt = f"""{question}

Provide three independent calculations and explanations, then determine the most
consistent result."""
show("Self-consistency (lab version)", lab_prompt, llm_model(lab_prompt, {"max_new_tokens": 512}))

# --- Part 2: real self-consistency (independent samples + voting) -----------
# Higher temperature, top_p and top_k let each sample take a different reasoning
# path. With the default top_k=1, every sample would be identical.
sampling_llm = get_llm({
    "max_new_tokens": 300,
    "temperature": 0.8,
    "top_p": 0.9,
    "top_k": 40,
})

sample_prompt = f"""{question}

Think step by step. End with a final line in exactly this format:
Answer: <number>"""

NUM_SAMPLES = 5
answers = []

print("\n" + "=" * 70)
print(f"Self-consistency (real version): {NUM_SAMPLES} independent samples")
print("=" * 70)

for i in range(NUM_SAMPLES):
    text = sampling_llm.invoke(sample_prompt).content

    # Find every "Answer: <number>" in the text and keep the last one.
    matches = re.findall(r"answer\W*(-?\d+)", text, flags=re.IGNORECASE)
    answer = matches[-1] if matches else "no answer found"

    answers.append(answer)
    print(f"Sample {i + 1}: {answer}")

votes = Counter(answers)
winner, count = votes.most_common(1)[0]
print(f"\nVotes: {dict(votes)}")
print(f"Final answer by majority vote: {winner} ({count}/{NUM_SAMPLES} samples agree)")
