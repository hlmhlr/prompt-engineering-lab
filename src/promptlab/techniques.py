"""Prompting technique examples and real self-consistency, for the demo app."""

import re
from collections import Counter

from promptlab.llm import get_llm

# One ready-made example per technique (same prompts as the lab scripts).
EXAMPLES = {
    "Zero-shot": {
        "description": "Clear instructions, no examples. The model relies only on what it already knows.",
        "prompt": "Classify the following statement as true or false:\n"
                  "'The Eiffel Tower is located in Berlin.'\n\nAnswer:",
        "params": {"max_new_tokens": 128, "temperature": 0.5, "top_k": 1},
    },
    "One-shot": {
        "description": "One example of the task shows the model the expected format.",
        "prompt": 'Here is an example of translating a sentence from English to French:\n\n'
                  'English: "How is the weather today?"\n'
                  'French: "Comment est le temps aujourd\'hui?"\n\n'
                  'Now, translate the following sentence from English to French:\n\n'
                  'English: "Where is the nearest supermarket?"\nFrench:',
        "params": {"max_new_tokens": 30, "temperature": 0.1, "top_k": 1},
    },
    "Few-shot": {
        "description": "Several examples make the pattern clearer than a single one.",
        "prompt": "Here are a few examples of classifying emotions in statements:\n\n"
                  "Statement: 'I just won my first marathon!'\nEmotion: Joy\n\n"
                  "Statement: 'I can't believe I lost my keys again.'\nEmotion: Frustration\n\n"
                  "Statement: 'My best friend is moving to another country.'\nEmotion: Sadness\n\n"
                  "Now, classify the emotion in the following statement:\n"
                  "Statement: 'That movie was so scary I had to cover my eyes.'\nEmotion:",
        "params": {"max_new_tokens": 10, "temperature": 0.5, "top_k": 1},
    },
    "Chain-of-thought": {
        "description": "Asking for step-by-step reasoning improves multi-step problems.",
        "prompt": "Consider the problem: 'A store had 22 apples. They sold 15 apples today and "
                  "got a new delivery of 8 apples. How many apples are there now?'\n\n"
                  "Break down each step of your calculation.",
        "params": {"max_new_tokens": 512, "temperature": 0.5, "top_k": 1},
    },
    "Self-consistency": {
        "description": "Solve the problem several independent times, then keep the most common answer.",
        "prompt": "When I was 6, my sister was half of my age. Now I am 70, what age is my sister?",
        "params": {"max_new_tokens": 300, "temperature": 0.8, "top_k": 40},
    },
}

# Finds "Answer: 67" (also "**Answer:** 67", "ANSWER - 67", ...).
ANSWER_PATTERN = re.compile(r"answer\W*(-?\d+)", re.IGNORECASE)


def extract_answer(text: str) -> str | None:
    """Return the number in the last 'Answer: <number>' line, if any."""
    matches = ANSWER_PATTERN.findall(text)
    return matches[-1] if matches else None


def self_consistency(question: str, num_samples: int = 5, params: dict | None = None) -> dict:
    """Ask the same question several times independently and vote on the answer."""
    # top_p 0.9 / top_k 40 / temperature 0.8: enough variety for samples to differ.
    llm = get_llm({"max_new_tokens": 300, "temperature": 0.8, "top_p": 0.9, "top_k": 40, **(params or {})})
    prompt = (
        f"{question}\n\nThink step by step. End with a final line in exactly this format:\n"
        "Answer: <number>"
    )

    samples = []
    for _ in range(num_samples):
        text = llm.invoke(prompt).content
        samples.append({"answer": extract_answer(text) or "no answer found", "reasoning": text})

    votes = Counter(sample["answer"] for sample in samples)
    winner, count = votes.most_common(1)[0]
    return {"samples": samples, "votes": dict(votes), "winner": winner, "count": count}
