"""Lab section: Zero-shot prompt (+ Exercise 2).

Zero-shot = clear instructions, but no examples. The model relies only on
what it already knows.

Run with:  uv run python scripts/02_zero_shot.py
"""

from promptlab.llm import llm_model
from promptlab.utils import show

# --- Lab example: classify a statement as true or false -------------------
prompt = """Classify the following statement as true or false:
'The Eiffel Tower is located in Berlin.'

Answer:"""
show("Zero-shot: fact check", prompt, llm_model(prompt))

# --- Exercise 2 ----------------------------------------------------------------

# 1. Movie review classification
movie_review_prompt = """Classify the following movie review as either 'positive' or 'negative'.
Reply with one word only.

Review: "The story dragged for two hours and the ending made no sense. I almost walked out."

Classification:"""

# 2. Summarize a paragraph about climate change
climate_change_prompt = """Summarize the following paragraph in no more than two sentences.

Paragraph: "Climate change is the long-term shift in global temperatures and weather
patterns. Since the 1800s, human activity, mainly burning coal, oil and gas, has been
the main cause, because it releases heat-trapping gases. The results include stronger
heat waves, droughts and storms, rising sea levels and melting glaciers, which affect
ecosystems, farming and human health."

Summary:"""

# 3. Translate English to Spanish (no examples given)
translation_prompt = """Translate the following English sentence into Spanish.

English: "Could you tell me where the train station is, please?"

Spanish:"""

exercises = {
    "Exercise 2.1: movie review": movie_review_prompt,
    "Exercise 2.2: climate summary": climate_change_prompt,
    "Exercise 2.3: translation": translation_prompt,
}

for title, prompt in exercises.items():
    show(title, prompt, llm_model(prompt))
