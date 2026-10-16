"""Lab section: Few-shot prompt.

Few-shot = give the model several examples (usually 2-5). More examples make
the pattern clearer than a single one.

Run with:  uv run python scripts/04_few_shot.py
"""

from promptlab.llm import llm_model
from promptlab.utils import show

# Only 10 tokens: we want a one-word label, not an explanation.
params = {"max_new_tokens": 10}

prompt = """Here are a few examples of classifying emotions in statements:

Statement: 'I just won my first marathon!'
Emotion: Joy

Statement: 'I can't believe I lost my keys again.'
Emotion: Frustration

Statement: 'My best friend is moving to another country.'
Emotion: Sadness

Now, classify the emotion in the following statement:
Statement: 'That movie was so scary I had to cover my eyes.'
Emotion:"""

show("Few-shot: emotion classification", prompt, llm_model(prompt, params))
