"""Lab section: Prompt template + LCEL chain.

A prompt template is a reusable prompt with placeholders. An LCEL chain
connects steps with the pipe operator: template | model | parser.

Run with:  uv run python scripts/07_prompt_template.py
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from promptlab.llm import get_llm

llm = get_llm({"max_new_tokens": 256, "temperature": 0.5})

# 1. Define a template with two placeholders.
template = "Tell me a {adjective} joke about {content}."
prompt = PromptTemplate.from_template(template)

print("Placeholders found:", prompt.input_variables)

# 2. See what the filled-in prompt looks like (no model call yet).
print("Formatted prompt:  ", prompt.format(adjective="funny", content="chickens"))

# 3. Build the chain. Data flows left to right:
#    dict of values -> filled prompt -> model's message -> plain text
joke_chain = prompt | llm | StrOutputParser()

# 4. Run it. The dictionary keys must match the placeholder names.
print("\n--- funny / chickens ---")
print(joke_chain.invoke({"adjective": "funny", "content": "chickens"}))

# 5. Reuse the same chain with different values.
print("\n--- sad / fish ---")
print(joke_chain.invoke({"adjective": "sad", "content": "fish"}))
