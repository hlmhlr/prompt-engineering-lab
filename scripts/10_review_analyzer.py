"""Lab Exercise 5: Product review analyzer.

Steps from the lab:
  1. Create a prompt template for the review text.
  2. Build an LCEL chain.
  3. Process the sample reviews and display the results.
  4. Modify the chain to change the output format.

Run with:  uv run python scripts/10_review_analyzer.py
"""

from promptlab.chains import build_chain, review_chain

reviews = [
    "I love this smartphone! The camera quality is exceptional and the battery lasts "
    "all day. The only downside is that it heats up a bit during gaming.",
    "This laptop is terrible. It's slow, crashes frequently, and the keyboard stopped "
    "working after just two months. Customer service was unhelpful.",
]

# --- Steps 1-3: the lab's bullet-point format -------------------------------
chain = review_chain()

for i, review in enumerate(reviews, start=1):
    print(f"==== Review #{i} ====")
    print(chain.invoke({"review": review}))
    print()

# --- Step 4: same task, different output format (one compact line) -----------
COMPACT_TEMPLATE = """Analyze this product review:
"{review}"

Reply with ONE line in exactly this format, and nothing else:
<sentiment> | <feature 1, feature 2, ...> | <one-sentence summary>"""

compact_chain = build_chain(COMPACT_TEMPLATE, {"max_new_tokens": 200, "temperature": 0.2})

print("==== Step 4: compact format ====")
for i, review in enumerate(reviews, start=1):
    print(f"#{i}: {compact_chain.invoke({'review': review})}")
