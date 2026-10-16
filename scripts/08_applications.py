"""Lab sections: Text summarization, Question answering, Text classification,
Code generation (SQL).

Run with:  uv run python scripts/08_applications.py
"""

from promptlab.chains import classification_chain, qa_chain, sql_chain, summarization_chain


def header(title: str) -> None:
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


# --- Text summarization ----------------------------------------------------
header("Text summarization")
content = """The rapid advancement of technology in the 21st century has transformed
healthcare, education, and transportation. AI-powered diagnostic tools improve the
speed and accuracy of medical diagnoses, smart transportation systems reduce traffic
congestion, and online learning platforms make education accessible around the world.
These developments raise productivity and create a more connected society."""

print(summarization_chain().invoke({"content": content}))

# --- Question answering ------------------------------------------------------
header("Question answering")
context = """The solar system consists of the Sun, eight planets, their moons, dwarf
planets, and smaller objects like asteroids and comets. The inner planets (Mercury,
Venus, Earth, and Mars) are rocky and solid. The outer planets (Jupiter, Saturn,
Uranus, and Neptune) are much larger and gaseous."""

qa = qa_chain()  # build once, reuse for several questions

questions = [
    "Which planets in the solar system are rocky and solid?",
    # Not answerable from the context: the model should say it's unsure.
    "How many moons does Jupiter have?",
]
for question in questions:
    print(f"Q: {question}")
    print(f"A: {qa.invoke({'content': context, 'question': question})}\n")

# --- Text classification ----------------------------------------------------
header("Text classification")
text = "The concert last night was an exhilarating experience with outstanding performances by all artists."
categories = "Entertainment, Food and Dining, Technology, Literature, Music"

print(classification_chain().invoke({"text": text, "categories": categories}))

# --- Code generation (SQL) ------------------------------------------------
header("Code generation (SQL)")
description = """Retrieve the names and email addresses of all customers from the
'customers' table who have made a purchase in the last 30 days. The 'purchases'
table has the columns 'customer_id' and 'purchase_date'."""

print(sql_chain().invoke({"description": description}))
