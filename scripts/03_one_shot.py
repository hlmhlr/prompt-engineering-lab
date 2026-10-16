"""Lab section: One-shot prompt (+ Exercise 3).

One-shot = give the model ONE example of the task, then ask it to do a new
one. The example shows the expected format and style.

Run with:  uv run python scripts/03_one_shot.py
"""

from promptlab.llm import llm_model
from promptlab.utils import show

# --- Lab example: English -> French with one example -----------------------
params = {"max_new_tokens": 20, "temperature": 0.1}

prompt = """Here is an example of translating a sentence from English to French:

English: "How is the weather today?"
French: "Comment est le temps aujourd'hui?"

Now, translate the following sentence from English to French:

English: "Where is the nearest supermarket?"
French:"""
show("One-shot: translation", prompt, llm_model(prompt, params))

# --- Exercise 3 ----------------------------------------------------------------

# 1. Formal email: one example, then a new topic
formal_email_prompt = """Here is an example of a formal email:

Subject: Request for Meeting Room Booking

Dear Facilities Team,

I would like to book Meeting Room B on Monday, 10 March, from 2:00 to 4:00 PM for a
project review with six participants. Could you please confirm availability?

Thank you for your help.

Kind regards,
Maria Rossi

---

Now write a formal email, in the same style, to a professor asking whether there are
open research assistant positions in their computer vision lab."""

# 2. Explain a technical concept in simple terms
technical_concept_prompt = """Here is an example of explaining a technical concept simply:

Technical concept: Cloud storage
Simple explanation: Cloud storage is like renting a locker in a building far away.
You can put your files in it and take them out from any device with internet, and
the company owning the building keeps it safe.

Now explain this concept in the same simple way:

Technical concept: Neural network
Simple explanation:"""

# 3. Keyword extraction
keyword_extraction_prompt = """Here is an example of extracting keywords from a sentence:

Sentence: "Electric cars reduce air pollution but need better charging infrastructure."
Keywords: electric cars, air pollution, charging infrastructure

Now extract the keywords from this sentence:

Sentence: "Computer vision models detect defects on factory assembly lines in real time."
Keywords:"""

exercises = {
    "Exercise 3.1: formal email": formal_email_prompt,
    "Exercise 3.2: simple explanation": technical_concept_prompt,
    "Exercise 3.3: keyword extraction": keyword_extraction_prompt,
}

for title, prompt in exercises.items():
    show(title, prompt, llm_model(prompt))
