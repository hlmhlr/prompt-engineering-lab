"""Lab section: Role playing.

The model takes on a role (a Dungeons & Dragons game master) and keeps a
chosen tone. Type your questions; type quit, exit or bye to stop.

Try: "Who are you?", "How do I create a balanced encounter for my players?",
     "Describe a mysterious forest setting for my adventure."

Run with:  uv run python scripts/09_role_playing.py
"""

from promptlab.chains import roleplay_chain

role = "Dungeons & Dragons game master"
tone = "engaging and immersive"

chain = roleplay_chain()

print(f"You are talking to: {role} (type 'quit' to stop)\n")

while True:
    query = input("Question: ").strip()

    if query.lower() in ["quit", "exit", "bye"]:
        print("Answer: Goodbye!")
        break

    if not query:  # ignore empty lines
        continue

    response = chain.invoke({"role": role, "tone": tone, "question": query})
    print(f"Answer: {response}\n")
