"""Check that Python can talk to your local Ollama model.

Run with:  uv run python scripts/00_check_setup.py
"""

from promptlab.llm import BASE_URL, MODEL, llm_model

print(f"Model:  {MODEL}")
print(f"Server: {BASE_URL}\n")

try:
    answer = llm_model("In one sentence, what is prompt engineering?")
except Exception as error:
    # Most common causes: Ollama isn't running, or the model isn't downloaded.
    print("Could not reach the model.")
    print(f"Error: {error}\n")
    print("Fixes to try:")
    print("  1. Start the Ollama app (or run `ollama serve` in another terminal).")
    print(f"  2. Download the model: ollama pull {MODEL}")
    raise SystemExit(1)

print("Model answered:")
print(answer)
print("\nSetup works. You're ready for the next stage.")
