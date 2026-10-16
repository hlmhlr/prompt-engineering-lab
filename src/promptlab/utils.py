"""Small helpers shared by the lab scripts."""


def show(title: str, prompt: str, response: str) -> None:
    """Print a prompt and the model's response in a readable block."""
    line = "=" * 70
    print(f"\n{line}\n{title}\n{line}")
    print(f"PROMPT:\n{prompt.strip()}\n")
    print(f"RESPONSE:\n{response.strip()}\n")
