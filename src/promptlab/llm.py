"""Connect LangChain to a local Ollama model.

This replaces the lab's `WatsonxLLM` setup. The model is still IBM Granite,
but it runs on your own computer through Ollama instead of IBM's cloud.
"""

import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

# Read settings from the .env file (if it exists) into environment variables.
load_dotenv()

# Which model to use and where Ollama is running. The second argument is the
# fallback value used when the .env file doesn't set it.
MODEL = os.getenv("OLLAMA_MODEL", "granite4:tiny-h")
BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Same default generation settings as the lab's `llm_model` function.
# (The lab's `min_new_tokens` is left out: Ollama has no equivalent setting.)
DEFAULT_PARAMS = {
    "max_new_tokens": 256,  # maximum length of the answer, in tokens
    "temperature": 0.5,     # randomness: low = focused, high = creative
    "top_p": 0.2,           # only sample from the most likely tokens (nucleus sampling)
    "top_k": 1,             # 1 = always pick the single most likely token
}


def get_llm(params: dict | None = None) -> ChatOllama:
    """Create a LangChain chat model, using the lab's parameter names.

    Any values in `params` override the defaults, e.g. {"temperature": 0.1}.
    """
    settings = {**DEFAULT_PARAMS, **(params or {})}

    return ChatOllama(
        model=MODEL,
        base_url=BASE_URL,
        num_predict=settings["max_new_tokens"],  # Ollama's name for max_new_tokens
        temperature=settings["temperature"],
        top_p=settings["top_p"],
        top_k=settings["top_k"],
    )


def llm_model(prompt_txt: str, params: dict | None = None) -> str:
    """Send one prompt to the model and return its answer as plain text.

    Same name and arguments as the lab's helper, so lab code works unchanged.
    """
    llm = get_llm(params)
    response = llm.invoke(prompt_txt)  # returns a message object
    return response.content            # keep only the text
