"""Shared test helpers.

Tests never call the real model: Ollama isn't available on GitHub's servers,
and real answers change from run to run. Instead we swap in a fake model that
returns answers we choose.
"""

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel


@pytest.fixture
def fake_llm(monkeypatch):
    """Replace get_llm everywhere with a fake model.

    Usage in a test:  fake_llm(["first answer", "second answer"])
    """
    def install(responses: list[str]):
        def fake_get_llm(params=None):
            return FakeListChatModel(responses=responses)

        # Each module imported get_llm under its own name, so patch all of them.
        for module in ["promptlab.llm", "promptlab.chains", "promptlab.techniques",
                       "promptlab.review_analyzer"]:
            monkeypatch.setattr(f"{module}.get_llm", fake_get_llm)

    return install
