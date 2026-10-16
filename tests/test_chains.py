import pytest
from langchain_core.prompts import PromptTemplate

from promptlab import chains


@pytest.mark.parametrize("template, expected", [
    (chains.SUMMARY_TEMPLATE, {"content"}),
    (chains.QA_TEMPLATE, {"content", "question"}),
    (chains.CLASSIFICATION_TEMPLATE, {"text", "categories"}),
    (chains.SQL_TEMPLATE, {"description"}),
    (chains.REVIEW_TEMPLATE, {"review"}),
])
def test_templates_have_the_expected_placeholders(template, expected):
    assert set(PromptTemplate.from_template(template).input_variables) == expected


def test_qa_chain_returns_plain_text(fake_llm):
    fake_llm(["Unsure about answer"])
    answer = chains.qa_chain().invoke({"content": "Some context.", "question": "Unrelated?"})
    assert answer == "Unsure about answer"


def test_roleplay_puts_the_role_in_a_system_message():
    from langchain_core.prompts import ChatPromptTemplate
    chain = chains.roleplay_chain()
    prompt = chain.first  # the first step of the chain is the prompt template
    assert isinstance(prompt, ChatPromptTemplate)
    messages = prompt.format_messages(role="game master", tone="fun", question="Hi")
    assert messages[0].type == "system" and "game master" in messages[0].content
    assert messages[1].type == "human" and messages[1].content == "Hi"
