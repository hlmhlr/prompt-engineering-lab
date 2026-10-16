"""Reusable LangChain chains for the lab's applications.

Each function returns a ready-to-use chain built with LCEL:

    prompt template  |  LLM  |  output parser

The scripts in `scripts/` call these, and the Streamlit demo will reuse them.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from promptlab.llm import get_llm

# Generation settings used in the lab's "Applications" section.
APP_PARAMS = {"max_new_tokens": 256, "temperature": 0.5}

# --- Prompt templates ----------------------------------------------------
# Words in {curly braces} are placeholders, filled in when the chain runs.

SUMMARY_TEMPLATE = """Summarize the following text in one sentence.

Text: {content}

Summary:"""

QA_TEMPLATE = """Answer the question using only the context below.
If the context does not contain the answer, reply exactly: Unsure about answer

Context: {content}

Question: {question}

Answer:"""

CLASSIFICATION_TEMPLATE = """Classify the text into exactly one of these categories: {categories}
Reply with the category name only.

Text: {text}

Category:"""

SQL_TEMPLATE = """Write an SQL query for the following request.
Return only the SQL query, with no explanation.

Request: {description}

SQL query:"""

REVIEW_TEMPLATE = """Analyze the following product review:
"{review}"

Provide your analysis in the following format:
- Sentiment: (positive, negative, or neutral)
- Key Features Mentioned: (list the product features mentioned)
- Summary: (one-sentence summary)"""


# --- Chain builders ------------------------------------------------------

def build_chain(template: str, params: dict | None = None):
    """Turn a template string into a chain: prompt | llm | text parser."""
    prompt = PromptTemplate.from_template(template)
    llm = get_llm(params or APP_PARAMS)
    return prompt | llm | StrOutputParser()


def summarization_chain(params: dict | None = None):
    """Input: {"content": ...}"""
    return build_chain(SUMMARY_TEMPLATE, params)


def qa_chain(params: dict | None = None):
    """Input: {"content": ..., "question": ...}"""
    return build_chain(QA_TEMPLATE, params)


def classification_chain(params: dict | None = None):
    """Input: {"text": ..., "categories": ...}"""
    return build_chain(CLASSIFICATION_TEMPLATE, params)


def sql_chain(params: dict | None = None):
    """Input: {"description": ...}"""
    return build_chain(SQL_TEMPLATE, params)


def review_chain(params: dict | None = None):
    """Input: {"review": ...}  (Exercise 5)"""
    return build_chain(REVIEW_TEMPLATE, params or {"max_new_tokens": 512, "temperature": 0.2})


def roleplay_chain(params: dict | None = None):
    """Input: {"role": ..., "tone": ..., "question": ...}

    Uses a chat template: the role goes in a *system* message (instructions
    the model follows for the whole conversation), the question in a *human*
    message.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert {role}. Keep the conversation {tone}."),
        ("human", "{question}"),
    ])
    llm = get_llm(params or APP_PARAMS)
    return prompt | llm | StrOutputParser()
