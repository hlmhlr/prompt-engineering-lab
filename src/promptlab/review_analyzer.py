"""Structured product review analysis (extends the lab's Exercise 5).

Instead of free text, the model returns data in a fixed shape (a Pydantic
model), so code can count, filter and save the results.
"""

from collections import Counter
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from promptlab.llm import MODEL, get_llm


# --- 1. The output shape we want from the model ------------------------------

class ReviewAnalysis(BaseModel):
    """The analysis of one product review."""

    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="Overall sentiment of the review"
    )
    features: list[str] = Field(
        description="Product features mentioned, as short lowercase phrases, e.g. 'battery life'"
    )
    summary: str = Field(description="One-sentence summary of the review")


# --- 2. The prompt (same task as Exercise 5, written as chat messages) -------

PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You analyze customer product reviews. Identify the overall sentiment "
        "(positive, negative, or neutral), list the product features the customer "
        "mentions, and write a one-sentence summary. Use short, consistent feature "
        "names such as 'battery life', 'camera', 'customer service'.",
    ),
    ("human", "Review:\n{review}"),
])


# --- 3. The chain ---------------------------------------------------------------

def review_analysis_chain(params: dict | None = None):
    """Chain: prompt | model forced to answer in the ReviewAnalysis shape."""
    # Temperature 0: same review -> same analysis, which is what we want for data.
    llm = get_llm(params or {"max_new_tokens": 512, "temperature": 0})
    return PROMPT | llm.with_structured_output(ReviewAnalysis)


def analyze_review(chain, review: str, retries: int = 1) -> ReviewAnalysis:
    """Analyze one review, trying again if the model's answer is invalid."""
    last_error = None
    for _attempt in range(retries + 1):
        try:
            result = chain.invoke({"review": review})
            if result is not None:
                return result
            last_error = ValueError("model returned no result")
        except Exception as error:  # invalid JSON, validation error, connection...
            last_error = error
    raise last_error


# --- 4. Turning many analyses into a report --------------------------------

def summarize(results: list[ReviewAnalysis], failed: int = 0) -> dict:
    """Count sentiments and features across all analyzed reviews."""
    sentiments = Counter(r.sentiment for r in results)

    # Normalize feature names so "Battery Life" and "battery life" count as one.
    all_features = Counter(f.strip().lower() for r in results for f in r.features)
    negative_features = Counter(
        f.strip().lower() for r in results if r.sentiment == "negative" for f in r.features
    )

    return {
        "total": len(results),
        "failed": failed,
        "sentiments": {s: sentiments.get(s, 0) for s in ("positive", "negative", "neutral")},
        "top_features": all_features.most_common(5),
        "top_negative_features": negative_features.most_common(5),
    }


def report_markdown(stats: dict) -> str:
    """Write the summary as a Markdown report."""
    total = stats["total"]
    lines = [
        "# Review Analysis Report",
        "",
        f"- Reviews analyzed: **{total}**" + (f" ({stats['failed']} failed)" if stats["failed"] else ""),
        f"- Model: `{MODEL}` (local, via Ollama)",
        "",
        "## Sentiment",
        "",
        "| Sentiment | Reviews | Share |",
        "|---|---|---|",
    ]
    for sentiment, count in stats["sentiments"].items():
        share = f"{count / total:.0%}" if total else "-"
        lines.append(f"| {sentiment} | {count} | {share} |")

    def feature_table(title: str, items: list[tuple[str, int]]) -> list[str]:
        if not items:
            return ["", f"## {title}", "", "None found."]
        rows = [f"| {feature} | {count} |" for feature, count in items]
        return ["", f"## {title}", "", "| Feature | Mentions |", "|---|---|", *rows]

    lines += feature_table("Most mentioned features", stats["top_features"])
    lines += feature_table("Most mentioned in negative reviews", stats["top_negative_features"])
    return "\n".join(lines) + "\n"
