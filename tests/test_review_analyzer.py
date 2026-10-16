import pytest
from pydantic import ValidationError

from promptlab.review_analyzer import ReviewAnalysis, analyze_review, report_markdown, summarize


def make(sentiment, features):
    return ReviewAnalysis(sentiment=sentiment, features=features, summary="...")


def test_schema_rejects_unknown_sentiment():
    with pytest.raises(ValidationError):
        ReviewAnalysis(sentiment="mostly good", features=[], summary="...")


def test_summarize_counts_and_normalizes_features():
    results = [
        make("positive", ["Camera", "battery life"]),
        make("negative", ["Battery Life ", "keyboard"]),
        make("negative", ["battery life"]),
    ]
    stats = summarize(results, failed=1)

    assert stats["total"] == 3
    assert stats["failed"] == 1
    assert stats["sentiments"] == {"positive": 1, "negative": 2, "neutral": 0}
    assert stats["top_features"][0] == ("battery life", 3)  # 3 spellings, one feature
    assert dict(stats["top_negative_features"]) == {"battery life": 2, "keyboard": 1}


def test_report_contains_the_numbers():
    report = report_markdown(summarize([make("positive", ["camera"])]))
    assert "| positive | 1 | 100% |" in report
    assert "| camera | 1 |" in report


class FlakyChain:
    """Fails a given number of times, then returns a valid analysis."""

    def __init__(self, failures):
        self.failures = failures

    def invoke(self, inputs):
        if self.failures > 0:
            self.failures -= 1
            raise ValueError("invalid JSON")
        return make("neutral", [])


def test_analyze_review_retries_once():
    assert analyze_review(FlakyChain(failures=1), "ok").sentiment == "neutral"


def test_analyze_review_gives_up_after_retries():
    with pytest.raises(ValueError):
        analyze_review(FlakyChain(failures=2), "ok")
