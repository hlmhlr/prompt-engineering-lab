import pytest

from promptlab.techniques import extract_answer, self_consistency


@pytest.mark.parametrize("text, expected", [
    ("Step 1...\nAnswer: 67", "67"),
    ("**Answer:** 67", "67"),               # markdown bold
    ("answer - 35", "35"),                  # different case and separator
    ("Answer: 3\n...rethinking...\nAnswer: 67", "67"),  # the last one wins
    ("I am not sure.", None),               # no answer line
])
def test_extract_answer(text, expected):
    assert extract_answer(text) == expected


def test_self_consistency_majority_vote(fake_llm):
    fake_llm(["Answer: 67", "Answer: 35", "Answer: 67", "Answer: 67", "no idea"])
    result = self_consistency("When I was 6...", num_samples=5)
    assert result["winner"] == "67"
    assert result["count"] == 3
    assert result["votes"] == {"67": 3, "35": 1, "no answer found": 1}
