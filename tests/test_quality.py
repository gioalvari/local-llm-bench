import pytest

from localllm_bench.quality import (
    normalize_answer,
    normalized_exact_match,
    numeric_match,
    token_f1,
)


def test_normalize_answer() -> None:
    assert normalize_answer("The Settlement, Interval!") == "settlement interval"


def test_normalized_exact_match_accepts_alias() -> None:
    assert normalized_exact_match("The 15 minutes.", ["15 minutes", "15 min"]) == 1.0


def test_token_f1_uses_best_reference() -> None:
    assert token_f1(
        "settlement every 15 minutes", ["hourly", "15 minutes"]
    ) == pytest.approx(2 / 3)
    assert token_f1("", [""]) == 1.0
    assert token_f1("anything", []) == 0.0


def test_numeric_match_uses_declared_tolerance() -> None:
    assert numeric_match("15.2 minutes", 15.0, absolute_tolerance=0.25) == 1.0
    assert numeric_match("15,2 minutes", 15.0, absolute_tolerance=0.1) == 0.0
    assert numeric_match("unknown", 15.0, absolute_tolerance=1.0) == 0.0
