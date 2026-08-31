import pytest

from localllm_bench.quality import (
    normalize_answer,
    normalized_exact_match,
    numeric_match,
    parse_structured_answer,
    score_structured_answer,
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
    assert numeric_match("1,000 Watts", 1000.0) == 1.0
    assert numeric_match("1,000,000 Watts", 1_000_000.0) == 1.0


def test_parse_structured_answer_requires_exact_schema() -> None:
    parsed = parse_structured_answer('{"answer":"natural gas","unit":null}')
    assert parsed is not None
    assert parsed.answer == "natural gas"
    assert parse_structured_answer("```json\n{}\n```") is None
    assert parse_structured_answer('{"answer":"x","extra":1}') is None
    assert parse_structured_answer('{"answer":"x"}') is None


def test_score_structured_text_answer() -> None:
    score = score_structured_answer(
        '{"answer":"Natural gas","unit":null}',
        answer_type="classification",
        references=["natural gas"],
    )
    assert score.schema_valid == 1.0
    assert score.answer_accuracy == 1.0


def test_score_structured_numeric_requires_number_and_unit() -> None:
    score = score_structured_answer(
        '{"answer":"24","unit":"percent"}',
        answer_type="numeric",
        references=["24"],
        expected_value=24,
        accepted_units=["percent", "%"],
    )
    assert score.answer_accuracy == 1.0
    wrong_unit = score_structured_answer(
        '{"answer":"24","unit":"MW"}',
        answer_type="numeric",
        references=["24"],
        expected_value=24,
        accepted_units=["percent"],
    )
    assert wrong_unit.numeric_accuracy == 1.0
    assert wrong_unit.answer_accuracy == 0.0


def test_invalid_schema_scores_zero() -> None:
    score = score_structured_answer(
        "24 percent", answer_type="numeric", references=["24"], expected_value=24
    )
    assert score.schema_valid == 0.0
    assert score.answer_accuracy == 0.0


def test_fenced_json_is_scored_but_not_schema_valid() -> None:
    score = score_structured_answer(
        '```json\n{"answer":"natural gas","unit":null}\n```',
        answer_type="classification",
        references=["natural gas"],
    )
    assert score.schema_valid == 0.0
    assert score.answer_accuracy == 1.0


def test_missing_unit_is_semantically_scorable_but_not_schema_valid() -> None:
    score = score_structured_answer(
        '{"answer":"five"}',
        answer_type="numeric",
        references=["5", "five"],
        expected_value=5,
    )
    assert score.schema_valid == 0.0
    assert score.numeric_accuracy == 1.0
    assert score.answer_accuracy == 1.0
