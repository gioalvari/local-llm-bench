"""Judge-independent answer quality metrics."""

import json
import re
import string
from collections import Counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

_ARTICLES = {"a", "an", "the"}
_NUMBER = re.compile(r"[-+]?\d+(?:[.,]\d+)*")
_FENCED_JSON = re.compile(r"^```(?:json)?\s*(\{.*\})\s*```$", re.DOTALL)


class StructuredAnswer(BaseModel):
    """Machine-readable answer contract requested from evaluated models."""

    model_config = ConfigDict(extra="forbid")

    answer: str
    unit: str | None


class ScorableAnswer(BaseModel):
    """Lenient content contract used only for semantic scoring."""

    model_config = ConfigDict(extra="forbid")

    answer: str
    unit: str | None = None


class ItemScore(BaseModel):
    """Judge-independent metrics for one benchmark answer."""

    schema_valid: float
    answer_accuracy: float
    exact_match: float
    token_f1: float
    numeric_accuracy: float | None = None
    unit_accuracy: float | None = None
    parsed_answer: str | None = None
    parsed_unit: str | None = None


def normalize_answer(value: str) -> str:
    """Normalize a short answer for deterministic comparison.

    Parameters
    ----------
    value
        Raw model or reference answer.

    Returns
    -------
    str
        Lowercase text without punctuation, articles, or repeated whitespace.
    """
    table = str.maketrans("", "", string.punctuation)
    tokens = value.lower().translate(table).split()
    return " ".join(token for token in tokens if token not in _ARTICLES)


def normalized_exact_match(prediction: str, references: list[str]) -> float:
    """Return one when the normalized prediction matches any reference."""
    normalized_prediction = normalize_answer(prediction)
    return float(
        any(normalized_prediction == normalize_answer(item) for item in references)
    )


def token_f1(prediction: str, references: list[str]) -> float:
    """Return the best bag-of-tokens F1 score over accepted references."""
    prediction_tokens = normalize_answer(prediction).split()
    if not references:
        return 0.0
    scores: list[float] = []
    for reference in references:
        reference_tokens = normalize_answer(reference).split()
        common = Counter(prediction_tokens) & Counter(reference_tokens)
        matches = sum(common.values())
        if not prediction_tokens or not reference_tokens:
            scores.append(float(prediction_tokens == reference_tokens))
            continue
        precision = matches / len(prediction_tokens)
        recall = matches / len(reference_tokens)
        scores.append(
            0.0
            if precision + recall == 0
            else 2 * precision * recall / (precision + recall)
        )
    return max(scores)


def numeric_match(
    prediction: str,
    expected: float,
    *,
    absolute_tolerance: float = 0.0,
    relative_tolerance: float = 0.0,
) -> float:
    """Compare the first numeric value in an answer with fixed tolerances."""
    match = _NUMBER.search(prediction)
    if match is None:
        return 0.0
    raw_number = match.group()
    if "," in raw_number and "." in raw_number:
        if raw_number.rfind(",") > raw_number.rfind("."):
            normalized_number = raw_number.replace(".", "").replace(",", ".")
        else:
            normalized_number = raw_number.replace(",", "")
    elif "," in raw_number:
        parts = raw_number.split(",")
        normalized_number = (
            "".join(parts)
            if len(parts) > 2 or all(len(part) == 3 for part in parts[1:])
            else raw_number.replace(",", ".")
        )
    else:
        normalized_number = raw_number
    observed = float(normalized_number)
    allowed = max(absolute_tolerance, abs(expected) * relative_tolerance)
    return float(abs(observed - expected) <= allowed)


def parse_structured_answer(value: str) -> StructuredAnswer | None:
    """Parse a response only when the complete string follows the contract."""
    try:
        payload = json.loads(value.strip())
        return StructuredAnswer.model_validate(payload)
    except (json.JSONDecodeError, ValidationError):
        return None


def _parse_scorable_answer(value: str) -> ScorableAnswer | None:
    stripped = value.strip()
    match = _FENCED_JSON.fullmatch(stripped)
    payload = match.group(1) if match is not None else stripped
    try:
        return ScorableAnswer.model_validate_json(payload)
    except ValidationError:
        return None


def score_structured_answer(
    response: str,
    *,
    answer_type: Literal["text", "classification", "numeric"],
    references: list[str],
    expected_value: float | None = None,
    absolute_tolerance: float = 0.0,
    relative_tolerance: float = 0.0,
    accepted_units: list[str] | None = None,
) -> ItemScore:
    """Score one structured response without an LLM judge."""
    strict_answer = parse_structured_answer(response)
    schema_valid = float(strict_answer is not None)
    scorable_answer = strict_answer or _parse_scorable_answer(response)
    if scorable_answer is None:
        return ItemScore(
            schema_valid=0.0,
            answer_accuracy=0.0,
            exact_match=0.0,
            token_f1=0.0,
        )
    exact = normalized_exact_match(scorable_answer.answer, references)
    f1 = token_f1(scorable_answer.answer, references)
    if answer_type != "numeric":
        return ItemScore(
            schema_valid=schema_valid,
            answer_accuracy=exact,
            exact_match=exact,
            token_f1=f1,
            parsed_answer=scorable_answer.answer,
            parsed_unit=scorable_answer.unit,
        )
    numeric_value_match = (
        numeric_match(
            scorable_answer.answer,
            expected_value,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )
        if expected_value is not None
        else 0.0
    )
    numeric = max(numeric_value_match, exact)
    unit = (
        normalized_exact_match(scorable_answer.unit or "", accepted_units)
        if accepted_units
        else 1.0
    )
    return ItemScore(
        schema_valid=schema_valid,
        answer_accuracy=numeric * unit,
        exact_match=exact,
        token_f1=f1,
        numeric_accuracy=numeric,
        unit_accuracy=unit,
        parsed_answer=scorable_answer.answer,
        parsed_unit=scorable_answer.unit,
    )
