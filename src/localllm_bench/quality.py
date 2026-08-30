"""Judge-independent answer quality metrics."""

import re
import string
from collections import Counter

_ARTICLES = {"a", "an", "the"}
_NUMBER = re.compile(r"[-+]?\d+(?:[.,]\d+)?")


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
    observed = float(match.group().replace(",", "."))
    allowed = max(absolute_tolerance, abs(expected) * relative_tolerance)
    return float(abs(observed - expected) <= allowed)
