"""Source-grounded quality benchmark records."""

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, model_validator


class EvaluationItem(BaseModel):
    """One provenance-preserving, objectively scored question."""

    item_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    source_document_id: str = Field(min_length=1)
    source_url: HttpUrl
    source_revision: str = Field(min_length=1)
    category: str = Field(min_length=1)
    context: str = Field(min_length=1)
    question: str = Field(min_length=1)
    answer_type: Literal["text", "classification", "numeric"]
    canonical_answer: str = Field(min_length=1)
    accepted_aliases: list[str] = Field(default_factory=list)
    expected_value: float | None = None
    accepted_units: list[str] = Field(default_factory=list)
    absolute_tolerance: float = Field(default=0.0, ge=0.0)
    relative_tolerance: float = Field(default=0.0, ge=0.0)
    evidence_spans: list[str] = Field(min_length=1)
    split: Literal["smoke", "train", "validation", "test"]

    @model_validator(mode="after")
    def validate_numeric_answer(self) -> "EvaluationItem":
        """Require an expected number for numeric questions."""
        if self.answer_type == "numeric" and self.expected_value is None:
            raise ValueError("numeric items require expected_value")
        return self

    @property
    def references(self) -> list[str]:
        """Return canonical and accepted textual answers."""
        return [self.canonical_answer, *self.accepted_aliases]


def load_dataset(path: Path) -> list[EvaluationItem]:
    """Load JSONL records and reject duplicate item identifiers."""
    items = [
        EvaluationItem.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    identifiers = [item.item_id for item in items]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("dataset contains duplicate item_id values")
    if not items:
        raise ValueError("dataset is empty")
    return items


def dataset_sha256(path: Path) -> str:
    """Return the digest of the exact dataset bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def item_as_dict(item: EvaluationItem) -> dict[str, object]:
    """Serialize an item without URL implementation details."""
    return json.loads(item.model_dump_json())
