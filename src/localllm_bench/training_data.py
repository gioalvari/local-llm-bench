"""Leakage-safe preparation of MLX chat fine-tuning datasets."""

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from localllm_bench.quality import normalize_answer


class TrainingItem(BaseModel):
    """One supervised energy-market question and answer."""

    item_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    source_document_id: str = Field(min_length=1)
    split: Literal["train", "valid", "test"]
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)


class PreparedDataset(BaseModel):
    """Paths and provenance for one prepared MLX dataset."""

    output_dir: Path
    source_sha256: str
    split_counts: dict[str, int]
    document_counts: dict[str, int]


def load_training_items(path: Path) -> list[TrainingItem]:
    """Load source records and validate item-level uniqueness."""
    items = [
        TrainingItem.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not items:
        raise ValueError("training dataset is empty")
    identifiers = [item.item_id for item in items]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("training dataset contains duplicate item_id values")
    questions = [normalize_answer(item.question) for item in items]
    if len(questions) != len(set(questions)):
        raise ValueError("training dataset contains duplicate normalized questions")
    return items


def validate_document_splits(items: list[TrainingItem]) -> None:
    """Reject document and answer leakage across dataset splits."""
    document_splits: dict[str, set[str]] = {}
    for item in items:
        document_splits.setdefault(item.source_document_id, set()).add(item.split)
    leaking_documents = [
        document for document, splits in document_splits.items() if len(splits) > 1
    ]
    if leaking_documents:
        raise ValueError("source documents occur in multiple splits")
    answers_by_split: dict[str, set[str]] = {
        "train": set(),
        "valid": set(),
        "test": set(),
    }
    for item in items:
        answers_by_split[item.split].add(normalize_answer(item.answer))
    if answers_by_split["train"] & (
        answers_by_split["valid"] | answers_by_split["test"]
    ):
        raise ValueError("normalized training answers leak into evaluation splits")
    if answers_by_split["valid"] & answers_by_split["test"]:
        raise ValueError("normalized validation answers leak into the test split")
    missing = [split for split, answers in answers_by_split.items() if not answers]
    if missing:
        raise ValueError(f"training dataset is missing required splits: {missing}")


def _chat_record(item: TrainingItem) -> dict[str, object]:
    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "Answer the energy-market question accurately and concisely."
                ),
            },
            {"role": "user", "content": item.question},
            {"role": "assistant", "content": item.answer},
        ]
    }


def prepare_training_dataset(source: Path, output_dir: Path) -> PreparedDataset:
    """Write MLX-compatible document-disjoint chat splits."""
    items = load_training_items(source)
    validate_document_splits(items)
    output_dir.mkdir(parents=True, exist_ok=False)
    split_names = {"train": "train.jsonl", "valid": "valid.jsonl", "test": "test.jsonl"}
    for split, filename in split_names.items():
        records = [_chat_record(item) for item in items if item.split == split]
        (output_dir / filename).write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
            encoding="utf-8",
        )
    split_counts = {
        split: sum(item.split == split for item in items) for split in split_names
    }
    document_counts = {
        split: len({item.source_document_id for item in items if item.split == split})
        for split in split_names
    }
    manifest = PreparedDataset(
        output_dir=output_dir,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        split_counts=split_counts,
        document_counts=document_counts,
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
