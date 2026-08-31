import json
from pathlib import Path

import pytest

from localllm_bench.training_data import (
    TrainingItem,
    prepare_training_dataset,
    validate_document_splits,
)


def _items() -> list[TrainingItem]:
    return [
        TrainingItem(
            item_id="train-1",
            source_document_id="train-doc",
            split="train",
            question="Training question?",
            answer="Training answer.",
        ),
        TrainingItem(
            item_id="valid-1",
            source_document_id="valid-doc",
            split="valid",
            question="Validation question?",
            answer="Validation answer.",
        ),
        TrainingItem(
            item_id="test-1",
            source_document_id="test-doc",
            split="test",
            question="Test question?",
            answer="Test answer.",
        ),
    ]


def test_validate_document_disjoint_items() -> None:
    validate_document_splits(_items())
    leaking = _items()
    leaking[-1] = leaking[-1].model_copy(update={"source_document_id": "train-doc"})
    with pytest.raises(ValueError, match="documents occur"):
        validate_document_splits(leaking)


def test_validate_rejects_answer_leakage_and_missing_splits() -> None:
    leaking = _items()
    leaking[-1] = leaking[-1].model_copy(update={"answer": "Training answer"})
    with pytest.raises(ValueError, match="answers leak"):
        validate_document_splits(leaking)
    with pytest.raises(ValueError, match="missing required"):
        validate_document_splits(_items()[:2])


def test_prepare_training_dataset(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(
        "".join(item.model_dump_json() + "\n" for item in _items()), encoding="utf-8"
    )
    result = prepare_training_dataset(source, tmp_path / "prepared")
    assert result.split_counts == {"train": 1, "valid": 1, "test": 1}
    assert result.document_counts == {"train": 1, "valid": 1, "test": 1}
    train_record = json.loads(
        (result.output_dir / "train.jsonl").read_text(encoding="utf-8")
    )
    assert [message["role"] for message in train_record["messages"]] == [
        "system",
        "user",
        "assistant",
    ]
    assert len(result.source_sha256) == 64
