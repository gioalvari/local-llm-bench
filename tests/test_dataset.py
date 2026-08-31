import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from localllm_bench.dataset import EvaluationItem, dataset_sha256, load_dataset


def _item(item_id: str = "item-1") -> dict[str, object]:
    return {
        "item_id": item_id,
        "source_document_id": "document",
        "source_url": "https://example.org/source",
        "source_revision": "2026-01-01",
        "category": "units",
        "context": "One kW is 1000 W.",
        "question": "How many W?",
        "answer_type": "numeric",
        "canonical_answer": "1000",
        "expected_value": 1000,
        "accepted_units": ["W"],
        "evidence_spans": ["1000 W"],
        "split": "smoke",
    }


def test_load_dataset_and_digest(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps(_item()) + "\n", encoding="utf-8")
    assert load_dataset(path)[0].item_id == "item-1"
    assert len(dataset_sha256(path)) == 64


def test_dataset_rejects_duplicates_and_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text(
        json.dumps(_item()) + "\n" + json.dumps(_item()) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_dataset(path)
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_dataset(path)


def test_numeric_item_requires_expected_value() -> None:
    item = _item()
    item["expected_value"] = None
    with pytest.raises(ValidationError, match="expected_value"):
        EvaluationItem.model_validate(item)
