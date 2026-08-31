import json
from pathlib import Path

import pytest

from localllm_bench.dataset import dataset_sha256
from localllm_bench.rescore import rescore_run


def test_rescore_quality_run(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    item = {
        "item_id": "item-1",
        "source_document_id": "doc",
        "source_url": "https://example.org",
        "source_revision": "1",
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
    dataset.write_text(json.dumps(item) + "\n", encoding="utf-8")
    manifest = {
        "run_type": "quality-evaluation",
        "dataset_path": str(dataset),
        "dataset_sha256": dataset_sha256(dataset),
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    record = {
        "item_id": "item-1",
        "prompt_arm": "engineered",
        "raw_response": '{"answer":"1,000","unit":"W"}',
        "ttft_ns": 1_000_000,
        "e2e_latency_ns": 10_000_000,
        "score": {},
    }
    (tmp_path / "evaluations.jsonl").write_text(
        json.dumps(record) + "\n", encoding="utf-8"
    )
    summary = rescore_run(tmp_path)
    assert summary["engineered"]["answer_accuracy"] == 1.0
    rescored = json.loads((tmp_path / "evaluations.jsonl").read_text(encoding="utf-8"))
    assert rescored["answer_type"] == "numeric"


def test_rescore_rejects_other_run_types(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps({"run_type": "llama-server"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="not a quality"):
        rescore_run(tmp_path)


def test_rescore_rejects_changed_dataset(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    manifest = {
        "run_type": "quality-evaluation",
        "dataset_path": str(dataset),
        "dataset_sha256": "0" * 64,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        rescore_run(tmp_path)
