"""Offline rescoring for persisted quality-evaluation records."""

import json
from pathlib import Path
from typing import Any

from localllm_bench.dataset import dataset_sha256, load_dataset
from localllm_bench.evaluation import aggregate_records
from localllm_bench.quality import score_structured_answer


def rescore_run(run_dir: Path) -> dict[str, dict[str, float | int]]:
    """Recompute quality metrics without repeating model inference."""
    manifest_path = run_dir / "manifest.json"
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("run_type") != "quality-evaluation":
        raise ValueError("run is not a quality evaluation")
    dataset_path = Path(str(manifest["dataset_path"]))
    current_digest = dataset_sha256(dataset_path)
    if current_digest != manifest.get("dataset_sha256"):
        raise ValueError("dataset SHA-256 does not match the run manifest")
    items = {item.item_id: item for item in load_dataset(dataset_path)}
    records = [
        json.loads(line)
        for line in (run_dir / "evaluations.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    for record in records:
        item = items[str(record["item_id"])]
        record["answer_type"] = item.answer_type
        record["score"] = score_structured_answer(
            str(record["raw_response"]),
            answer_type=item.answer_type,
            references=item.references,
            expected_value=item.expected_value,
            absolute_tolerance=item.absolute_tolerance,
            relative_tolerance=item.relative_tolerance,
            accepted_units=item.accepted_units,
        ).model_dump(mode="json")
    (run_dir / "evaluations.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    summary = aggregate_records(records)
    manifest["summary"] = summary
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
