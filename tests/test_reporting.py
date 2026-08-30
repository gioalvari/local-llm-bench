import json
from pathlib import Path

import pytest

from localllm_bench.reporting import generate_report


def test_generate_report(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps({"run_id": "test-run"}), encoding="utf-8"
    )
    observation = {
        "cell": {
            "cell_id": "abc",
            "workload_name": "short",
            "batch_size": 32,
            "ubatch_size": 16,
            "threads": 2,
            "gpu_layers": -1,
            "flash_attention": "on",
        },
        "metrics": {"n_prompt": 16, "n_gen": 0, "backends": "Metal", "avg_ts": 100.0},
        "peak_process_tree_rss_bytes": 1024**3,
    }
    (tmp_path / "measurements.jsonl").write_text(
        json.dumps(observation) + "\n", encoding="utf-8"
    )
    report = generate_report(tmp_path)
    content = report.read_text(encoding="utf-8")
    assert "test-run" in content
    assert "100.00" in content
    assert "1.000" in content


def test_generate_report_skips_dry_runs_and_handles_missing_metrics(
    tmp_path: Path,
) -> None:
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    observations = [
        {"dry_run": True},
        {"cell": {"cell_id": "missing"}, "metrics": {}},
    ]
    (tmp_path / "measurements.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in observations), encoding="utf-8"
    )
    report = generate_report(tmp_path)
    content = report.read_text(encoding="utf-8")
    assert "missing" in content
    assert "n/a" in content


def test_generate_report_requires_manifest(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="manifest"):
        generate_report(tmp_path)
