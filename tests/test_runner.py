import json
from pathlib import Path

import pytest

from localllm_bench.config import ExperimentSpec
from localllm_bench.planner import expand_plan
from localllm_bench.runner import (
    build_command,
    parse_llama_bench_output,
    run_experiment,
)


def _spec() -> ExperimentSpec:
    return ExperimentSpec.model_validate(
        {
            "experiment_id": "test",
            "model": {
                "name": "tiny",
                "hf_repo": "owner/model:Q4_K_M",
                "quantization": "Q4_K_M",
                "offline": True,
            },
            "matrix": {
                "repetitions": 3,
                "workloads": [
                    {"name": "short", "prompt_tokens": 16, "generation_tokens": 8}
                ],
                "batch_sizes": [32],
                "ubatch_sizes": [16],
                "threads": [2],
                "gpu_layers": [-1],
                "flash_attention": ["on"],
            },
        }
    )


def test_build_command_uses_explicit_settings() -> None:
    spec = _spec()
    command = build_command(spec, expand_plan(spec).cells[0])
    assert command[:4] == [
        "llama-bench",
        "--hf-repo",
        "owner/model:Q4_K_M",
        "--offline",
    ]
    assert command[-2:] == ["--output", "json"]
    assert command[command.index("--threads") + 1] == "2"


def test_parse_llama_bench_output() -> None:
    assert parse_llama_bench_output(json.dumps([{"avg_ts": 12.5}])) == [
        {"avg_ts": 12.5}
    ]
    with pytest.raises(ValueError, match="array"):
        parse_llama_bench_output(json.dumps({"avg_ts": 12.5}))


def test_dry_run_writes_manifest_and_commands(tmp_path: Path) -> None:
    spec = _spec().model_copy(update={"output_dir": tmp_path})
    result = run_experiment(spec, dry_run=True)
    assert result.completed_cells == 1
    assert result.failed_cells == 0
    assert (result.run_dir / "manifest.json").is_file()
    measurement = json.loads(
        (result.run_dir / "measurements.jsonl").read_text(encoding="utf-8")
    )
    assert measurement["dry_run"] is True


def _write_fake_benchmark(path: Path, body: str) -> None:
    path.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    path.chmod(0o755)


def test_run_experiment_records_success(tmp_path: Path) -> None:
    executable = tmp_path / "fake-bench"
    _write_fake_benchmark(
        executable,
        'sleep 0.05; printf \'[{"avg_ts": 12.5, "n_prompt": 16}]\'',
    )
    spec = _spec().model_copy(
        update={"output_dir": tmp_path / "runs", "llama_bench_binary": str(executable)}
    )
    result = run_experiment(spec)
    assert result.completed_cells == 1
    assert result.failed_cells == 0
    measurement = json.loads(
        (result.run_dir / "measurements.jsonl").read_text(encoding="utf-8")
    )
    assert measurement["metrics"]["avg_ts"] == 12.5
    assert measurement["process_wall_time_ns"] > 0


@pytest.mark.parametrize(
    ("body", "expected_key"),
    [("exit 2", "return_code"), ("printf 'not-json'", "error")],
)
def test_run_experiment_records_failures(
    tmp_path: Path, body: str, expected_key: str
) -> None:
    executable = tmp_path / "fake-bench"
    _write_fake_benchmark(executable, body)
    spec = _spec().model_copy(
        update={"output_dir": tmp_path / "runs", "llama_bench_binary": str(executable)}
    )
    result = run_experiment(spec)
    assert result.completed_cells == 0
    assert result.failed_cells == 1
    failure = json.loads(
        (result.run_dir / "failures.jsonl").read_text(encoding="utf-8")
    )
    assert expected_key in failure


def test_run_experiment_rejects_missing_local_model(tmp_path: Path) -> None:
    spec = _spec().model_copy(
        update={
            "output_dir": tmp_path,
            "model": _spec().model.model_copy(
                update={"path": tmp_path / "missing.gguf", "hf_repo": None}
            ),
        }
    )
    with pytest.raises(FileNotFoundError, match="model file"):
        run_experiment(spec, dry_run=True)
