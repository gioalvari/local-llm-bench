"""Isolated llama.cpp microbenchmark execution."""

import json
import subprocess
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

import psutil
from pydantic import BaseModel

from localllm_bench.config import ExperimentSpec, ModelSpec
from localllm_bench.doctor import inspect_capabilities
from localllm_bench.planner import Plan, RunCell, expand_plan


class RunResult(BaseModel):
    """Location and completion counts for one experiment execution."""

    run_id: str
    run_dir: Path
    completed_cells: int
    failed_cells: int


def _model_args(model: ModelSpec) -> list[str]:
    if model.path is not None:
        arguments = ["--model", str(model.path)]
    else:
        arguments = ["--hf-repo", str(model.hf_repo)]
        if model.hf_file is not None:
            arguments.extend(["--hf-file", model.hf_file])
    if model.offline:
        arguments.append("--offline")
    return arguments


def build_command(spec: ExperimentSpec, cell: RunCell) -> list[str]:
    """Build a shell-free `llama-bench` command for one cell."""
    return [
        spec.llama_bench_binary,
        *_model_args(spec.model),
        "--n-prompt",
        str(cell.prompt_tokens),
        "--n-gen",
        str(cell.generation_tokens),
        "--batch-size",
        str(cell.batch_size),
        "--ubatch-size",
        str(cell.ubatch_size),
        "--threads",
        str(cell.threads),
        "--n-gpu-layers",
        str(cell.gpu_layers),
        "--flash-attn",
        cell.flash_attention.value,
        "--repetitions",
        str(cell.repetitions),
        "--output",
        "json",
    ]


def parse_llama_bench_output(value: str) -> list[dict[str, Any]]:
    """Parse and validate the JSON array emitted by `llama-bench`."""
    parsed = json.loads(value)
    if not isinstance(parsed, list) or not all(
        isinstance(item, dict) for item in parsed
    ):
        raise ValueError("llama-bench output must be a JSON array of objects")
    return parsed


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_jsonl(stream: TextIO, value: Any) -> None:
    stream.write(json.dumps(value, sort_keys=True) + "\n")
    stream.flush()


def _resource_sample(process: psutil.Process, started_ns: int) -> dict[str, int]:
    process_rss = 0
    process_tree_rss = 0
    try:
        process_rss = process.memory_info().rss
        process_tree_rss = process_rss + sum(
            child.memory_info().rss for child in process.children(recursive=True)
        )
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    host = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return {
        "monotonic_offset_ns": time.monotonic_ns() - started_ns,
        "process_rss_bytes": process_rss,
        "process_tree_rss_bytes": process_tree_rss,
        "host_available_bytes": host.available,
        "swap_used_bytes": swap.used,
    }


def _new_run_id(experiment_id: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{experiment_id}-{timestamp}-{uuid.uuid4().hex[:8]}"


def _validate_local_model(model: ModelSpec) -> None:
    if model.path is not None and not model.path.is_file():
        raise FileNotFoundError(f"model file does not exist: {model.path}")


def _prepare_run(spec: ExperimentSpec, plan: Plan) -> tuple[str, Path]:
    run_id = _new_run_id(spec.experiment_id)
    run_dir = spec.output_dir / run_id
    (run_dir / "logs").mkdir(parents=True)
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "experiment": spec.model_dump(mode="json"),
        "plan": plan.model_dump(mode="json"),
        "capabilities": inspect_capabilities(run_dir).model_dump(mode="json"),
    }
    _write_json(run_dir / "manifest.json", manifest)
    return run_id, run_dir


def run_experiment(spec: ExperimentSpec, *, dry_run: bool = False) -> RunResult:
    """Execute every valid matrix cell and persist raw observations.

    Parameters
    ----------
    spec
        Validated experiment configuration.
    dry_run
        Write the manifest and commands without invoking `llama-bench`.

    Returns
    -------
    RunResult
        Run location and cell completion counts.
    """
    _validate_local_model(spec.model)
    plan = expand_plan(spec)
    run_id, run_dir = _prepare_run(spec, plan)
    completed = 0
    failed = 0
    with (
        (run_dir / "measurements.jsonl").open("a", encoding="utf-8") as measurements,
        (run_dir / "resource_samples.jsonl").open("a", encoding="utf-8") as resources,
        (run_dir / "failures.jsonl").open("a", encoding="utf-8") as failures,
    ):
        for cell in plan.cells:
            command = build_command(spec, cell)
            if dry_run:
                _write_jsonl(
                    measurements,
                    {
                        "cell": cell.model_dump(mode="json"),
                        "command": command,
                        "dry_run": True,
                    },
                )
                completed += 1
                continue
            stdout_path = run_dir / "logs" / f"{cell.cell_id}.stdout.log"
            stderr_path = run_dir / "logs" / f"{cell.cell_id}.stderr.log"
            started_ns = time.monotonic_ns()
            with (
                stdout_path.open("w", encoding="utf-8") as stdout,
                stderr_path.open("w", encoding="utf-8") as stderr,
            ):
                process = subprocess.Popen(
                    command, stdout=stdout, stderr=stderr, text=True
                )
                tracked = psutil.Process(process.pid)
                samples: list[dict[str, int]] = []
                while process.poll() is None:
                    sample = _resource_sample(tracked, started_ns)
                    samples.append(sample)
                    _write_jsonl(resources, {"cell_id": cell.cell_id, **sample})
                    time.sleep(spec.sample_interval_ms / 1000)
                return_code = process.wait()
            elapsed_ns = time.monotonic_ns() - started_ns
            peak_rss = max(
                (sample["process_tree_rss_bytes"] for sample in samples), default=0
            )
            if return_code != 0:
                failed += 1
                _write_jsonl(
                    failures,
                    {
                        "cell_id": cell.cell_id,
                        "return_code": return_code,
                        "command": command,
                        "stderr_path": str(stderr_path.relative_to(run_dir)),
                    },
                )
                continue
            try:
                output = parse_llama_bench_output(
                    stdout_path.read_text(encoding="utf-8")
                )
            except (json.JSONDecodeError, ValueError) as error:
                failed += 1
                _write_jsonl(
                    failures,
                    {
                        "cell_id": cell.cell_id,
                        "return_code": return_code,
                        "error": str(error),
                        "stdout_path": str(stdout_path.relative_to(run_dir)),
                    },
                )
                continue
            for observation in output:
                _write_jsonl(
                    measurements,
                    {
                        "cell": cell.model_dump(mode="json"),
                        "metrics": observation,
                        "process_wall_time_ns": elapsed_ns,
                        "peak_process_tree_rss_bytes": peak_rss,
                    },
                )
            completed += 1
    return RunResult(
        run_id=run_id,
        run_dir=run_dir,
        completed_cells=completed,
        failed_cells=failed,
    )
