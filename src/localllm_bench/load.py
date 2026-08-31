"""Closed-loop concurrent load benchmarks for llama.cpp server."""

import concurrent.futures
import json
import math
import statistics
import subprocess
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError

import psutil
from pydantic import BaseModel

from localllm_bench.config import ExperimentSpec, ServerSpec
from localllm_bench.doctor import inspect_capabilities
from localllm_bench.model import validate_model
from localllm_bench.server import (
    available_port,
    build_server_command,
    complete_prompt,
    stop_server,
    wait_until_ready,
)
from localllm_bench.telemetry import ResourceMonitor


class LoadRunResult(BaseModel):
    """Location and aggregate counts for one concurrent load run."""

    run_id: str
    run_dir: Path
    completed_requests: int
    failed_requests: int
    summary: list[dict[str, float | int]]


def percentile(values: list[float], percentile_value: float) -> float:
    """Return a nearest-rank percentile for a non-empty sample."""
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0 < percentile_value <= 100:
        raise ValueError("percentile must be in the interval (0, 100]")
    ordered = sorted(values)
    index = max(0, math.ceil(percentile_value / 100 * len(ordered)) - 1)
    return ordered[index]


def summarize_level(
    concurrency: int,
    records: list[dict[str, Any]],
    duration_ns: int,
    resource_samples: list[dict[str, int]],
) -> dict[str, float | int]:
    """Aggregate one concurrency level without hiding failed requests."""
    completed = [record for record in records if "error" not in record]
    failed = len(records) - len(completed)
    duration_seconds = duration_ns / 1_000_000_000
    output_tokens = sum(
        int(record["output_tokens"])
        for record in completed
        if isinstance(record.get("output_tokens"), int)
    )
    ttft_ms = [int(record["ttft_ns"]) / 1_000_000 for record in completed]
    e2e_ms = [int(record["e2e_latency_ns"]) / 1_000_000 for record in completed]
    waves: dict[int, list[int]] = {}
    for record in records:
        if "wave_index" in record and "wave_started_ns" in record:
            waves.setdefault(int(record["wave_index"]), []).append(
                int(record["wave_started_ns"])
            )
    launch_spreads_ms = [
        (max(starts) - min(starts)) / 1_000_000 for starts in waves.values() if starts
    ]
    return {
        "concurrency": concurrency,
        "requests": len(records),
        "completed_requests": len(completed),
        "failed_requests": failed,
        "error_rate": failed / len(records) if records else 0.0,
        "duration_ns": duration_ns,
        "aggregate_output_tokens_per_second": (
            output_tokens / duration_seconds if duration_seconds > 0 else 0.0
        ),
        "requests_per_second": (
            len(completed) / duration_seconds if duration_seconds > 0 else 0.0
        ),
        "median_ttft_ms": statistics.median(ttft_ms) if ttft_ms else 0.0,
        "p95_ttft_ms": percentile(ttft_ms, 95) if ttft_ms else 0.0,
        "median_e2e_ms": statistics.median(e2e_ms) if e2e_ms else 0.0,
        "p95_e2e_ms": percentile(e2e_ms, 95) if e2e_ms else 0.0,
        "max_wave_launch_spread_ms": max(launch_spreads_ms, default=0.0),
        "peak_process_tree_rss_bytes": max(
            (sample["process_tree_rss_bytes"] for sample in resource_samples),
            default=0,
        ),
    }


def _run_wave(
    base_url: str,
    server: ServerSpec,
    concurrency: int,
    first_request_index: int,
) -> list[dict[str, Any]]:
    barrier = threading.Barrier(concurrency)

    def worker(request_index: int) -> dict[str, Any]:
        barrier.wait()
        wave_started_ns = time.monotonic_ns()
        result = complete_prompt(
            base_url, server, server.prompt, request_index=request_index
        )
        return {"wave_started_ns": wave_started_ns, **result}

    records: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(worker, first_request_index + offset): first_request_index
            + offset
            for offset in range(concurrency)
        }
        for future, request_index in futures.items():
            try:
                records.append(future.result())
            except (HTTPError, URLError, TimeoutError, ValueError) as error:
                records.append({"request_index": request_index, "error": str(error)})
    return sorted(records, key=lambda record: int(record["request_index"]))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run_load_benchmark(experiment: ExperimentSpec) -> LoadRunResult:
    """Run synchronized closed-loop request waves against a managed server."""
    if experiment.server is None:
        raise ValueError("experiment does not define a server section")
    if experiment.load is None:
        raise ValueError("experiment does not define a load section")
    validate_model(experiment.model)
    load = experiment.load
    max_concurrency = max(load.concurrency_levels)
    runtime_server = experiment.server.model_copy(
        update={
            "parallel": max_concurrency,
            "context_size": experiment.server.context_size * max_concurrency,
        }
    )
    port = runtime_server.port or available_port()
    run_id = (
        f"{experiment.experiment_id}-load-"
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    )
    run_dir = experiment.output_dir / run_id
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True)
    command = build_server_command(experiment, runtime_server, port)
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "run_type": "concurrency-load",
        "experiment": experiment.model_dump(mode="json"),
        "effective_server": runtime_server.model_dump(mode="json"),
        "context_tokens_per_slot": experiment.server.context_size,
        "command": command,
        "capabilities": inspect_capabilities(run_dir).model_dump(mode="json"),
    }
    manifest_path = run_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    records: list[dict[str, Any]] = []
    summaries: list[dict[str, float | int]] = []
    started_ns = time.monotonic_ns()
    with (
        (logs_dir / "server.stdout.log").open("w", encoding="utf-8") as stdout,
        (logs_dir / "server.stderr.log").open("w", encoding="utf-8") as stderr,
    ):
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, text=True)
        monitor = ResourceMonitor(
            psutil.Process(process.pid), experiment.sample_interval_ms / 1000
        )
        monitor.start()
        try:
            base_url = f"http://127.0.0.1:{port}"
            manifest["model_load_time_ns"] = wait_until_ready(
                base_url, process, runtime_server.startup_timeout_seconds
            )
            for warmup_index in range(load.warmup_requests):
                complete_prompt(
                    base_url,
                    runtime_server,
                    runtime_server.prompt,
                    request_index=-(warmup_index + 1),
                )
            manifest["warmup_completed"] = load.warmup_requests
            _write_json(manifest_path, manifest)
            request_index = 0
            for concurrency in load.concurrency_levels:
                level_started_offset_ns = time.monotonic_ns() - monitor.started_ns
                level_started_ns = time.monotonic_ns()
                level_records: list[dict[str, Any]] = []
                for wave_index in range(load.waves_per_level):
                    wave_records = _run_wave(
                        base_url,
                        runtime_server,
                        int(concurrency),
                        request_index,
                    )
                    for record in wave_records:
                        record["concurrency"] = int(concurrency)
                        record["wave_index"] = wave_index
                    level_records.extend(wave_records)
                    request_index += int(concurrency)
                    if experiment.fail_fast and any(
                        "error" in record for record in wave_records
                    ):
                        break
                level_duration_ns = time.monotonic_ns() - level_started_ns
                level_ended_offset_ns = time.monotonic_ns() - monitor.started_ns
                level_samples = [
                    sample
                    for sample in monitor.samples
                    if level_started_offset_ns
                    <= sample["monotonic_offset_ns"]
                    <= level_ended_offset_ns
                ]
                summaries.append(
                    summarize_level(
                        int(concurrency),
                        level_records,
                        level_duration_ns,
                        level_samples,
                    )
                )
                records.extend(level_records)
                if experiment.fail_fast and any(
                    "error" in record for record in level_records
                ):
                    break
        finally:
            stop_server(process)
            samples = monitor.stop()
    with (run_dir / "requests.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    with (run_dir / "resource_samples.jsonl").open("w", encoding="utf-8") as stream:
        for sample in samples:
            stream.write(json.dumps(sample, sort_keys=True) + "\n")
    _write_json(run_dir / "summary.json", summaries)
    manifest["summary"] = summaries
    manifest["process_wall_time_ns"] = time.monotonic_ns() - started_ns
    manifest["peak_process_tree_rss_bytes"] = max(
        (sample["process_tree_rss_bytes"] for sample in samples), default=0
    )
    _write_json(manifest_path, manifest)
    completed = sum("error" not in record for record in records)
    return LoadRunResult(
        run_id=run_id,
        run_dir=run_dir,
        completed_requests=completed,
        failed_requests=len(records) - completed,
        summary=summaries,
    )
