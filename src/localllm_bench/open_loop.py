"""Deterministic fixed-rate open-loop serving benchmarks."""

import concurrent.futures
import hashlib
import json
import math
import statistics
import subprocess
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError

import psutil
from pydantic import BaseModel, Field

from localllm_bench.config import ExperimentSpec, ServerSpec
from localllm_bench.doctor import inspect_capabilities
from localllm_bench.load import percentile
from localllm_bench.model import validate_model
from localllm_bench.server import (
    available_port,
    build_server_command,
    complete_prompt,
    stop_server,
    wait_until_ready,
)
from localllm_bench.telemetry import ResourceMonitor


class LoadPrompt(BaseModel):
    """One deterministic open-loop prompt."""

    prompt_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    prompt: str = Field(min_length=1)


class OpenLoopRunResult(BaseModel):
    """Location and aggregate counts for an open-loop run."""

    run_id: str
    run_dir: Path
    completed_requests: int
    failed_requests: int
    summary: list[dict[str, float | int]]


def load_prompts(path: Path) -> list[LoadPrompt]:
    """Load prompt records and reject empty or duplicate corpora."""
    prompts = [
        LoadPrompt.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not prompts:
        raise ValueError("prompt dataset is empty")
    identifiers = [prompt.prompt_id for prompt in prompts]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("prompt dataset contains duplicate prompt_id values")
    return prompts


def prompt_dataset_sha256(path: Path) -> str:
    """Return the digest of the exact prompt dataset bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def request_count(rate_rps: float, duration_seconds: float) -> int:
    """Return the deterministic number of arrivals for a rate window."""
    return max(1, math.ceil(rate_rps * duration_seconds))


def _max_in_flight(records: list[dict[str, Any]]) -> int:
    events: list[tuple[int, int]] = []
    for record in records:
        if "client_started_offset_ns" not in record:
            continue
        events.append((int(record["client_started_offset_ns"]), 1))
        events.append((int(record["client_completed_offset_ns"]), -1))
    current = 0
    maximum = 0
    for _, delta in sorted(events, key=lambda event: (event[0], -event[1])):
        current += delta
        maximum = max(maximum, current)
    return maximum


def summarize_rate(
    rate_rps: float,
    duration_ns: int,
    records: list[dict[str, Any]],
    resource_samples: list[dict[str, int]],
    latency_slo_ms: float,
) -> dict[str, float | int]:
    """Aggregate one fixed-rate window, including scheduler and SLO metrics."""
    completed = [record for record in records if "error" not in record]
    failed = len(records) - len(completed)
    duration_seconds = duration_ns / 1_000_000_000
    ttft_ms = [int(record["ttft_ns"]) / 1_000_000 for record in completed]
    e2e_ms = [int(record["e2e_latency_ns"]) / 1_000_000 for record in completed]
    schedule_delay_ms = [
        int(record["client_schedule_delay_ns"]) / 1_000_000 for record in records
    ]
    output_tokens = sum(
        int(record["output_tokens"])
        for record in completed
        if isinstance(record.get("output_tokens"), int)
    )
    good = sum(value <= latency_slo_ms for value in e2e_ms)
    return {
        "offered_requests_per_second": rate_rps,
        "requests": len(records),
        "completed_requests": len(completed),
        "failed_requests": failed,
        "error_rate": failed / len(records) if records else 0.0,
        "duration_ns": duration_ns,
        "achieved_requests_per_second": (
            len(completed) / duration_seconds if duration_seconds > 0 else 0.0
        ),
        "aggregate_output_tokens_per_second": (
            output_tokens / duration_seconds if duration_seconds > 0 else 0.0
        ),
        "goodput_requests_per_second": (
            good / duration_seconds if duration_seconds > 0 else 0.0
        ),
        "slo_attainment_rate": good / len(records) if records else 0.0,
        "median_ttft_ms": statistics.median(ttft_ms) if ttft_ms else 0.0,
        "p95_ttft_ms": percentile(ttft_ms, 95) if ttft_ms else 0.0,
        "median_e2e_ms": statistics.median(e2e_ms) if e2e_ms else 0.0,
        "p95_e2e_ms": percentile(e2e_ms, 95) if e2e_ms else 0.0,
        "p95_client_schedule_delay_ms": (
            percentile(schedule_delay_ms, 95) if schedule_delay_ms else 0.0
        ),
        "max_client_in_flight": _max_in_flight(records),
        "peak_process_tree_rss_bytes": max(
            (sample["process_tree_rss_bytes"] for sample in resource_samples),
            default=0,
        ),
    }


def _run_request(
    base_url: str,
    server: ServerSpec,
    prompt: LoadPrompt,
    request_index: int,
    level_started_ns: int,
    scheduled_ns: int,
) -> dict[str, Any]:
    client_started_ns = time.monotonic_ns()
    metadata: dict[str, Any] = {
        "request_index": request_index,
        "prompt_id": prompt.prompt_id,
        "scheduled_offset_ns": scheduled_ns - level_started_ns,
        "client_started_offset_ns": client_started_ns - level_started_ns,
        "client_schedule_delay_ns": max(0, client_started_ns - scheduled_ns),
    }
    try:
        measurement = complete_prompt(
            base_url,
            server,
            prompt.prompt,
            request_index=request_index,
        )
        return {
            **metadata,
            **measurement,
            "client_completed_offset_ns": time.monotonic_ns() - level_started_ns,
        }
    except (HTTPError, URLError, TimeoutError, ValueError) as error:
        return {
            **metadata,
            "client_completed_offset_ns": time.monotonic_ns() - level_started_ns,
            "error": str(error),
        }


def run_rate_window(
    base_url: str,
    server: ServerSpec,
    prompts: list[LoadPrompt],
    rate_rps: float,
    duration_seconds: float,
    max_client_workers: int,
    first_request_index: int,
) -> tuple[list[dict[str, Any]], int]:
    """Schedule fixed-spacing arrivals independently of request completion."""
    count = request_count(rate_rps, duration_seconds)
    level_started_ns = time.monotonic_ns()
    futures: list[concurrent.futures.Future[dict[str, Any]]] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_client_workers
    ) as executor:
        for offset in range(count):
            scheduled_ns = level_started_ns + int(offset * 1_000_000_000 / rate_rps)
            remaining_ns = scheduled_ns - time.monotonic_ns()
            if remaining_ns > 0:
                time.sleep(remaining_ns / 1_000_000_000)
            futures.append(
                executor.submit(
                    _run_request,
                    base_url,
                    server,
                    prompts[offset % len(prompts)],
                    first_request_index + offset,
                    level_started_ns,
                    scheduled_ns,
                )
            )
        records = [future.result() for future in futures]
    configured_end_ns = level_started_ns + int(duration_seconds * 1_000_000_000)
    remaining_ns = configured_end_ns - time.monotonic_ns()
    if remaining_ns > 0:
        time.sleep(remaining_ns / 1_000_000_000)
    duration_ns = time.monotonic_ns() - level_started_ns
    return sorted(records, key=lambda record: int(record["request_index"])), duration_ns


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run_open_loop_benchmark(experiment: ExperimentSpec) -> OpenLoopRunResult:
    """Run deterministic fixed-rate arrivals against a managed llama-server."""
    if experiment.server is None:
        raise ValueError("experiment does not define a server section")
    if experiment.open_loop is None:
        raise ValueError("experiment does not define an open_loop section")
    validate_model(experiment.model)
    load = experiment.open_loop
    prompts = load_prompts(load.prompt_dataset)
    runtime_server = experiment.server.model_copy(
        update={
            "parallel": load.server_slots,
            "context_size": experiment.server.context_size * load.server_slots,
        }
    )
    port = runtime_server.port or available_port()
    run_id = (
        f"{experiment.experiment_id}-open-loop-"
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    )
    run_dir = experiment.output_dir / run_id
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True)
    command = build_server_command(experiment, runtime_server, port)
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "run_type": "open-loop-load",
        "experiment": experiment.model_dump(mode="json"),
        "effective_server": runtime_server.model_dump(mode="json"),
        "context_tokens_per_slot": experiment.server.context_size,
        "prompt_dataset_sha256": prompt_dataset_sha256(load.prompt_dataset),
        "prompt_count": len(prompts),
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
                prompt = prompts[warmup_index % len(prompts)]
                complete_prompt(
                    base_url,
                    runtime_server,
                    prompt.prompt,
                    request_index=-(warmup_index + 1),
                )
            manifest["warmup_completed"] = load.warmup_requests
            _write_json(manifest_path, manifest)
            request_index = 0
            for rate in load.arrival_rates_rps:
                level_started_offset_ns = time.monotonic_ns() - monitor.started_ns
                level_records, duration_ns = run_rate_window(
                    base_url,
                    runtime_server,
                    prompts,
                    float(rate),
                    float(load.duration_seconds),
                    int(load.max_client_workers),
                    request_index,
                )
                level_ended_offset_ns = time.monotonic_ns() - monitor.started_ns
                for record in level_records:
                    record["offered_requests_per_second"] = float(rate)
                level_samples = [
                    sample
                    for sample in monitor.samples
                    if level_started_offset_ns
                    <= sample["monotonic_offset_ns"]
                    <= level_ended_offset_ns
                ]
                summaries.append(
                    summarize_rate(
                        float(rate),
                        duration_ns,
                        level_records,
                        level_samples,
                        float(load.latency_slo_ms),
                    )
                )
                records.extend(level_records)
                request_index += len(level_records)
                if experiment.fail_fast and any(
                    "error" in record for record in level_records
                ):
                    break
                if load.cooldown_seconds > 0:
                    time.sleep(float(load.cooldown_seconds))
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
    return OpenLoopRunResult(
        run_id=run_id,
        run_dir=run_dir,
        completed_requests=completed,
        failed_requests=len(records) - completed,
        summary=summaries,
    )
