"""Reproducible open-loop serving benchmarks."""

import concurrent.futures
import hashlib
import json
import math
import random
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

from localllm_bench.config import (
    MAX_ARRIVALS_PER_WINDOW,
    ArrivalProcess,
    ExperimentSpec,
    OpenLoopSpec,
    ServerSpec,
)
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

OPEN_LOOP_SCHEDULER_VERSION = "precomputed-monotonic-offsets-v1"
OPEN_LOOP_REQUEST_PROTOCOL_VERSION = "llama-completion-stream-greedy-v1"
OPEN_LOOP_SUMMARY_VERSION = "open-loop-summary-v2"


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
    summary: list[dict[str, float | int | None]]


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


def _arrival_stream_sha256(rate_rps: float, arrival_seed: int) -> str:
    material = f"open-loop-poisson-v1\0{arrival_seed}\0{float(rate_rps).hex()}".encode(
        "ascii"
    )
    return hashlib.sha256(material).hexdigest()


def arrival_offsets_ns(
    rate_rps: float,
    duration_seconds: float,
    arrival_process: ArrivalProcess,
    arrival_seed: int,
) -> list[int]:
    """Generate exact arrival offsets for one configured rate window."""
    if (
        not math.isfinite(rate_rps)
        or not math.isfinite(duration_seconds)
        or rate_rps <= 0
        or duration_seconds <= 0
    ):
        raise ValueError("arrival rate and duration must be finite and positive")
    if rate_rps * duration_seconds > MAX_ARRIVALS_PER_WINDOW:
        raise ValueError(
            f"arrival window exceeds {MAX_ARRIVALS_PER_WINDOW} expected arrivals"
        )
    if arrival_process is ArrivalProcess.FIXED:
        return [
            int(offset * 1_000_000_000 / rate_rps)
            for offset in range(request_count(rate_rps, duration_seconds))
        ]
    stream_digest = _arrival_stream_sha256(rate_rps, arrival_seed)
    generator = random.Random(int(stream_digest, 16))
    offsets: list[int] = []
    elapsed_seconds = 0.0
    while True:
        uniform = generator.random()
        elapsed_seconds += -math.log1p(-uniform) / rate_rps
        if elapsed_seconds >= duration_seconds:
            break
        if len(offsets) >= MAX_ARRIVALS_PER_WINDOW:
            raise ValueError(
                f"arrival realization exceeds {MAX_ARRIVALS_PER_WINDOW} arrivals"
            )
        offsets.append(int(elapsed_seconds * 1_000_000_000))
    return offsets


def build_arrival_schedule(load: OpenLoopSpec) -> dict[str, Any]:
    """Build the complete replayable arrival schedule before measurement."""
    process = load.arrival_process
    windows: list[dict[str, Any]] = []
    for rate_value in load.arrival_rates_rps:
        rate_rps = float(rate_value)
        windows.append(
            {
                "offered_requests_per_second": rate_rps,
                "arrival_window_ns": int(float(load.duration_seconds) * 1_000_000_000),
                "stream_sha256": (
                    _arrival_stream_sha256(rate_rps, int(load.arrival_seed))
                    if process is ArrivalProcess.POISSON
                    else None
                ),
                "scheduled_offsets_ns": arrival_offsets_ns(
                    rate_rps,
                    float(load.duration_seconds),
                    process,
                    int(load.arrival_seed),
                ),
            }
        )
    return {
        "schema_version": "1",
        "arrival_process": process.value,
        "algorithm": (
            "poisson-exponential-v1"
            if process is ArrivalProcess.POISSON
            else "fixed-spacing-v1"
        ),
        "arrival_seed": int(load.arrival_seed),
        "windows": windows,
    }


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
    arrival_window_ns: int | None = None,
) -> dict[str, float | int | None]:
    """Aggregate one arrival window, including scheduler and SLO metrics."""
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
    configured_window_ns = arrival_window_ns or duration_ns
    configured_window_seconds = configured_window_ns / 1_000_000_000
    return {
        "offered_requests_per_second": rate_rps,
        "realized_offered_requests_per_second": (
            len(records) / configured_window_seconds
            if configured_window_seconds > 0
            else 0.0
        ),
        "requests": len(records),
        "completed_requests": len(completed),
        "failed_requests": failed,
        "error_rate": failed / len(records) if records else None,
        "arrival_window_ns": configured_window_ns,
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
        "slo_attainment_rate": good / len(records) if records else None,
        "median_ttft_ms": statistics.median(ttft_ms) if ttft_ms else None,
        "p95_ttft_ms": percentile(ttft_ms, 95) if ttft_ms else None,
        "median_e2e_ms": statistics.median(e2e_ms) if e2e_ms else None,
        "p95_e2e_ms": percentile(e2e_ms, 95) if e2e_ms else None,
        "p95_client_schedule_delay_ms": (
            percentile(schedule_delay_ms, 95) if schedule_delay_ms else None
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
    rate_request_index: int,
    level_started_ns: int,
    scheduled_ns: int,
) -> dict[str, Any]:
    client_started_ns = time.monotonic_ns()
    metadata: dict[str, Any] = {
        "request_index": request_index,
        "rate_request_index": rate_request_index,
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
    scheduled_offsets_ns: list[int],
    duration_seconds: float,
    max_client_workers: int,
    first_request_index: int,
) -> tuple[list[dict[str, Any]], int]:
    """Execute precomputed arrivals independently of request completion."""
    level_started_ns = time.monotonic_ns()
    futures: list[concurrent.futures.Future[dict[str, Any]]] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_client_workers
    ) as executor:
        for rate_request_index, scheduled_offset_ns in enumerate(scheduled_offsets_ns):
            scheduled_ns = level_started_ns + scheduled_offset_ns
            remaining_ns = scheduled_ns - time.monotonic_ns()
            if remaining_ns > 0:
                time.sleep(remaining_ns / 1_000_000_000)
            futures.append(
                executor.submit(
                    _run_request,
                    base_url,
                    server,
                    prompts[rate_request_index % len(prompts)],
                    first_request_index + rate_request_index,
                    rate_request_index,
                    level_started_ns,
                    scheduled_ns,
                )
            )
        records = [future.result() for future in futures]
    configured_end_ns = level_started_ns + int(duration_seconds * 1_000_000_000)
    remaining_ns = configured_end_ns - time.monotonic_ns()
    if remaining_ns > 0:
        time.sleep(remaining_ns / 1_000_000_000)
    duration_ns = max(
        int(duration_seconds * 1_000_000_000),
        max(
            (int(record["client_completed_offset_ns"]) for record in records),
            default=0,
        ),
    )
    return sorted(records, key=lambda record: int(record["request_index"])), duration_ns


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run_open_loop_benchmark(experiment: ExperimentSpec) -> OpenLoopRunResult:
    """Run configured open-loop arrivals against a managed llama-server."""
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
    arrival_schedule = build_arrival_schedule(load)
    schedule_path = run_dir / "arrival_schedule.json"
    _write_json(schedule_path, arrival_schedule)
    command = build_server_command(experiment, runtime_server, port)
    manifest: dict[str, Any] = {
        "artifact_schema_version": "2",
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "run_type": "open-loop-load",
        "experiment": experiment.model_dump(mode="json"),
        "effective_server": runtime_server.model_dump(mode="json"),
        "context_tokens_per_slot": experiment.server.context_size,
        "prompt_dataset_sha256": prompt_dataset_sha256(load.prompt_dataset),
        "prompt_count": len(prompts),
        "arrival_process": load.arrival_process.value,
        "arrival_algorithm": arrival_schedule["algorithm"],
        "scheduler_version": OPEN_LOOP_SCHEDULER_VERSION,
        "request_protocol_version": OPEN_LOOP_REQUEST_PROTOCOL_VERSION,
        "summary_version": OPEN_LOOP_SUMMARY_VERSION,
        "rate_order_protocol": load.rate_order_protocol.value,
        "rate_order_offset": int(load.rate_order_offset),
        "arrival_seed": int(load.arrival_seed),
        "arrival_schedule_sha256": hashlib.sha256(
            schedule_path.read_bytes()
        ).hexdigest(),
        "command": command,
        "capabilities": inspect_capabilities(
            run_dir,
            llama_bench_binary=experiment.llama_bench_binary,
            llama_server_binary=experiment.llama_server_binary,
        ).model_dump(mode="json"),
    }
    manifest_path = run_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    records: list[dict[str, Any]] = []
    summaries: list[dict[str, float | int | None]] = []
    started_ns = time.monotonic_ns()
    with (
        (logs_dir / "server.stdout.log").open("w", encoding="utf-8") as stdout,
        (logs_dir / "server.stderr.log").open("w", encoding="utf-8") as stderr,
    ):
        process = subprocess.Popen(
            command,
            stdout=stdout,
            stderr=stderr,
            text=True,
            start_new_session=True,
        )
        monitor: ResourceMonitor | None = None
        monitor_started = False
        try:
            monitor = ResourceMonitor(
                psutil.Process(process.pid), experiment.sample_interval_ms / 1000
            )
            monitor.start()
            monitor_started = True
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
            for rate_index, (rate, window) in enumerate(
                zip(load.arrival_rates_rps, arrival_schedule["windows"], strict=True)
            ):
                level_started_offset_ns = time.monotonic_ns() - monitor.started_ns
                level_records, duration_ns = run_rate_window(
                    base_url,
                    runtime_server,
                    prompts,
                    [int(value) for value in window["scheduled_offsets_ns"]],
                    float(load.duration_seconds),
                    int(load.max_client_workers),
                    request_index,
                )
                level_ended_offset_ns = time.monotonic_ns() - monitor.started_ns
                for record in level_records:
                    record["offered_requests_per_second"] = float(rate)
                    record["open_loop_rate_index"] = rate_index
                level_samples = [
                    sample
                    for sample in monitor.samples
                    if level_started_offset_ns
                    <= sample["monotonic_offset_ns"]
                    <= level_ended_offset_ns
                ]
                for sample in level_samples:
                    sample["open_loop_rate_index"] = rate_index
                rate_summary = summarize_rate(
                    float(rate),
                    duration_ns,
                    level_records,
                    level_samples,
                    float(load.latency_slo_ms),
                    int(window["arrival_window_ns"]),
                )
                summaries.append(rate_summary)
                records.extend(level_records)
                request_index += len(level_records)
                if experiment.fail_fast and any(
                    "error" in record for record in level_records
                ):
                    break
                if load.cooldown_seconds > 0:
                    time.sleep(float(load.cooldown_seconds))
        finally:
            try:
                stop_server(process, process_group=True)
            finally:
                samples = monitor.stop() if monitor_started and monitor else []
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
