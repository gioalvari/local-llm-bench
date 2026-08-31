"""Exact-token prompt and context-window benchmarks."""

import itertools
import json
import statistics
import subprocess
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError

import psutil
from pydantic import BaseModel

from localllm_bench.config import ContextCaseSpec, ExperimentSpec, ServerSpec
from localllm_bench.doctor import inspect_capabilities
from localllm_bench.load import percentile
from localllm_bench.model import validate_model
from localllm_bench.open_loop import prompt_dataset_sha256
from localllm_bench.server import (
    available_port,
    build_server_command,
    complete_token_prompt,
    stop_server,
    tokenize,
    wait_until_ready,
)
from localllm_bench.telemetry import ResourceMonitor


class ContextRunResult(BaseModel):
    """Location and aggregate counts for a context sweep."""

    run_id: str
    run_dir: Path
    completed_requests: int
    failed_requests: int
    summary: list[dict[str, float | int | str]]


def calibrate_prompt_tokens(base_tokens: list[int], target: int) -> list[int]:
    """Cycle a non-empty token corpus to an exact target length."""
    if not base_tokens:
        raise ValueError("base token corpus is empty")
    if target <= 0:
        raise ValueError("target token count must be positive")
    return list(itertools.islice(itertools.cycle(base_tokens), target))


def summarize_context_case(
    case: ContextCaseSpec,
    records: list[dict[str, Any]],
    load_time_ns: int,
    resource_samples: list[dict[str, int]],
) -> dict[str, float | int | str]:
    """Aggregate one fresh-server context case."""
    completed = [record for record in records if "error" not in record]
    failed = len(records) - len(completed)
    ttft_ms = [int(record["ttft_ns"]) / 1_000_000 for record in completed]
    e2e_ms = [int(record["e2e_latency_ns"]) / 1_000_000 for record in completed]
    prompt_ms = [float(record["backend_timings"]["prompt_ms"]) for record in completed]
    prompt_rates = [
        float(record["backend_timings"]["prompt_per_second"]) for record in completed
    ]
    decode_rates = [
        float(record["backend_timings"]["predicted_per_second"]) for record in completed
    ]
    return {
        "case": case.name,
        "series": ",".join(case.series),
        "context_size": int(case.context_size),
        "prompt_tokens": int(case.prompt_tokens),
        "requests": len(records),
        "completed_requests": len(completed),
        "failed_requests": failed,
        "error_rate": failed / len(records) if records else 0.0,
        "model_load_time_ms": load_time_ns / 1_000_000,
        "median_prompt_eval_ms": statistics.median(prompt_ms) if prompt_ms else 0.0,
        "median_prompt_tokens_per_second": (
            statistics.median(prompt_rates) if prompt_rates else 0.0
        ),
        "median_ttft_ms": statistics.median(ttft_ms) if ttft_ms else 0.0,
        "p95_ttft_ms": percentile(ttft_ms, 95) if ttft_ms else 0.0,
        "median_e2e_ms": statistics.median(e2e_ms) if e2e_ms else 0.0,
        "p95_e2e_ms": percentile(e2e_ms, 95) if e2e_ms else 0.0,
        "median_decode_tokens_per_second": (
            statistics.median(decode_rates) if decode_rates else 0.0
        ),
        "peak_process_tree_rss_bytes": max(
            (sample["process_tree_rss_bytes"] for sample in resource_samples),
            default=0,
        ),
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _run_context_case(
    experiment: ExperimentSpec,
    server: ServerSpec,
    case: ContextCaseSpec,
    corpus: str,
    run_dir: Path,
    first_request_index: int,
) -> tuple[list[dict[str, Any]], list[dict[str, int]], int]:
    port = available_port()
    command = build_server_command(experiment, server, port)
    stdout_path = run_dir / "logs" / f"{case.name}.stdout.log"
    stderr_path = run_dir / "logs" / f"{case.name}.stderr.log"
    records: list[dict[str, Any]] = []
    with (
        stdout_path.open("w", encoding="utf-8") as stdout,
        stderr_path.open("w", encoding="utf-8") as stderr,
    ):
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, text=True)
        monitor = ResourceMonitor(
            psutil.Process(process.pid), experiment.sample_interval_ms / 1000
        )
        monitor.start()
        try:
            base_url = f"http://127.0.0.1:{port}"
            load_time_ns = wait_until_ready(
                base_url, process, server.startup_timeout_seconds
            )
            base_tokens = tokenize(base_url, corpus, server.request_timeout_seconds)
            prompt = calibrate_prompt_tokens(base_tokens, int(case.prompt_tokens))
            sweep = experiment.context_sweep
            if sweep is None:
                raise ValueError("experiment does not define a context_sweep section")
            for warmup_index in range(sweep.warmup_requests):
                complete_token_prompt(
                    base_url,
                    server,
                    prompt,
                    request_index=-(warmup_index + 1),
                    output_tokens=int(sweep.output_tokens),
                )
            for repetition in range(sweep.repetitions):
                request_index = first_request_index + repetition
                try:
                    measurement = complete_token_prompt(
                        base_url,
                        server,
                        prompt,
                        request_index=request_index,
                        output_tokens=int(sweep.output_tokens),
                    )
                    observed = measurement["backend_timings"].get("prompt_n")
                    if observed != int(case.prompt_tokens):
                        raise ValueError(
                            f"backend reported {observed} prompt tokens, "
                            f"expected {case.prompt_tokens}"
                        )
                    records.append(
                        {
                            "case": case.name,
                            "series": list(case.series),
                            "context_size": int(case.context_size),
                            "target_prompt_tokens": int(case.prompt_tokens),
                            "repetition": repetition,
                            **measurement,
                        }
                    )
                except (HTTPError, URLError, TimeoutError, ValueError) as error:
                    records.append(
                        {
                            "case": case.name,
                            "request_index": request_index,
                            "repetition": repetition,
                            "error": str(error),
                        }
                    )
                    if experiment.fail_fast:
                        break
        finally:
            stop_server(process)
            samples = monitor.stop()
    return records, samples, load_time_ns


def run_context_sweep(experiment: ExperimentSpec) -> ContextRunResult:
    """Run exact prompt lengths with a fresh server per context window."""
    if experiment.server is None:
        raise ValueError("experiment does not define a server section")
    if experiment.context_sweep is None:
        raise ValueError("experiment does not define a context_sweep section")
    validate_model(experiment.model)
    sweep = experiment.context_sweep
    corpus = sweep.corpus.read_text(encoding="utf-8").strip()
    if not corpus:
        raise ValueError("context sweep corpus is empty")
    run_id = (
        f"{experiment.experiment_id}-context-"
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    )
    run_dir = experiment.output_dir / run_id
    (run_dir / "logs").mkdir(parents=True)
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "run_type": "context-sweep",
        "experiment": experiment.model_dump(mode="json"),
        "corpus_sha256": prompt_dataset_sha256(sweep.corpus),
        "capabilities": inspect_capabilities(run_dir).model_dump(mode="json"),
    }
    manifest_path = run_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    all_records: list[dict[str, Any]] = []
    all_samples: list[dict[str, Any]] = []
    summaries: list[dict[str, float | int | str]] = []
    started_ns = time.monotonic_ns()
    request_index = 0
    for case in sweep.cases:
        runtime_server = experiment.server.model_copy(
            update={"context_size": int(case.context_size), "parallel": 1}
        )
        records, samples, load_time_ns = _run_context_case(
            experiment,
            runtime_server,
            case,
            corpus,
            run_dir,
            request_index,
        )
        all_records.extend(records)
        all_samples.extend({"case": case.name, **sample} for sample in samples)
        summaries.append(summarize_context_case(case, records, load_time_ns, samples))
        request_index += len(records)
        if experiment.fail_fast and any("error" in record for record in records):
            break
    with (run_dir / "requests.jsonl").open("w", encoding="utf-8") as stream:
        for record in all_records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    with (run_dir / "resource_samples.jsonl").open("w", encoding="utf-8") as stream:
        for sample in all_samples:
            stream.write(json.dumps(sample, sort_keys=True) + "\n")
    _write_json(run_dir / "summary.json", summaries)
    manifest["summary"] = summaries
    manifest["process_wall_time_ns"] = time.monotonic_ns() - started_ns
    _write_json(manifest_path, manifest)
    completed = sum("error" not in record for record in all_records)
    return ContextRunResult(
        run_id=run_id,
        run_dir=run_dir,
        completed_requests=completed,
        failed_requests=len(all_records) - completed,
        summary=summaries,
    )
