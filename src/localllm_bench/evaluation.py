"""Objective, source-grounded model quality evaluation."""

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

from localllm_bench.config import ExperimentSpec
from localllm_bench.dataset import dataset_sha256, load_dataset
from localllm_bench.doctor import inspect_capabilities
from localllm_bench.model import validate_model
from localllm_bench.prompts import PROMPT_VERSION, build_evaluation_prompt
from localllm_bench.quality import score_structured_answer
from localllm_bench.server import (
    available_port,
    build_server_command,
    complete_chat_prompt,
    stop_server,
    wait_until_ready,
)
from localllm_bench.telemetry import ResourceMonitor


class EvaluationRunResult(BaseModel):
    """Location and item counts for one quality evaluation."""

    run_id: str
    run_dir: Path
    completed_items: int
    failed_items: int
    summary: dict[str, dict[str, float | int]]


def aggregate_records(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, float | int]]:
    """Aggregate item-level scores separately for each prompt arm."""
    arms = sorted({str(record["prompt_arm"]) for record in records})
    summary: dict[str, dict[str, float | int]] = {}
    for arm in arms:
        selected = [record for record in records if record["prompt_arm"] == arm]
        scores = [record["score"] for record in selected]
        numeric_records = [
            record for record in selected if record.get("answer_type") == "numeric"
        ]
        numeric_scores = [
            float(record["score"].get("numeric_accuracy") or 0.0)
            for record in numeric_records
        ]
        unit_scores = [
            float(record["score"].get("unit_accuracy") or 0.0)
            for record in numeric_records
        ]
        total_latency_seconds = (
            sum(int(record["e2e_latency_ns"]) for record in selected) / 1_000_000_000
        )
        correct = sum(float(score["answer_accuracy"]) for score in scores)
        summary[arm] = {
            "items": len(selected),
            "scorable_response_rate": statistics.fmean(
                float(score.get("parsed_answer") is not None) for score in scores
            ),
            "schema_valid_rate": statistics.fmean(
                float(score["schema_valid"]) for score in scores
            ),
            "answer_accuracy": statistics.fmean(
                float(score["answer_accuracy"]) for score in scores
            ),
            "exact_match": statistics.fmean(
                float(score["exact_match"]) for score in scores
            ),
            "token_f1": statistics.fmean(float(score["token_f1"]) for score in scores),
            "numeric_accuracy": (
                statistics.fmean(numeric_scores) if numeric_scores else 0.0
            ),
            "unit_accuracy": statistics.fmean(unit_scores) if unit_scores else 0.0,
            "median_ttft_ms": statistics.median(
                int(record["ttft_ns"]) / 1_000_000 for record in selected
            ),
            "median_e2e_ms": statistics.median(
                int(record["e2e_latency_ns"]) / 1_000_000 for record in selected
            ),
            "quality_adjusted_answers_per_second": (
                correct / total_latency_seconds if total_latency_seconds > 0 else 0.0
            ),
        }
    return summary


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run_evaluation(experiment: ExperimentSpec) -> EvaluationRunResult:
    """Evaluate all configured prompt arms through a managed llama-server."""
    if experiment.server is None:
        raise ValueError("experiment does not define a server section")
    if experiment.evaluation is None:
        raise ValueError("experiment does not define an evaluation section")
    validate_model(experiment.model)
    evaluation = experiment.evaluation
    items = load_dataset(evaluation.dataset)
    server = experiment.server
    port = server.port or available_port()
    run_id = (
        f"{experiment.experiment_id}-quality-"
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    )
    run_dir = experiment.output_dir / run_id
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True)
    command = build_server_command(experiment, server, port)
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "run_type": "quality-evaluation",
        "dataset_path": str(evaluation.dataset),
        "dataset_sha256": dataset_sha256(evaluation.dataset),
        "dataset_items": len(items),
        "prompt_version": PROMPT_VERSION,
        "experiment": experiment.model_dump(mode="json"),
        "command": command,
        "capabilities": inspect_capabilities(run_dir).model_dump(mode="json"),
    }
    manifest_path = run_dir / "manifest.json"
    _write_manifest(manifest_path, manifest)
    records: list[dict[str, Any]] = []
    failed = 0
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
                base_url, process, server.startup_timeout_seconds
            )
            _write_manifest(manifest_path, manifest)
            with (run_dir / "evaluations.jsonl").open("w", encoding="utf-8") as stream:
                request_index = 0
                should_stop = False
                for arm in evaluation.prompt_arms:
                    for item in items:
                        try:
                            measurement = complete_chat_prompt(
                                base_url,
                                server,
                                build_evaluation_prompt(item, arm),
                                request_index=request_index,
                                output_tokens=evaluation.output_tokens,
                            )
                            response = str(measurement.pop("response_text"))
                            score = score_structured_answer(
                                response,
                                answer_type=item.answer_type,
                                references=item.references,
                                expected_value=item.expected_value,
                                absolute_tolerance=item.absolute_tolerance,
                                relative_tolerance=item.relative_tolerance,
                                accepted_units=item.accepted_units,
                            )
                            record = {
                                "item_id": item.item_id,
                                "source_document_id": item.source_document_id,
                                "category": item.category,
                                "answer_type": item.answer_type,
                                "prompt_arm": arm.value,
                                "raw_response": response,
                                "score": score.model_dump(mode="json"),
                                **measurement,
                            }
                            records.append(record)
                            stream.write(json.dumps(record, sort_keys=True) + "\n")
                            stream.flush()
                        except (HTTPError, URLError, TimeoutError, ValueError) as error:
                            failed += 1
                            with (run_dir / "failures.jsonl").open(
                                "a", encoding="utf-8"
                            ) as failures:
                                failures.write(
                                    json.dumps(
                                        {
                                            "request_index": request_index,
                                            "item_id": item.item_id,
                                            "prompt_arm": arm.value,
                                            "error": str(error),
                                        },
                                        sort_keys=True,
                                    )
                                    + "\n"
                                )
                            should_stop = experiment.fail_fast
                        request_index += 1
                        if should_stop:
                            break
                    if should_stop:
                        break
        finally:
            stop_server(process)
            samples = monitor.stop()
    with (run_dir / "resource_samples.jsonl").open("w", encoding="utf-8") as stream:
        for sample in samples:
            stream.write(json.dumps(sample, sort_keys=True) + "\n")
    summary = aggregate_records(records)
    manifest["process_wall_time_ns"] = time.monotonic_ns() - started_ns
    manifest["peak_process_tree_rss_bytes"] = max(
        (sample["process_tree_rss_bytes"] for sample in samples), default=0
    )
    manifest["summary"] = summary
    _write_manifest(manifest_path, manifest)
    _write_manifest(run_dir / "summary.json", summary)
    return EvaluationRunResult(
        run_id=run_id,
        run_dir=run_dir,
        completed_items=len(records),
        failed_items=failed,
        summary=summary,
    )
