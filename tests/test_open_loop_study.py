import hashlib
import json
import statistics
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from localllm_bench.config import (
    ArrivalProcess,
    ExperimentSpec,
    OpenLoopSpec,
    RateOrderProtocol,
)
from localllm_bench.open_loop import (
    OPEN_LOOP_REQUEST_PROTOCOL_VERSION,
    OPEN_LOOP_SCHEDULER_VERSION,
    OPEN_LOOP_SUMMARY_VERSION,
    OpenLoopRunResult,
    build_arrival_schedule,
    summarize_rate,
)
from localllm_bench.open_loop_study import (
    OpenLoopAnalysisArtifact,
    analyze_open_loop_runs,
    bootstrap_run_mean_interval,
    run_repeated_open_loop_benchmark,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_run(
    root: Path,
    run_id: str,
    seed: int,
    *,
    rates: list[float] | None = None,
    arrival_process: ArrivalProcess = ArrivalProcess.POISSON,
    rate_order_offset: int = 0,
) -> Path:
    root.mkdir(parents=True)
    configured_rates = rates or [2.0, 4.0]
    canonical_rates = sorted(configured_rates)
    ordered_rates = (
        canonical_rates[rate_order_offset:] + canonical_rates[:rate_order_offset]
    )
    open_loop = OpenLoopSpec(
        prompt_dataset=Path("prompts.jsonl"),
        arrival_rates_rps=ordered_rates,
        arrival_process=arrival_process,
        arrival_seed=seed,
        duration_seconds=1.0,
        warmup_requests=1,
        server_slots=2,
        max_client_workers=4,
        latency_slo_ms=500.0,
        cooldown_seconds=0.0,
        independent_runs=1,
        bootstrap_iterations=100,
        bootstrap_seed=42,
        rate_order_protocol=RateOrderProtocol.CYCLIC,
        rate_order_offset=rate_order_offset,
    )
    schedule = build_arrival_schedule(open_loop)
    schedule_path = root / "arrival_schedule.json"
    schedule_path.write_text(
        json.dumps(schedule, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    experiment = {
        "schema_version": "2",
        "sample_interval_ms": 100,
        "fail_fast": False,
        "model": {
            "sha256": "a" * 64,
            "quantization": "Q4_K_M",
            "source_repo": "owner/model",
            "source_revision": "revision",
        },
        "open_loop": open_loop.model_dump(mode="json"),
    }
    records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    resources: list[dict[str, int]] = []
    request_index = 0
    for rate_index, window in enumerate(schedule["windows"]):
        rate = float(window["offered_requests_per_second"])
        rate_records: list[dict[str, Any]] = []
        for rate_request_index, offset in enumerate(window["scheduled_offsets_ns"]):
            completed_offset = int(offset) + 100_000_000
            record = {
                "request_index": request_index,
                "rate_request_index": rate_request_index,
                "prompt_id": "prompt",
                "scheduled_offset_ns": int(offset),
                "client_started_offset_ns": int(offset),
                "client_schedule_delay_ns": 0,
                "client_completed_offset_ns": completed_offset,
                "ttft_ns": 10_000_000,
                "e2e_latency_ns": 100_000_000,
                "output_tokens": 10,
                "offered_requests_per_second": rate,
                "open_loop_rate_index": rate_index,
            }
            request_index += 1
            rate_records.append(record)
            records.append(record)
        resource_start = rate_index * 2_000_000_000
        rate_samples = [
            {
                "monotonic_offset_ns": resource_start + 500_000_000,
                "process_tree_rss_bytes": 1024**3,
                "open_loop_rate_index": rate_index,
            }
        ]
        resources.extend(rate_samples)
        duration_ns = max(
            1_000_000_000,
            max(
                (int(record["client_completed_offset_ns"]) for record in rate_records),
                default=0,
            ),
        )
        summary = summarize_rate(
            rate,
            duration_ns,
            rate_records,
            rate_samples,
            500.0,
            1_000_000_000,
        )
        summaries.append(summary)
    (root / "requests.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    (root / "resource_samples.jsonl").write_text(
        "".join(json.dumps(sample, sort_keys=True) + "\n" for sample in resources),
        encoding="utf-8",
    )
    capabilities = {
        "architecture": "arm64",
        "machine_model": "Mac16,7",
        "processor": "Apple M4 Pro",
        "logical_cpus": 10,
        "memory_bytes": 16 * 1024**3,
        "os": "Darwin",
        "os_release": "25.0",
        "physical_cpus": 10,
        "unified_memory": True,
        "llama_server": {"sha256": "server"},
    }
    manifest = {
        "artifact_schema_version": "2",
        "run_id": run_id,
        "run_type": "open-loop-load",
        "experiment": experiment,
        "effective_server": {
            "output_tokens": 64,
            "context_size": 4096,
            "batch_size": 512,
            "ubatch_size": 128,
            "threads": 10,
            "gpu_layers": -1,
            "flash_attention": "on",
            "parallel": 2,
            "startup_timeout_seconds": 120,
            "request_timeout_seconds": 120,
        },
        "context_tokens_per_slot": 2048,
        "prompt_dataset_sha256": "dataset",
        "prompt_count": 8,
        "arrival_process": arrival_process.value,
        "arrival_algorithm": schedule["algorithm"],
        "scheduler_version": OPEN_LOOP_SCHEDULER_VERSION,
        "request_protocol_version": OPEN_LOOP_REQUEST_PROTOCOL_VERSION,
        "summary_version": OPEN_LOOP_SUMMARY_VERSION,
        "rate_order_protocol": RateOrderProtocol.CYCLIC.value,
        "rate_order_offset": rate_order_offset,
        "arrival_seed": seed,
        "arrival_schedule_sha256": _sha256(schedule_path),
        "capabilities": capabilities,
        "summary": summaries,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return root


def _experiment(tmp_path: Path) -> ExperimentSpec:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"prompt_id":"one","prompt":"hello"}\n', encoding="utf-8")
    return ExperimentSpec.model_validate(
        {
            "schema_version": "2",
            "experiment_id": "study-test",
            "output_dir": tmp_path / "runs",
            "model": {
                "name": "tiny",
                "path": str(model),
                "sha256": hashlib.sha256(b"model").hexdigest(),
                "quantization": "Q4",
            },
            "matrix": {
                "workloads": [
                    {"name": "short", "prompt_tokens": 8, "generation_tokens": 4}
                ]
            },
            "server": {"prompt": "hello"},
            "open_loop": {
                "prompt_dataset": str(prompts),
                "arrival_rates_rps": [2, 4],
                "arrival_process": "poisson",
                "independent_runs": 6,
                "bootstrap_iterations": 100,
                "cooldown_seconds": 0,
            },
        }
    )


def test_bootstrap_run_mean_interval_is_deterministic() -> None:
    first = bootstrap_run_mean_interval([1.0, 3.0, 8.0], iterations=200, seed=7)
    repeated = bootstrap_run_mean_interval([1.0, 3.0, 8.0], iterations=200, seed=7)
    assert first == repeated
    assert first != bootstrap_run_mean_interval([1.0, 3.0, 8.0], iterations=200, seed=8)
    with pytest.raises(ValueError, match="at least two"):
        bootstrap_run_mean_interval([1.0])
    with pytest.raises(ValueError, match="iterations"):
        bootstrap_run_mean_interval([1.0, 2.0], iterations=0)


def test_analyze_open_loop_runs_writes_portable_run_level_artifacts(
    tmp_path: Path,
) -> None:
    runs = [
        _write_run(
            tmp_path / f"run-{seed}",
            f"run-{seed}",
            seed,
            rate_order_offset=(seed - 42) % 2,
        )
        for seed in (46, 43, 45, 42, 47, 44)
    ]
    result = analyze_open_loop_runs(
        runs, tmp_path / "analysis", bootstrap_iterations=200, bootstrap_seed=7
    )
    artifact = result.artifact
    assert [source.arrival_seed for source in artifact.source_runs] == [
        42,
        43,
        44,
        45,
        46,
        47,
    ]
    estimate = artifact.rates[0].metrics["achieved_requests_per_second"]
    assert len(estimate.run_values) == 6
    assert estimate.mean == statistics.fmean(
        float(value) for value in estimate.run_values if value is not None
    )
    assert estimate.ci95_low is not None
    payload = (result.output_dir / "analysis.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in payload
    assert OpenLoopAnalysisArtifact.model_validate_json(payload) == artifact
    html = (result.output_dir / "analysis.html").read_text(encoding="utf-8")
    assert "resampling unit is one complete run" in html
    assert "Mean [95% CI]" in html


def test_analyze_open_loop_runs_preserves_undefined_metrics(tmp_path: Path) -> None:
    runs = [
        _write_run(
            tmp_path / f"run-{seed}",
            f"run-{seed}",
            seed,
            rates=[0.1, 4.0],
            rate_order_offset=(seed - 42) % 2,
        )
        for seed in range(42, 48)
    ]
    result = analyze_open_loop_runs(
        runs, tmp_path / "analysis", bootstrap_iterations=20
    )
    estimate = result.artifact.rates[0].metrics["median_e2e_ms"]
    assert estimate.defined_runs == 1
    assert estimate.mean is None
    assert "n/a" in (result.output_dir / "analysis.html").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duplicate-seed", "unique arrival seeds"),
        ("tampered-schedule", "digest"),
        ("coherent-schedule-edit", "configured protocol"),
        ("coherent-summary-edit", "raw request"),
        ("incomplete-grid", "rate grid"),
        ("missing-metric", "invalid open-loop summary"),
        ("extra-metric", "invalid open-loop summary"),
        ("false-cyclic-offset", "declared rotation offset"),
        ("invalid-request-tag", "invalid open-loop rate tag"),
        ("different-hardware", "hardware"),
        ("different-protocol", "protocol"),
    ],
)
def test_analyze_open_loop_runs_rejects_incompatible_inputs(
    tmp_path: Path, mutation: str, message: str
) -> None:
    seeds = [42, 43, 44, 45, 46, 47]
    if mutation == "duplicate-seed":
        seeds[-1] = 42
    runs = [
        _write_run(
            tmp_path / f"run-{index}",
            f"run-{index}",
            seed,
            rate_order_offset=index % 2,
        )
        for index, seed in enumerate(seeds)
    ]
    second = runs[-1]
    manifest_path = second / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "tampered-schedule":
        (second / "arrival_schedule.json").write_text("{}\n", encoding="utf-8")
    elif mutation == "coherent-schedule-edit":
        schedule_path = second / "arrival_schedule.json"
        schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
        schedule["windows"][0]["scheduled_offsets_ns"][0] += 1
        schedule_path.write_text(
            json.dumps(schedule, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        manifest["arrival_schedule_sha256"] = _sha256(schedule_path)
    elif mutation == "coherent-summary-edit":
        summary = manifest["summary"]
        summary[0]["achieved_requests_per_second"] += 1
        (second / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    elif mutation == "incomplete-grid":
        summary = manifest["summary"][:-1]
        manifest["summary"] = summary
        (second / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    elif mutation == "missing-metric":
        summary = manifest["summary"]
        del summary[0]["median_e2e_ms"]
        (second / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    elif mutation == "extra-metric":
        summary = manifest["summary"]
        summary[0]["unknown_metric"] = 1
        (second / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    elif mutation == "false-cyclic-offset":
        manifest["rate_order_offset"] = 0
        manifest["experiment"]["open_loop"]["rate_order_offset"] = 0
    elif mutation == "invalid-request-tag":
        requests_path = second / "requests.jsonl"
        requests = [
            json.loads(line)
            for line in requests_path.read_text(encoding="utf-8").splitlines()
        ]
        requests.append(
            {
                "request_index": 999,
                "rate_request_index": 0,
                "open_loop_rate_index": 99,
                "offered_requests_per_second": 2.0,
            }
        )
        requests_path.write_text(
            "".join(json.dumps(item) + "\n" for item in requests), encoding="utf-8"
        )
    elif mutation == "different-hardware":
        manifest["capabilities"]["architecture"] = "x86_64"
    else:
        manifest["effective_server"]["threads"] = 8
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match=message):
        analyze_open_loop_runs(runs, tmp_path / "analysis", bootstrap_iterations=20)
    assert not (tmp_path / "analysis").exists()


def test_analyze_fixed_runs_allows_repeated_irrelevant_seed(tmp_path: Path) -> None:
    runs = [
        _write_run(
            tmp_path / f"run-{index}",
            f"run-{index}",
            42,
            arrival_process=ArrivalProcess.FIXED,
            rate_order_offset=index % 2,
        )
        for index in range(6)
    ]
    result = analyze_open_loop_runs(
        runs, tmp_path / "analysis", bootstrap_iterations=20
    )
    assert result.artifact.arrival_process == "fixed"
    assert {source.arrival_seed for source in result.artifact.source_runs} == {42}


def test_run_repeated_open_loop_uses_consecutive_seeds_and_child_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment = _experiment(tmp_path)
    observed: list[tuple[int, Path]] = []
    monkeypatch.setattr(
        "localllm_bench.open_loop_study.inspect_capabilities",
        lambda **_: SimpleNamespace(
            llama_server=SimpleNamespace(available=True, sha256="server")
        ),
    )

    def fake_run(child: ExperimentSpec) -> OpenLoopRunResult:
        assert child.open_loop is not None
        seed = int(child.open_loop.arrival_seed)
        observed.append((seed, child.output_dir))
        run_dir = _write_run(
            child.output_dir / f"run-{seed}",
            f"run-{seed}",
            seed,
            rates=[float(value) for value in child.open_loop.arrival_rates_rps],
            rate_order_offset=int(child.open_loop.rate_order_offset),
        )
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        return OpenLoopRunResult(
            run_id=f"run-{seed}",
            run_dir=run_dir,
            completed_requests=sum(item["completed_requests"] for item in summary),
            failed_requests=0,
            summary=summary,
        )

    monkeypatch.setattr(
        "localllm_bench.open_loop_study.run_open_loop_benchmark", fake_run
    )
    result = run_repeated_open_loop_benchmark(experiment)
    assert [seed for seed, _ in observed] == [42, 43, 44, 45, 46, 47]
    assert all(path == result.study_dir / "repetitions" for _, path in observed)
    assert result.repetition_run_ids == [
        "run-42",
        "run-43",
        "run-44",
        "run-45",
        "run-46",
        "run-47",
    ]
    assert (result.analysis_dir / "analysis.json").is_file()


def test_repeated_open_loop_failure_stops_and_writes_no_analysis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment = _experiment(tmp_path)
    attempts: list[int] = []
    monkeypatch.setattr(
        "localllm_bench.open_loop_study.inspect_capabilities",
        lambda **_: SimpleNamespace(
            llama_server=SimpleNamespace(available=True, sha256="server")
        ),
    )

    def failing_run(child: ExperimentSpec) -> OpenLoopRunResult:
        assert child.open_loop is not None
        seed = int(child.open_loop.arrival_seed)
        attempts.append(seed)
        if seed == 43:
            raise RuntimeError("server failed")
        run_dir = _write_run(
            child.output_dir / f"run-{seed}",
            f"run-{seed}",
            seed,
            rates=[float(value) for value in child.open_loop.arrival_rates_rps],
            rate_order_offset=int(child.open_loop.rate_order_offset),
        )
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        return OpenLoopRunResult(
            run_id=f"run-{seed}",
            run_dir=run_dir,
            completed_requests=sum(item["completed_requests"] for item in summary),
            failed_requests=0,
            summary=summary,
        )

    monkeypatch.setattr(
        "localllm_bench.open_loop_study.run_open_loop_benchmark", failing_run
    )
    with pytest.raises(RuntimeError, match="server failed"):
        run_repeated_open_loop_benchmark(experiment)
    assert attempts == [42, 43]
    studies = list((tmp_path / "runs").iterdir())
    assert len(studies) == 1
    failure = json.loads((studies[0] / "failure.json").read_text(encoding="utf-8"))
    assert failure["failed_repetition"] == 1
    assert failure["completed_run_ids"] == ["run-42"]
    assert not (studies[0] / "analysis").exists()


def test_repeated_open_loop_rejects_returned_fail_fast_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment = _experiment(tmp_path)
    monkeypatch.setattr(
        "localllm_bench.open_loop_study.inspect_capabilities",
        lambda **_: SimpleNamespace(
            llama_server=SimpleNamespace(available=True, sha256="server")
        ),
    )

    def failed_run(child: ExperimentSpec) -> OpenLoopRunResult:
        assert child.open_loop is not None
        seed = int(child.open_loop.arrival_seed)
        run_dir = _write_run(
            child.output_dir / f"run-{seed}",
            f"run-{seed}",
            seed,
            rates=[float(value) for value in child.open_loop.arrival_rates_rps],
            rate_order_offset=int(child.open_loop.rate_order_offset),
        )
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        return OpenLoopRunResult(
            run_id=f"run-{seed}",
            run_dir=run_dir,
            completed_requests=1,
            failed_requests=1,
            summary=summary,
        )

    monkeypatch.setattr(
        "localllm_bench.open_loop_study.run_open_loop_benchmark", failed_run
    )
    with pytest.raises(RuntimeError, match="did not complete every rate"):
        run_repeated_open_loop_benchmark(experiment)
