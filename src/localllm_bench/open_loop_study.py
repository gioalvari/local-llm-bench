"""Independent-run analysis for reproducible open-loop benchmarks."""

import hashlib
import html
import json
import math
import random
import shutil
import statistics
import uuid
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from localllm_bench.config import (
    ArrivalProcess,
    ExperimentSpec,
    OpenLoopSpec,
    RateOrderProtocol,
)
from localllm_bench.doctor import inspect_capabilities
from localllm_bench.model import validate_model
from localllm_bench.open_loop import (
    OPEN_LOOP_REQUEST_PROTOCOL_VERSION,
    OPEN_LOOP_SCHEDULER_VERSION,
    OPEN_LOOP_SUMMARY_VERSION,
    OpenLoopRunResult,
    build_arrival_schedule,
    load_prompts,
    run_open_loop_benchmark,
    summarize_rate,
)

MIN_INDEPENDENT_RUNS = 5

_METRIC_UNITS = {
    "requests": "requests",
    "completed_requests": "requests",
    "failed_requests": "requests",
    "realized_offered_requests_per_second": "requests/s",
    "achieved_requests_per_second": "requests/s",
    "aggregate_output_tokens_per_second": "tokens/s",
    "goodput_requests_per_second": "requests/s",
    "slo_attainment_rate": "fraction",
    "error_rate": "fraction",
    "median_ttft_ms": "ms",
    "p95_ttft_ms": "ms",
    "median_e2e_ms": "ms",
    "p95_e2e_ms": "ms",
    "p95_client_schedule_delay_ms": "ms",
    "max_client_in_flight": "requests",
    "peak_process_tree_rss_bytes": "bytes",
}


class BootstrapMetadata(BaseModel):
    """Frozen run-level bootstrap protocol."""

    method: str = "percentile-bootstrap-of-run-means-v1"
    resampling_unit: str = "independent-run"
    statistic: str = "arithmetic-mean"
    confidence_level: float = 0.95
    iterations: int
    seed: int


class MetricEstimate(BaseModel):
    """Run values and their run-level mean confidence interval."""

    unit: str
    run_values: list[float | int | None]
    defined_runs: int
    mean: float | None
    ci95_low: float | None
    ci95_high: float | None


class RateEstimate(BaseModel):
    """Independent-run estimates for one configured arrival rate."""

    offered_requests_per_second: float
    metrics: dict[str, MetricEstimate]


class SourceRun(BaseModel):
    """Identity and integrity fingerprints for one source run."""

    repetition: int
    run_id: str
    arrival_seed: int
    manifest_sha256: str
    summary_sha256: str
    arrival_schedule_sha256: str
    requests_sha256: str
    resource_samples_sha256: str
    rate_order_offset: int


OptionalNonNegativeFloat = Annotated[float | None, Field(ge=0, allow_inf_nan=False)]
OptionalRate = Annotated[float | None, Field(ge=0, le=1, allow_inf_nan=False)]


class OpenLoopRateSummary(BaseModel):
    """Strict validated summary for one measured arrival-rate window."""

    model_config = ConfigDict(strict=True, extra="forbid")

    offered_requests_per_second: float = Field(gt=0, allow_inf_nan=False)
    realized_offered_requests_per_second: float = Field(ge=0, allow_inf_nan=False)
    requests: int = Field(ge=0)
    completed_requests: int = Field(ge=0)
    failed_requests: int = Field(ge=0)
    error_rate: OptionalRate
    arrival_window_ns: int = Field(gt=0)
    duration_ns: int = Field(gt=0)
    achieved_requests_per_second: float = Field(ge=0, allow_inf_nan=False)
    aggregate_output_tokens_per_second: float = Field(ge=0, allow_inf_nan=False)
    goodput_requests_per_second: float = Field(ge=0, allow_inf_nan=False)
    slo_attainment_rate: OptionalRate
    median_ttft_ms: OptionalNonNegativeFloat
    p95_ttft_ms: OptionalNonNegativeFloat
    median_e2e_ms: OptionalNonNegativeFloat
    p95_e2e_ms: OptionalNonNegativeFloat
    p95_client_schedule_delay_ms: OptionalNonNegativeFloat
    max_client_in_flight: int = Field(ge=0)
    peak_process_tree_rss_bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_counts_and_missing_values(self) -> "OpenLoopRateSummary":
        """Require consistent counts and explicit nulls for empty samples."""
        if self.requests != self.completed_requests + self.failed_requests:
            raise ValueError("requests must equal completed plus failed requests")
        if self.requests == 0:
            optional_values = (
                self.error_rate,
                self.slo_attainment_rate,
                self.median_ttft_ms,
                self.p95_ttft_ms,
                self.median_e2e_ms,
                self.p95_e2e_ms,
                self.p95_client_schedule_delay_ms,
            )
            if any(value is not None for value in optional_values):
                raise ValueError("empty rate windows require null sample statistics")
        elif self.error_rate is None or self.slo_attainment_rate is None:
            raise ValueError("non-empty rate windows require error and SLO rates")
        if self.completed_requests == 0 and any(
            value is not None
            for value in (
                self.median_ttft_ms,
                self.p95_ttft_ms,
                self.median_e2e_ms,
                self.p95_e2e_ms,
            )
        ):
            raise ValueError("runs without completions require null latency metrics")
        return self


class OpenLoopAnalysisArtifact(BaseModel):
    """Portable repeated-run open-loop analysis."""

    artifact_schema_version: str = "1"
    analysis_type: str = "open-loop-repetition-bootstrap"
    repetition_count: int
    bootstrap: BootstrapMetadata
    model_sha256: str
    protocol_sha256: str
    hardware_sha256: str
    llama_server_sha256: str
    prompt_dataset_sha256: str
    arrival_process: str
    arrival_algorithm: str
    source_runs: list[SourceRun]
    rates: list[RateEstimate]


class OpenLoopAnalysisResult(BaseModel):
    """Output location and portable repeated-run analysis."""

    output_dir: Path
    artifact: OpenLoopAnalysisArtifact


class OpenLoopStudyResult(BaseModel):
    """Completed child runs and their repeated-run analysis."""

    study_id: str
    study_dir: Path
    repetition_run_ids: list[str]
    analysis_dir: Path
    analysis: OpenLoopAnalysisArtifact


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _linear_percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def bootstrap_run_mean_interval(
    run_values: list[float], *, iterations: int = 10_000, seed: int = 42
) -> tuple[float, float]:
    """Return a deterministic 95% percentile interval over independent runs."""
    if len(run_values) < 2 or not all(math.isfinite(value) for value in run_values):
        raise ValueError("bootstrap requires at least two finite run values")
    if iterations <= 0:
        raise ValueError("bootstrap iterations must be positive")
    generator = random.Random(seed)
    estimates = [
        statistics.fmean(
            run_values[generator.randrange(len(run_values))] for _ in run_values
        )
        for _ in range(iterations)
    ]
    return (
        _linear_percentile(estimates, 0.025),
        _linear_percentile(estimates, 0.975),
    )


def _hardware_sha256(manifest: dict[str, Any]) -> str:
    capabilities = manifest["capabilities"]
    hardware = {
        key: capabilities.get(key)
        for key in (
            "architecture",
            "machine_model",
            "logical_cpus",
            "memory_bytes",
            "os",
            "os_release",
            "physical_cpus",
            "processor",
            "unified_memory",
        )
    }
    payload = json.dumps(hardware, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _protocol_sha256(manifest: dict[str, Any]) -> str:
    experiment = manifest["experiment"]
    open_loop = experiment["open_loop"]
    effective_server = manifest["effective_server"]
    protocol = {
        "schema_version": experiment["schema_version"],
        "model": {
            key: experiment["model"].get(key)
            for key in ("sha256", "quantization", "source_repo", "source_revision")
        },
        "effective_server": {
            key: effective_server.get(key)
            for key in (
                "output_tokens",
                "context_size",
                "batch_size",
                "ubatch_size",
                "threads",
                "gpu_layers",
                "flash_attention",
                "parallel",
                "startup_timeout_seconds",
                "request_timeout_seconds",
            )
        },
        "open_loop": {
            "arrival_rates_rps": sorted(
                float(value) for value in open_loop["arrival_rates_rps"]
            ),
            **{
                key: open_loop.get(key)
                for key in (
                    "arrival_process",
                    "duration_seconds",
                    "warmup_requests",
                    "server_slots",
                    "max_client_workers",
                    "latency_slo_ms",
                    "cooldown_seconds",
                    "rate_order_protocol",
                )
            },
        },
        "arrival_algorithm": manifest["arrival_algorithm"],
        "scheduler_version": manifest["scheduler_version"],
        "request_protocol_version": manifest["request_protocol_version"],
        "summary_version": manifest["summary_version"],
        "context_tokens_per_slot": manifest["context_tokens_per_slot"],
        "prompt_dataset_sha256": manifest["prompt_dataset_sha256"],
        "prompt_count": manifest["prompt_count"],
        "sample_interval_ms": experiment["sample_interval_ms"],
        "fail_fast": experiment["fail_fast"],
    }
    payload = json.dumps(protocol, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _rate_grid(values: list[float | int]) -> tuple[str, ...]:
    rates = tuple(float(value).hex() for value in values)
    if len(rates) != len(set(rates)):
        raise ValueError("open-loop run contains duplicate arrival rates")
    return rates


def _validated_run(
    run_dir: Path,
) -> tuple[SourceRun, dict[str, Any], list[OpenLoopRateSummary]]:
    manifest_path = run_dir / "manifest.json"
    summary_path = run_dir / "summary.json"
    schedule_path = run_dir / "arrival_schedule.json"
    requests_path = run_dir / "requests.jsonl"
    resources_path = run_dir / "resource_samples.jsonl"
    manifest = _read_json(manifest_path)
    raw_summary = _read_json(summary_path)
    schedule = _read_json(schedule_path)
    if not isinstance(manifest, dict) or not isinstance(schedule, dict):
        raise ValueError("open-loop manifest and schedule must be JSON objects")
    if not isinstance(raw_summary, list) or not all(
        isinstance(item, dict) for item in raw_summary
    ):
        raise ValueError("open-loop summary must be a list of JSON objects")
    try:
        summary = [OpenLoopRateSummary.model_validate(item) for item in raw_summary]
    except ValueError as error:
        raise ValueError(f"invalid open-loop summary: {error}") from error
    if manifest.get("run_type") != "open-loop-load":
        raise ValueError("analysis requires open-loop-load runs")
    if manifest.get("artifact_schema_version") != "2":
        raise ValueError("analysis requires artifact_schema_version 2 runs")
    protocol_versions = {
        "scheduler_version": OPEN_LOOP_SCHEDULER_VERSION,
        "request_protocol_version": OPEN_LOOP_REQUEST_PROTOCOL_VERSION,
        "summary_version": OPEN_LOOP_SUMMARY_VERSION,
    }
    if any(manifest.get(key) != value for key, value in protocol_versions.items()):
        raise ValueError("open-loop run uses an unsupported measurement protocol")
    if _sha256(schedule_path) != manifest.get("arrival_schedule_sha256"):
        raise ValueError("arrival schedule digest does not match the manifest")
    if raw_summary != manifest.get("summary"):
        raise ValueError("summary artifact does not match the manifest")
    configured_rates = manifest["experiment"]["open_loop"]["arrival_rates_rps"]
    summary_rates = [item.offered_requests_per_second for item in summary]
    windows = schedule.get("windows")
    if not isinstance(windows, list):
        raise ValueError("arrival schedule does not contain rate windows")
    schedule_rates = [item["offered_requests_per_second"] for item in windows]
    if not (
        _rate_grid(configured_rates)
        == _rate_grid(summary_rates)
        == _rate_grid(schedule_rates)
    ):
        raise ValueError("open-loop run has an incomplete or different rate grid")
    seed = int(manifest["arrival_seed"])
    if seed != int(schedule["arrival_seed"]):
        raise ValueError("arrival seed differs between manifest and schedule")
    if manifest["arrival_process"] != schedule["arrival_process"]:
        raise ValueError("arrival process differs between manifest and schedule")
    if manifest["arrival_algorithm"] != schedule["algorithm"]:
        raise ValueError("arrival algorithm differs between manifest and schedule")
    open_loop_config = manifest["experiment"]["open_loop"]
    if manifest.get("rate_order_protocol") != open_loop_config.get(
        "rate_order_protocol"
    ) or int(manifest.get("rate_order_offset", -1)) != int(
        open_loop_config.get("rate_order_offset", -1)
    ):
        raise ValueError("rate order metadata differs from the experiment protocol")
    expected_schedule = build_arrival_schedule(
        OpenLoopSpec.model_validate(manifest["experiment"]["open_loop"])
    )
    if schedule != expected_schedule:
        raise ValueError("arrival schedule does not match the configured protocol")
    requests = [
        json.loads(line)
        for line in requests_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    resources = [
        json.loads(line)
        for line in resources_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    valid_rate_indices = set(range(len(windows)))
    for record in requests:
        rate_index = record.get("open_loop_rate_index")
        rate_request_index = record.get("rate_request_index")
        if (
            not isinstance(rate_index, int)
            or isinstance(rate_index, bool)
            or rate_index not in valid_rate_indices
            or not isinstance(rate_request_index, int)
            or isinstance(rate_request_index, bool)
            or rate_request_index < 0
        ):
            raise ValueError("request record has an invalid open-loop rate tag")
    for sample in resources:
        if "open_loop_rate_index" not in sample:
            continue
        rate_index = sample["open_loop_rate_index"]
        if (
            not isinstance(rate_index, int)
            or isinstance(rate_index, bool)
            or rate_index not in valid_rate_indices
        ):
            raise ValueError("resource sample has an invalid open-loop rate tag")
        rss = sample.get("process_tree_rss_bytes")
        if not isinstance(rss, int) or isinstance(rss, bool) or rss < 0:
            raise ValueError("tagged resource sample has invalid process RSS")
    rebuilt: list[OpenLoopRateSummary] = []
    for rate_index, (persisted, window) in enumerate(
        zip(summary, windows, strict=True)
    ):
        rate = persisted.offered_requests_per_second
        rate_records = [
            record
            for record in requests
            if int(record["open_loop_rate_index"]) == rate_index
        ]
        if any(
            float(record["offered_requests_per_second"]) != rate
            for record in rate_records
        ):
            raise ValueError("request rate identity differs from the rate window")
        scheduled_offsets = [
            int(record["scheduled_offset_ns"]) for record in rate_records
        ]
        if [int(record["rate_request_index"]) for record in rate_records] != list(
            range(len(rate_records))
        ):
            raise ValueError("request records have invalid within-rate indexes")
        if scheduled_offsets != [
            int(value) for value in window["scheduled_offsets_ns"]
        ]:
            raise ValueError("request records do not match the arrival schedule")
        rate_samples = [
            sample
            for sample in resources
            if sample.get("open_loop_rate_index") == rate_index
        ]
        duration_ns = max(
            int(window["arrival_window_ns"]),
            max(
                (int(record["client_completed_offset_ns"]) for record in rate_records),
                default=0,
            ),
        )
        rebuilt_value = summarize_rate(
            rate,
            duration_ns,
            rate_records,
            rate_samples,
            float(manifest["experiment"]["open_loop"]["latency_slo_ms"]),
            int(window["arrival_window_ns"]),
        )
        rebuilt.append(OpenLoopRateSummary.model_validate(rebuilt_value))
    if rebuilt != summary:
        raise ValueError(
            "summary metrics do not match raw request and resource artifacts"
        )
    model_sha256 = manifest["experiment"]["model"].get("sha256")
    server_sha256 = manifest["capabilities"]["llama_server"].get("sha256")
    if not model_sha256 or not server_sha256:
        raise ValueError("analysis requires model and llama-server fingerprints")
    source = SourceRun(
        repetition=0,
        run_id=str(manifest["run_id"]),
        arrival_seed=seed,
        manifest_sha256=_sha256(manifest_path),
        summary_sha256=_sha256(summary_path),
        arrival_schedule_sha256=_sha256(schedule_path),
        requests_sha256=_sha256(requests_path),
        resource_samples_sha256=_sha256(resources_path),
        rate_order_offset=int(manifest["rate_order_offset"]),
    )
    return source, manifest, summary


def _metric_estimate(
    values: list[float | int | None], unit: str, iterations: int, seed: int
) -> MetricEstimate:
    defined = [float(value) for value in values if value is not None]
    if len(defined) != len(values):
        return MetricEstimate(
            unit=unit,
            run_values=values,
            defined_runs=len(defined),
            mean=None,
            ci95_low=None,
            ci95_high=None,
        )
    low, high = bootstrap_run_mean_interval(defined, iterations=iterations, seed=seed)
    return MetricEstimate(
        unit=unit,
        run_values=values,
        defined_runs=len(defined),
        mean=statistics.fmean(defined),
        ci95_low=low,
        ci95_high=high,
    )


def _render_html(artifact: OpenLoopAnalysisArtifact) -> str:
    rows: list[str] = []
    for rate in artifact.rates:
        for name, estimate in rate.metrics.items():
            interval = (
                f"{estimate.mean:.3f} [{estimate.ci95_low:.3f}, "
                f"{estimate.ci95_high:.3f}]"
                if estimate.mean is not None
                and estimate.ci95_low is not None
                and estimate.ci95_high is not None
                else "n/a"
            )
            values = [
                f"{rate.offered_requests_per_second:.2f}",
                name,
                estimate.unit,
                interval,
                estimate.defined_runs,
            ]
            rows.append(
                "<tr>"
                + "".join(f"<td>{html.escape(str(value))}</td>" for value in values)
                + "</tr>"
            )
    seeds = ", ".join(str(source.arrival_seed) for source in artifact.source_runs)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Open-loop repeated-run analysis</title><style>
body {{ font-family: ui-monospace, monospace; margin: 2rem; background: #101417;
color: #e8efe9; }} h1 {{ color: #9be15d; }} table {{ border-collapse: collapse;
width: 100%; }} th, td {{ padding: .7rem; border-bottom: 1px solid #344038;
text-align: right; }} th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
th {{ color: #101417; background: #9be15d; }} .panel {{ overflow-x: auto; }}
</style></head><body><h1>Open-loop repeated-run analysis</h1>
<p>{artifact.repetition_count} independent fresh-server runs; arrival seeds:
{html.escape(seeds)}.</p><p>Intervals are deterministic 95% percentile bootstrap
intervals of arithmetic run means. The resampling unit is one complete run, not
an individual request.</p><div class="panel"><table><thead><tr><th>Offered req/s</th>
<th>Metric</th><th>Unit</th><th>Mean [95% CI]</th><th>Defined runs</th></tr>
</thead><tbody>{"".join(rows)}</tbody></table></div></body></html>"""


def analyze_open_loop_runs(
    run_dirs: list[Path],
    output_dir: Path,
    *,
    bootstrap_iterations: int = 10_000,
    bootstrap_seed: int = 42,
) -> OpenLoopAnalysisResult:
    """Validate and analyze completed independent open-loop runs."""
    if len(run_dirs) < MIN_INDEPENDENT_RUNS:
        raise ValueError(
            f"at least {MIN_INDEPENDENT_RUNS} independent open-loop runs are required"
        )
    if bootstrap_iterations <= 0 or bootstrap_seed < 0:
        raise ValueError("bootstrap iterations must be positive and seed non-negative")
    runs = [_validated_run(run_dir) for run_dir in run_dirs]
    runs.sort(key=lambda item: (item[0].arrival_seed, item[0].run_id))
    sources = [item[0] for item in runs]
    if len({source.run_id for source in sources}) != len(sources):
        raise ValueError("open-loop run IDs must be unique")
    manifests = [item[1] for item in runs]
    arrival_process = str(manifests[0]["arrival_process"])
    if arrival_process == ArrivalProcess.POISSON.value and len(
        {source.arrival_seed for source in sources}
    ) != len(sources):
        raise ValueError("independent Poisson runs require unique arrival seeds")
    order_protocol = str(manifests[0]["rate_order_protocol"])
    rate_count = len(manifests[0]["experiment"]["open_loop"]["arrival_rates_rps"])
    offsets = [source.rate_order_offset for source in sources]
    if rate_count > 1 and order_protocol != RateOrderProtocol.CYCLIC.value:
        raise ValueError("multi-rate analysis requires cyclic rate ordering")
    if order_protocol == RateOrderProtocol.CYCLIC.value and (
        len(sources) % rate_count != 0
        or any(
            offsets.count(offset) != len(sources) // rate_count
            for offset in range(rate_count)
        )
    ):
        raise ValueError("cyclic rate order is not exactly counterbalanced")
    controls = {
        "model": {
            str(manifest["experiment"]["model"]["sha256"]) for manifest in manifests
        },
        "protocol": {_protocol_sha256(manifest) for manifest in manifests},
        "hardware": {_hardware_sha256(manifest) for manifest in manifests},
        "server": {
            str(manifest["capabilities"]["llama_server"]["sha256"])
            for manifest in manifests
        },
        "dataset": {str(manifest["prompt_dataset_sha256"]) for manifest in manifests},
        "process": {str(manifest["arrival_process"]) for manifest in manifests},
        "algorithm": {str(manifest["arrival_algorithm"]) for manifest in manifests},
    }
    mismatched = [name for name, values in controls.items() if len(values) != 1]
    if mismatched:
        raise ValueError(f"open-loop runs differ in {', '.join(mismatched)}")
    for repetition, source in enumerate(sources):
        source.repetition = repetition
    rate_values = sorted(item.offered_requests_per_second for item in runs[0][2])
    estimates: list[RateEstimate] = []
    for rate in rate_values:
        metrics: dict[str, MetricEstimate] = {}
        for name, unit in _METRIC_UNITS.items():
            values = [
                getattr(
                    next(
                        item
                        for item in run[2]
                        if item.offered_requests_per_second == rate
                    ),
                    name,
                )
                for run in runs
            ]
            if not all(
                value is None or isinstance(value, int | float) for value in values
            ):
                raise ValueError(f"open-loop metric {name} is not numeric")
            metrics[name] = _metric_estimate(
                values, unit, bootstrap_iterations, bootstrap_seed
            )
        estimates.append(
            RateEstimate(offered_requests_per_second=rate, metrics=metrics)
        )
    first = manifests[0]
    artifact = OpenLoopAnalysisArtifact(
        repetition_count=len(runs),
        bootstrap=BootstrapMetadata(
            iterations=bootstrap_iterations, seed=bootstrap_seed
        ),
        model_sha256=next(iter(controls["model"])),
        protocol_sha256=next(iter(controls["protocol"])),
        hardware_sha256=next(iter(controls["hardware"])),
        llama_server_sha256=next(iter(controls["server"])),
        prompt_dataset_sha256=next(iter(controls["dataset"])),
        arrival_process=str(first["arrival_process"]),
        arrival_algorithm=str(first["arrival_algorithm"]),
        source_runs=sources,
        rates=estimates,
    )
    if output_dir.exists():
        raise FileExistsError(f"analysis output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.with_name(f".{output_dir.name}-{uuid.uuid4().hex[:8]}")
    temporary.mkdir()
    try:
        _write_json(temporary / "analysis.json", artifact.model_dump(mode="json"))
        (temporary / "analysis.html").write_text(
            _render_html(artifact), encoding="utf-8"
        )
        temporary.rename(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return OpenLoopAnalysisResult(output_dir=output_dir, artifact=artifact)


def run_repeated_open_loop_benchmark(
    experiment: ExperimentSpec,
) -> OpenLoopStudyResult:
    """Run and analyze independent open-loop repetitions sequentially."""
    if experiment.open_loop is None:
        raise ValueError("experiment does not define an open_loop section")
    if experiment.server is None:
        raise ValueError("experiment does not define a server section")
    load = experiment.open_loop
    if load.independent_runs < MIN_INDEPENDENT_RUNS:
        raise ValueError(
            f"repeated open-loop study requires at least {MIN_INDEPENDENT_RUNS} runs"
        )
    validate_model(experiment.model)
    if experiment.model.sha256 is None:
        raise ValueError("repeated open-loop study requires a model SHA-256")
    load_prompts(load.prompt_dataset)
    canonical_rates = sorted(float(value) for value in load.arrival_rates_rps)
    for repetition in range(int(load.independent_runs)):
        offset = repetition % len(canonical_rates)
        ordered_rates = canonical_rates[offset:] + canonical_rates[:offset]
        build_arrival_schedule(
            load.model_copy(
                update={
                    "arrival_seed": int(load.arrival_seed) + repetition,
                    "arrival_rates_rps": ordered_rates,
                    "rate_order_protocol": RateOrderProtocol.CYCLIC,
                    "rate_order_offset": offset,
                }
            )
        )
    capabilities = inspect_capabilities(
        llama_bench_binary=experiment.llama_bench_binary,
        llama_server_binary=experiment.llama_server_binary,
    )
    if not capabilities.llama_server.available or not capabilities.llama_server.sha256:
        raise ValueError(
            "repeated open-loop study requires a fingerprinted llama-server"
        )
    study_id = (
        f"{experiment.experiment_id}-open-loop-study-"
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    )
    study_dir = experiment.output_dir / study_id
    repetitions_dir = study_dir / "repetitions"
    repetitions_dir.mkdir(parents=True)
    completed: list[OpenLoopRunResult] = []
    failed_repetition = 0
    arrival_seed = int(load.arrival_seed)
    try:
        for repetition in range(int(load.independent_runs)):
            failed_repetition = repetition
            arrival_seed = int(load.arrival_seed) + repetition
            offset = repetition % len(canonical_rates)
            ordered_rates = canonical_rates[offset:] + canonical_rates[:offset]
            child_load = load.model_copy(
                update={
                    "arrival_seed": arrival_seed,
                    "arrival_rates_rps": ordered_rates,
                    "independent_runs": 1,
                    "rate_order_protocol": RateOrderProtocol.CYCLIC,
                    "rate_order_offset": offset,
                }
            )
            child_experiment = experiment.model_copy(
                update={"output_dir": repetitions_dir, "open_loop": child_load}
            )
            result = run_open_loop_benchmark(child_experiment)
            if len(result.summary) != len(load.arrival_rates_rps) or (
                experiment.fail_fast and result.failed_requests > 0
            ):
                raise RuntimeError("open-loop repetition did not complete every rate")
            completed.append(result)
        analysis = analyze_open_loop_runs(
            [result.run_dir for result in completed],
            study_dir / "analysis",
            bootstrap_iterations=int(load.bootstrap_iterations),
            bootstrap_seed=int(load.bootstrap_seed),
        )
    except Exception as error:
        with suppress(OSError):
            _write_json(
                study_dir / "failure.json",
                {
                    "artifact_schema_version": "1",
                    "status": "failed",
                    "study_id": study_id,
                    "failed_repetition": failed_repetition,
                    "arrival_seed": arrival_seed,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "completed_run_ids": [result.run_id for result in completed],
                },
            )
        raise
    return OpenLoopStudyResult(
        study_id=study_id,
        study_dir=study_dir,
        repetition_run_ids=[result.run_id for result in completed],
        analysis_dir=analysis.output_dir,
        analysis=analysis.artifact,
    )
