"""Server-agnostic multi-agent prefix-sharing benchmark."""

import concurrent.futures
import json
import platform
import random
import statistics
import subprocess
import time
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator

from localllm_bench.load import percentile
from localllm_bench.logging import get_logger
from localllm_bench.open_loop import load_prompts, prompt_dataset_sha256
from localllm_bench.server import (
    _read_stream,
    available_port,
    stop_server,
    wait_until_ready,
)

LOGGER = get_logger(__file__)
FOCI = ("generation", "transmission", "storage", "markets", "reliability")
CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
HEALTH_PATH = "/health"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 120
IN_BAND_ERROR_PREFIX = "[ERROR"
NANOSECONDS_PER_MILLISECOND = 1_000_000
NANOSECONDS_PER_SECOND = 1_000_000_000
RESULTS_TABLE_HEADER = (
    "| target | phase | n | TTFT p50 ms | TTFT p95 ms | rep p50 range | "
    "E2E p50 ms | speedup vs baseline |"
)
WALL_CLOCK_TABLE_HEADER = (
    "| target | concurrency | median rep wall s | requests/s | cached-token ratio |"
)


class PrefixSharingWorkload(BaseModel):
    """Deterministic multi-agent conversation workload."""

    model_config = ConfigDict(extra="forbid")

    agents: PositiveInt
    shared_prefix_words: PositiveInt
    turns: PositiveInt
    max_tokens: PositiveInt
    temperature: float = 0.0
    concurrency: list[PositiveInt] = Field(min_length=1)
    repetitions: PositiveInt
    corpus_path: Path
    prompts_path: Path
    agent_id_header: str | None = None

    @model_validator(mode="after")
    def validate_concurrency(self) -> "PrefixSharingWorkload":
        """Require unique, increasing concurrency levels."""
        levels = [int(value) for value in self.concurrency]
        if levels != sorted(set(levels)):
            raise ValueError("concurrency must be unique and increasing")
        return self


class PrefixSharingTarget(BaseModel):
    """One OpenAI-compatible server implementation to benchmark."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    command: list[str] = Field(min_length=1)
    stats_path: str | None = None
    startup_timeout_seconds: PositiveInt = 120
    baseline: bool = False


class PrefixSharingConfig(BaseModel):
    """Top-level configuration for a prefix-sharing benchmark."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    seed: int
    output_dir: Path = Path("runs")
    model_path: Path
    workload: PrefixSharingWorkload
    targets: list[PrefixSharingTarget] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_baseline(self) -> "PrefixSharingConfig":
        """Require one and only one comparison baseline."""
        if sum(target.baseline for target in self.targets) != 1:
            raise ValueError("exactly one target must set baseline: true")
        names = [target.name for target in self.targets]
        if len(names) != len(set(names)):
            raise ValueError("target names must be unique")
        return self


class PrefixSharingRunResult(BaseModel):
    """Location and aggregate counts for a prefix-sharing run."""

    run_dir: Path
    completed_requests: int
    failed_requests: int
    summary: dict[str, Any]


@dataclass(frozen=True)
class PreparedWorkload:
    """Fully materialized deterministic prompts for all agent turns."""

    systems: list[str]
    user_prompts: list[list[str]]


def _resolve_path(path: Path, config_dir: Path) -> Path:
    """Resolve a configured filesystem path relative to its YAML document."""
    return path if path.is_absolute() else (config_dir / path).resolve()


def _resolve_command(
    command: list[str], config_dir: Path, model_path: Path
) -> list[str]:
    """Substitute command placeholders and resolve slash-containing executables."""
    resolved: list[str] = []
    for index, argument in enumerate(command):
        value = argument.replace("{model}", str(model_path))
        if index == 0 and "/" in value and not Path(value).is_absolute():
            value = str((config_dir / value).resolve())
        resolved.append(value)
    return resolved


def load_prefix_sharing_config(path: Path) -> PrefixSharingConfig:
    """Load, validate, and resolve a prefix-sharing YAML configuration.

    Parameters
    ----------
    path
        Path to the YAML configuration document.

    Returns
    -------
    PrefixSharingConfig
        Validated configuration with input paths resolved against the document.
    """
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    config = PrefixSharingConfig.model_validate(value)
    config_dir = path.parent.resolve()
    workload = config.workload.model_copy(
        update={
            "corpus_path": _resolve_path(config.workload.corpus_path, config_dir),
            "prompts_path": _resolve_path(config.workload.prompts_path, config_dir),
        }
    )
    model_path = _resolve_path(config.model_path, config_dir)
    targets = [
        target.model_copy(
            update={"command": _resolve_command(target.command, config_dir, model_path)}
        )
        for target in config.targets
    ]
    return config.model_copy(
        update={
            "model_path": model_path,
            "workload": workload,
            "targets": targets,
        }
    )


def build_shared_prompt(corpus: str, shared_prefix_words: int) -> str:
    """Build a repeated corpus prefix with a fixed team instruction."""
    words = corpus.split()
    if not words:
        raise ValueError("prefix-sharing corpus is empty")
    selected = [words[index % len(words)] for index in range(shared_prefix_words)]
    instruction = (
        "You are part of a team of energy-market analyst agents. "
        "Use the reference material below."
    )
    return f"{instruction}\n\n{' '.join(selected)}"


def prepare_workload(config: PrefixSharingConfig) -> PreparedWorkload:
    """Materialize deterministic systems and user prompts from a configuration."""
    workload = config.workload
    shared_prompt = build_shared_prompt(
        workload.corpus_path.read_text(encoding="utf-8"),
        int(workload.shared_prefix_words),
    )
    prompts = load_prompts(workload.prompts_path)
    chooser = random.Random(config.seed)
    systems = [
        f"{shared_prompt}\n\nAgent profile: agent-{agent}, "
        f"focus: {FOCI[agent % len(FOCI)]}."
        for agent in range(int(workload.agents))
    ]
    user_prompts = [
        [chooser.choice(prompts).prompt for _ in range(int(workload.turns))]
        for _ in range(int(workload.agents))
    ]
    return PreparedWorkload(systems=systems, user_prompts=user_prompts)


def phase_for_request(concurrency: int, turn: int, agent: int) -> str:
    """Return the comparison phase for one request in dispatch order."""
    if turn > 0:
        return "continuation"
    if concurrency == 1 and agent == 0:
        return "cold"
    return "fanout"


def _chat_completion(
    base_url: str,
    target_name: str,
    messages: list[dict[str, str]],
    config: PrefixSharingConfig,
    agent: int | None = None,
) -> dict[str, Any]:
    """Send one generic OpenAI-compatible streaming chat request.

    When ``workload.agent_id_header`` is set and ``agent`` is given, the request
    carries that header with value ``agent-<agent>`` (e.g. ``X-Agent-Id`` for
    per-agent memory namespaces in a memory proxy).
    """
    payload = {
        "model": target_name,
        "messages": messages,
        "max_tokens": int(config.workload.max_tokens),
        "temperature": config.workload.temperature,
        "seed": config.seed,
        "stream": True,
    }
    request = Request(
        f"{base_url}{CHAT_COMPLETIONS_PATH}",
        data=json.dumps(payload).encode("utf-8"),
        headers=_request_headers(config.workload.agent_id_header, agent),
        method="POST",
    )
    started_ns = time.monotonic_ns()
    with urlopen(request, timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS) as response:
        measurement = _read_stream(response, started_ns)
    if measurement["output_tokens"] is None:
        measurement["output_tokens"] = int(measurement["content_chunk_count"])
    validate_completion(str(measurement.get("response_text", "")))
    return measurement


def _request_headers(agent_id_header: str | None, agent: int | None) -> dict[str, str]:
    """Build request headers, optionally identifying the calling agent."""
    headers = {"Content-Type": "application/json"}
    if agent_id_header is not None and agent is not None:
        headers[agent_id_header] = f"agent-{agent}"
    return headers


def validate_completion(text: str) -> None:
    """Reject completions that signal a server-side failure.

    Some servers (RadixForge) report errors as streamed content instead of an
    HTTP error; an empty completion is also treated as a failure so a server
    that aborts prefill cannot be reported as fast.

    Parameters
    ----------
    text : str
        Concatenated streamed content of one completion.

    Raises
    ------
    ValueError
        If the text is empty or is an in-band error marker.
    """
    stripped = text.strip()
    if not stripped:
        raise ValueError("empty completion")
    if stripped.startswith(IN_BAND_ERROR_PREFIX):
        raise ValueError(f"in-band server error: {stripped[:120]}")


def _messages_for_turn(
    system: str, user_prompts: list[str], assistant_texts: list[str], turn: int
) -> list[dict[str, str]]:
    """Build chat history ending in the specified user turn."""
    messages = [{"role": "system", "content": system}]
    for previous_turn in range(turn):
        messages.extend(
            [
                {"role": "user", "content": user_prompts[previous_turn]},
                {"role": "assistant", "content": assistant_texts[previous_turn]},
            ]
        )
    messages.append({"role": "user", "content": user_prompts[turn]})
    return messages


def _write_json(path: Path, value: Any) -> None:
    """Write reproducible pretty JSON with a trailing newline."""
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _safe_stats_snapshot(base_url: str, stats_path: str) -> dict[str, Any]:
    """Fetch one optional server stats document."""
    with urlopen(
        f"{base_url}{stats_path}", timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS
    ) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise ValueError("stats endpoint returned a non-object JSON document")
    return payload


def _run_repetition(
    config: PrefixSharingConfig,
    target: PrefixSharingTarget,
    concurrency: int,
    repetition: int,
    prepared: PreparedWorkload,
    run_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any] | None, list[str]]:
    """Launch one fresh server and execute all turns for one benchmark cell."""
    port = available_port()
    command = [argument.replace("{port}", str(port)) for argument in target.command]
    log_path = run_dir / "logs" / f"{target.name}-c{concurrency}-r{repetition}.log"
    records: list[dict[str, Any]] = []
    resolved = {
        "target": target.name,
        "concurrency": concurrency,
        "repetition": repetition,
        "command": command,
    }
    status: dict[str, Any] = {
        "target": target.name,
        "concurrency": concurrency,
        "repetition": repetition,
        "status": "failed",
        "wall_ns": 0,
    }
    stats: dict[str, Any] | None = None
    with log_path.open("w", encoding="utf-8") as log:
        process: subprocess.Popen[str] | None = None
        try:
            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, text=True
            )
            base_url = f"http://127.0.0.1:{port}"
            wait_until_ready(base_url, process, int(target.startup_timeout_seconds))
            try:
                _chat_completion(
                    base_url,
                    target.name,
                    [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Say OK."},
                    ],
                    config.model_copy(
                        update={
                            "workload": config.workload.model_copy(
                                update={"max_tokens": 4}
                            )
                        }
                    ),
                )
            except (HTTPError, OSError, TimeoutError, ValueError) as error:
                LOGGER.warning("Warmup failed for %s: %s", target.name, error)
            assistant_texts: list[list[str]] = [[] for _ in prepared.systems]
            turns_started_ns = time.monotonic_ns()
            for turn in range(int(config.workload.turns)):

                def worker(agent: int, request_turn: int = turn) -> dict[str, Any]:
                    messages = _messages_for_turn(
                        prepared.systems[agent],
                        prepared.user_prompts[agent],
                        assistant_texts[agent],
                        request_turn,
                    )
                    record: dict[str, Any] = {
                        "target": target.name,
                        "concurrency": concurrency,
                        "repetition": repetition,
                        "turn": request_turn,
                        "agent": agent,
                        "phase": phase_for_request(concurrency, request_turn, agent),
                        "prompt_chars": sum(
                            len(message["content"]) for message in messages
                        ),
                        "backend_timings": {},
                        "ttft_ns": None,
                        "e2e_latency_ns": None,
                        "output_tokens": 0,
                        "error": None,
                    }
                    try:
                        measurement = _chat_completion(
                            base_url, target.name, messages, config, agent
                        )
                        record.update(measurement)
                    except (HTTPError, OSError, TimeoutError, ValueError) as error:
                        record["error"] = str(error)
                    return record

                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=concurrency
                ) as executor:
                    futures = [
                        executor.submit(worker, agent)
                        for agent in range(len(prepared.systems))
                    ]
                    turn_records = [future.result() for future in futures]
                for record in sorted(turn_records, key=lambda item: int(item["agent"])):
                    if record["error"] is None:
                        text = str(record.pop("response_text"))
                        record["response_chars"] = len(text)
                        assistant_texts[int(record["agent"])].append(text)
                    else:
                        assistant_texts[int(record["agent"])].append("")
                    records.append(record)
            status.update(
                {"status": "ok", "wall_ns": time.monotonic_ns() - turns_started_ns}
            )
            if target.stats_path is not None:
                try:
                    stats = _safe_stats_snapshot(base_url, target.stats_path)
                except (HTTPError, OSError, TimeoutError, ValueError) as error:
                    stats = {"error": str(error)}
        except (OSError, RuntimeError, TimeoutError) as error:
            status["error"] = str(error)
            LOGGER.warning(
                "Server startup failed for %s c=%s r=%s: %s",
                target.name,
                concurrency,
                repetition,
                error,
            )
        finally:
            if process is not None:
                stop_server(process)
    return records, status, stats, [json.dumps(resolved, sort_keys=True)]


def _numeric_timing(record: dict[str, Any], name: str) -> float | None:
    """Return a nanosecond timing converted to milliseconds when present."""
    value = record.get(name)
    return (
        float(value) / NANOSECONDS_PER_MILLISECOND if isinstance(value, int) else None
    )


def _prompt_cache_values(records: list[dict[str, Any]]) -> list[tuple[float, float]]:
    """Extract reported prompt and cache token counts from successful records."""
    values: list[tuple[float, float]] = []
    for record in records:
        timings = record.get("backend_timings")
        if not isinstance(timings, dict):
            continue
        prompt = timings.get("prompt_n")
        cache = timings.get("cache_n")
        if isinstance(prompt, (int, float)) and isinstance(cache, (int, float)):
            values.append((float(prompt), float(cache)))
    return values


def summarize_prefix_sharing(
    records: list[dict[str, Any]], repetitions: list[dict[str, Any]], baseline: str
) -> dict[str, Any]:
    """Aggregate pooled request timings, repetition variation, and cache reuse."""
    groups: list[dict[str, Any]] = []
    keys = sorted(
        {
            (str(record["target"]), int(record["concurrency"]), str(record["phase"]))
            for record in records
        }
    )
    for target, concurrency, phase in keys:
        group_records = [
            record
            for record in records
            if (record["target"], record["concurrency"], record["phase"])
            == (target, concurrency, phase)
        ]
        successful = [record for record in group_records if record.get("error") is None]
        ttft_ms = [
            value
            for record in successful
            if (value := _numeric_timing(record, "ttft_ns")) is not None
        ]
        e2e_ms = [
            value
            for record in successful
            if (value := _numeric_timing(record, "e2e_latency_ns")) is not None
        ]
        rep_p50s = [
            statistics.median(values)
            for repetition in sorted({int(item["repetition"]) for item in successful})
            if (
                values := [
                    value
                    for item in successful
                    if int(item["repetition"]) == repetition
                    and (value := _numeric_timing(item, "ttft_ns")) is not None
                ]
            )
        ]
        groups.append(
            {
                "target": target,
                "concurrency": concurrency,
                "phase": phase,
                "n_ok": len(successful),
                "n_err": len(group_records) - len(successful),
                "ttft_p50_ms": percentile(ttft_ms, 50) if ttft_ms else 0.0,
                "ttft_p95_ms": percentile(ttft_ms, 95) if ttft_ms else 0.0,
                "e2e_p50_ms": percentile(e2e_ms, 50) if e2e_ms else 0.0,
                "rep_ttft_p50_median_ms": statistics.median(rep_p50s)
                if rep_p50s
                else 0.0,
                "rep_ttft_p50_min_ms": min(rep_p50s, default=0.0),
                "rep_ttft_p50_max_ms": max(rep_p50s, default=0.0),
            }
        )
    baseline_ttft = {
        (int(group["concurrency"]), str(group["phase"])): float(group["ttft_p50_ms"])
        for group in groups
        if group["target"] == baseline
    }
    for group in groups:
        base = baseline_ttft.get((int(group["concurrency"]), str(group["phase"])), 0.0)
        timing = float(group["ttft_p50_ms"])
        group["speedup_vs_baseline"] = base / timing if base > 0 and timing > 0 else 0.0

    wall_clock: list[dict[str, Any]] = []
    for target, concurrency in sorted(
        {(item["target"], item["concurrency"]) for item in repetitions}
    ):
        cells = [
            item
            for item in repetitions
            if item["target"] == target
            and item["concurrency"] == concurrency
            and item["status"] == "ok"
        ]
        wall_values = [
            int(item["wall_ns"]) for item in cells if int(item["wall_ns"]) > 0
        ]
        cell_records = [
            record
            for record in records
            if record["target"] == target
            and record["concurrency"] == concurrency
            and record.get("error") is None
        ]
        prompt_cache = _prompt_cache_values(cell_records)
        denominator = sum(prompt + cache for prompt, cache in prompt_cache)
        total_wall_ns = sum(wall_values)
        wall_clock.append(
            {
                "target": target,
                "concurrency": concurrency,
                "median_rep_wall_seconds": (
                    statistics.median(wall_values) / NANOSECONDS_PER_SECOND
                    if wall_values
                    else 0.0
                ),
                "requests_per_second": (
                    len(cell_records) * NANOSECONDS_PER_SECOND / total_wall_ns
                    if total_wall_ns > 0
                    else 0.0
                ),
                "cached_token_ratio": (
                    sum(cache for _, cache in prompt_cache) / denominator
                    if denominator > 0
                    else 0.0
                ),
            }
        )
    return {"groups": groups, "wall_clock": wall_clock, "repetitions": repetitions}


def render_prefix_sharing_report(
    config: PrefixSharingConfig, summary: dict[str, Any]
) -> str:
    """Render a deterministic Markdown report for one completed benchmark."""
    cpu = "unavailable"
    with suppress(OSError, subprocess.SubprocessError):
        cpu = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True, timeout=2
        ).strip()
    model_size = config.model_path.stat().st_size if config.model_path.exists() else 0
    workload = config.workload
    lines = [
        "# Prefix-sharing multi-agent benchmark",
        "",
        "## Environment",
        "",
        f"- Platform: {platform.platform()}",
        f"- CPU: {cpu}",
        f"- Model: {config.model_path.name} ({model_size} bytes)",
        "",
        "## Workload",
        "",
        (
            f"- Agents: {workload.agents}; turns: {workload.turns}; "
            f"repetitions: {workload.repetitions}"
        ),
        (
            f"- Shared prefix words: {workload.shared_prefix_words}; "
            f"max tokens: {workload.max_tokens}"
        ),
        f"- Concurrency: {', '.join(str(value) for value in workload.concurrency)}",
    ]
    groups = list(summary["groups"])
    for concurrency in workload.concurrency:
        lines.extend(
            [
                "",
                f"## Concurrency {concurrency}",
                "",
                RESULTS_TABLE_HEADER,
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for group in (item for item in groups if item["concurrency"] == concurrency):
            range_text = (
                f"{group['rep_ttft_p50_min_ms']:.2f}–{group['rep_ttft_p50_max_ms']:.2f}"
            )
            lines.append(
                f"| {group['target']} | {group['phase']} | "
                f"{group['n_ok']}/{group['n_err']} | {group['ttft_p50_ms']:.2f} | "
                f"{group['ttft_p95_ms']:.2f} | {range_text} | "
                f"{group['e2e_p50_ms']:.2f} | {group['speedup_vs_baseline']:.2f} |"
            )
    lines.extend(
        [
            "",
            "## Wall clock",
            "",
            WALL_CLOCK_TABLE_HEADER,
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["wall_clock"]:
        lines.append(
            f"| {row['target']} | {row['concurrency']} | "
            f"{row['median_rep_wall_seconds']:.2f} | "
            f"{row['requests_per_second']:.2f} | {row['cached_token_ratio']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def _git_sha() -> str | None:
    """Return the current repository revision when Git is available."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, timeout=2
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def run_prefix_sharing(config: PrefixSharingConfig) -> PrefixSharingRunResult:
    """Execute all fresh-server prefix-sharing benchmark repetitions."""
    prepared = prepare_workload(config)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = config.output_dir / f"{config.name}-{timestamp}"
    (run_dir / "logs").mkdir(parents=True)
    manifest: dict[str, Any] = {
        "config": config.model_dump(mode="json"),
        "corpus_sha256": prompt_dataset_sha256(config.workload.corpus_path),
        "prompts_sha256": prompt_dataset_sha256(config.workload.prompts_path),
        "started_at": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "platform": platform.platform(),
        "resolved_commands": [],
    }
    manifest_path = run_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    records: list[dict[str, Any]] = []
    repetitions: list[dict[str, Any]] = []
    stats_records: list[dict[str, Any]] = []
    for target in config.targets:
        for concurrency in config.workload.concurrency:
            for repetition in range(int(config.workload.repetitions)):
                cell_records, status, stats, resolved = _run_repetition(
                    config, target, int(concurrency), repetition, prepared, run_dir
                )
                records.extend(cell_records)
                repetitions.append(status)
                manifest["resolved_commands"].extend(
                    json.loads(item) for item in resolved
                )
                if stats is not None:
                    stats_records.append(
                        {
                            "target": target.name,
                            "concurrency": int(concurrency),
                            "repetition": repetition,
                            "stats": stats,
                        }
                    )
    with (run_dir / "requests.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    with (run_dir / "repetitions.jsonl").open("w", encoding="utf-8") as stream:
        for repetition_record in repetitions:
            stream.write(json.dumps(repetition_record, sort_keys=True) + "\n")
    if stats_records:
        with (run_dir / "stats.jsonl").open("w", encoding="utf-8") as stream:
            for stats in stats_records:
                stream.write(json.dumps(stats, sort_keys=True) + "\n")
    baseline = next(target.name for target in config.targets if target.baseline)
    summary = summarize_prefix_sharing(records, repetitions, baseline)
    _write_json(run_dir / "summary.json", summary)
    (run_dir / "report.md").write_text(
        render_prefix_sharing_report(config, summary), encoding="utf-8"
    )
    _write_json(manifest_path, manifest)
    completed = sum(record.get("error") is None for record in records)
    return PrefixSharingRunResult(
        run_dir=run_dir,
        completed_requests=completed,
        failed_requests=len(records) - completed,
        summary=summary,
    )
