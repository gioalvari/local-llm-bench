"""Paired factorial analysis for llama.cpp microbenchmark runs."""

import hashlib
import html
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

Phase = Literal["prompt", "generation"]

_FACTOR_LABELS = {
    "gpu_layers": "All vs zero model layers offloaded",
    "flash_attention": "Flash Attention on vs off",
    "batch_size": "Larger vs smaller batch",
}


class FactorEffect(BaseModel):
    """Paired multiplicative effect for one phase and factor."""

    factor: str
    label: str
    phase: Phase
    reference: str
    treatment: str
    pair_count: int
    geometric_mean_ratio: float
    effect_percent: float
    median_ratio: float
    minimum_ratio: float
    maximum_ratio: float
    bootstrap_low_ratio: float
    bootstrap_high_ratio: float


class ParetoPoint(BaseModel):
    """A non-dominated speed and process-RSS configuration."""

    workload_name: str
    batch_size: int
    ubatch_size: int
    threads: int
    gpu_layers: int
    flash_attention: str
    prompt_tokens_per_second: float
    generation_tokens_per_second: float
    peak_process_tree_rss_bytes: int


class FactorAnalysisResult(BaseModel):
    """Portable output from one microbenchmark factor analysis."""

    run_dir: Path
    output_dir: Path
    quantization: str
    model_sha256: str
    manifest_sha256: str
    measurements_sha256: str
    effects: list[FactorEffect]
    pareto_frontier: list[ParetoPoint]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _phase(observation: dict[str, Any]) -> Phase:
    metrics = observation["metrics"]
    prompt_tokens = int(metrics["n_prompt"])
    generation_tokens = int(metrics["n_gen"])
    if prompt_tokens > 0 and generation_tokens == 0:
        return "prompt"
    if generation_tokens > 0 and prompt_tokens == 0:
        return "generation"
    raise ValueError("measurement must contain exactly one benchmark phase")


def _linear_percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def bootstrap_geometric_mean_interval(
    ratios: list[float],
    *,
    iterations: int = 10_000,
    seed: int = 42,
) -> tuple[float, float]:
    """Return a deterministic percentile interval for a geometric mean."""
    if not ratios or any(ratio <= 0 for ratio in ratios):
        raise ValueError("bootstrap ratios must be positive and non-empty")
    if iterations <= 0:
        raise ValueError("bootstrap iterations must be positive")
    generator = random.Random(seed)
    log_ratios = [math.log(ratio) for ratio in ratios]
    estimates = [
        math.exp(
            statistics.fmean(
                log_ratios[generator.randrange(len(log_ratios))] for _ in log_ratios
            )
        )
        for _ in range(iterations)
    ]
    return (
        _linear_percentile(estimates, 0.025),
        _linear_percentile(estimates, 0.975),
    )


def _pair_key(
    observation: dict[str, Any], factor: str
) -> tuple[tuple[str, object], ...]:
    cell = observation["cell"]
    fields = (
        "workload_name",
        "prompt_tokens",
        "generation_tokens",
        "batch_size",
        "ubatch_size",
        "threads",
        "gpu_layers",
        "flash_attention",
    )
    values = [(field, cell[field]) for field in fields if field != factor]
    values.append(("phase", _phase(observation)))
    return tuple(values)


def _factor_levels(
    observations: list[dict[str, Any]], factor: str
) -> tuple[object, object]:
    if factor not in _FACTOR_LABELS:
        raise ValueError(f"unsupported factor: {factor}")
    levels = {observation["cell"][factor] for observation in observations}
    if factor == "gpu_layers":
        expected: tuple[object, object] = (0, -1)
    elif factor == "flash_attention":
        expected = ("off", "on")
    else:
        ordered = sorted(int(level) for level in levels)
        if len(ordered) != 2:
            raise ValueError(f"factor {factor} must have exactly two levels")
        expected = (ordered[0], ordered[1])
    if levels != set(expected):
        raise ValueError(f"factor {factor} must have levels {expected}")
    return expected


def paired_ratios(
    observations: list[dict[str, Any]],
    factor: str,
    phase: Phase,
) -> list[float]:
    """Pair treatment and reference cells while holding all else constant."""
    reference, treatment = _factor_levels(observations, factor)
    grouped: dict[tuple[tuple[str, object], ...], dict[object, float]] = {}
    for observation in observations:
        if _phase(observation) != phase:
            continue
        level = observation["cell"][factor]
        if level not in (reference, treatment):
            raise ValueError(f"unexpected {factor} level: {level}")
        key = _pair_key(observation, factor)
        levels = grouped.setdefault(key, {})
        if level in levels:
            raise ValueError(f"duplicate {factor} level for a matched cell")
        levels[level] = float(observation["metrics"]["avg_ts"])
    if not grouped:
        raise ValueError(f"no {phase} observations for factor {factor}")
    incomplete = [key for key, levels in grouped.items() if len(levels) != 2]
    if incomplete:
        raise ValueError(
            f"incomplete paired cells for factor {factor} and phase {phase}"
        )
    return [levels[treatment] / levels[reference] for levels in grouped.values()]


def analyze_effects(observations: list[dict[str, Any]]) -> list[FactorEffect]:
    """Calculate paired factor effects separately for each benchmark phase."""
    effects: list[FactorEffect] = []
    for factor in _FACTOR_LABELS:
        reference, treatment = _factor_levels(observations, factor)
        for phase in ("prompt", "generation"):
            ratios = paired_ratios(observations, factor, phase)
            geometric_mean = statistics.geometric_mean(ratios)
            low, high = bootstrap_geometric_mean_interval(ratios)
            effects.append(
                FactorEffect(
                    factor=factor,
                    label=_FACTOR_LABELS[factor],
                    phase=phase,
                    reference=str(reference),
                    treatment=str(treatment),
                    pair_count=len(ratios),
                    geometric_mean_ratio=geometric_mean,
                    effect_percent=(geometric_mean - 1) * 100,
                    median_ratio=statistics.median(ratios),
                    minimum_ratio=min(ratios),
                    maximum_ratio=max(ratios),
                    bootstrap_low_ratio=low,
                    bootstrap_high_ratio=high,
                )
            )
    return effects


def _configuration_points(observations: list[dict[str, Any]]) -> list[ParetoPoint]:
    grouped: dict[str, dict[str, Any]] = {}
    for observation in observations:
        cell = observation["cell"]
        point = grouped.setdefault(
            str(cell["cell_id"]),
            {
                "cell": cell,
                "rss": int(observation["peak_process_tree_rss_bytes"]),
            },
        )
        point["rss"] = max(
            int(point["rss"]), int(observation["peak_process_tree_rss_bytes"])
        )
        point[_phase(observation)] = float(observation["metrics"]["avg_ts"])
    points: list[ParetoPoint] = []
    for point in grouped.values():
        if "prompt" not in point or "generation" not in point:
            raise ValueError("configuration lacks prompt or generation measurement")
        cell = point["cell"]
        points.append(
            ParetoPoint(
                workload_name=str(cell["workload_name"]),
                batch_size=int(cell["batch_size"]),
                ubatch_size=int(cell["ubatch_size"]),
                threads=int(cell["threads"]),
                gpu_layers=int(cell["gpu_layers"]),
                flash_attention=str(cell["flash_attention"]),
                prompt_tokens_per_second=float(point["prompt"]),
                generation_tokens_per_second=float(point["generation"]),
                peak_process_tree_rss_bytes=int(point["rss"]),
            )
        )
    return points


def _dominates(candidate: ParetoPoint, point: ParetoPoint) -> bool:
    no_worse = (
        candidate.prompt_tokens_per_second >= point.prompt_tokens_per_second
        and candidate.generation_tokens_per_second >= point.generation_tokens_per_second
        and candidate.peak_process_tree_rss_bytes <= point.peak_process_tree_rss_bytes
    )
    strictly_better = (
        candidate.prompt_tokens_per_second > point.prompt_tokens_per_second
        or candidate.generation_tokens_per_second > point.generation_tokens_per_second
        or candidate.peak_process_tree_rss_bytes < point.peak_process_tree_rss_bytes
    )
    return no_worse and strictly_better


def pareto_frontier(observations: list[dict[str, Any]]) -> list[ParetoPoint]:
    """Return non-dominated speed/process-RSS points within each workload."""
    points = _configuration_points(observations)
    frontier = [
        point
        for point in points
        if not any(
            candidate.workload_name == point.workload_name
            and _dominates(candidate, point)
            for candidate in points
            if candidate is not point
        )
    ]
    return sorted(
        frontier,
        key=lambda point: (
            point.workload_name,
            point.peak_process_tree_rss_bytes,
            -point.prompt_tokens_per_second,
        ),
    )


def _render_html(
    quantization: str,
    effects: list[FactorEffect],
    frontier: list[ParetoPoint],
) -> str:
    effect_rows = []
    for effect in effects:
        values = [
            effect.label,
            effect.phase,
            effect.pair_count,
            f"{effect.geometric_mean_ratio:.3f}x",
            f"{effect.effect_percent:+.1f}%",
            f"{effect.bootstrap_low_ratio:.3f}x",
            f"{effect.bootstrap_high_ratio:.3f}x",
            f"{effect.minimum_ratio:.3f}x",
            f"{effect.maximum_ratio:.3f}x",
        ]
        effect_rows.append(
            "<tr>"
            + "".join(f"<td>{html.escape(str(value))}</td>" for value in values)
            + "</tr>"
        )
    pareto_rows = []
    for point in frontier:
        values = [
            point.workload_name,
            point.batch_size,
            point.gpu_layers,
            point.flash_attention,
            f"{point.prompt_tokens_per_second:.2f}",
            f"{point.generation_tokens_per_second:.2f}",
            f"{point.peak_process_tree_rss_bytes / (1024**2):.1f}",
        ]
        pareto_rows.append(
            "<tr>"
            + "".join(f"<td>{html.escape(str(value))}</td>" for value in values)
            + "</tr>"
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LocalLLM Bench factor analysis</title>
<style>body {{ font-family: ui-monospace, monospace; margin: 2rem;
background: #101417; color: #e8efe9; }} h1, h2 {{ color: #9be15d; }}
table {{ border-collapse: collapse; width: 100%; margin-bottom: 2rem; }}
th, td {{ padding: .7rem; border-bottom: 1px solid #344038; text-align: right; }}
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
th {{ color: #101417; background: #9be15d; }} .panel {{ overflow-x: auto; }}
</style></head><body><h1>{html.escape(quantization)} factor analysis</h1>
<p>Paired cell effects. Bootstrap intervals are descriptive, not population CIs.</p>
<div class="panel"><table><thead><tr><th>Factor</th><th>Phase</th><th>Pairs</th>
<th>Geomean ratio</th><th>Effect</th><th>Bootstrap low</th><th>Bootstrap high</th>
<th>Minimum</th><th>Maximum</th></tr></thead>
<tbody>{"".join(effect_rows)}</tbody></table></div>
<h2>Process-RSS Pareto frontier</h2>
<p>Apple unified-memory allocations are not fully represented by process RSS.</p>
<div class="panel"><table><thead><tr><th>Workload</th><th>Batch</th>
<th>GPU layers</th><th>Flash Attention</th><th>Prompt token/s</th>
<th>Generation token/s</th><th>Peak RSS MiB</th></tr></thead>
<tbody>{"".join(pareto_rows)}</tbody></table></div></body></html>"""


def analyze_factor_run(run_dir: Path, output_dir: Path) -> FactorAnalysisResult:
    """Analyze a complete two-level microbenchmark factorial run."""
    manifest_path = run_dir / "manifest.json"
    measurements_path = run_dir / "measurements.jsonl"
    manifest = _read_json(manifest_path)
    if manifest.get("run_type") is not None:
        raise ValueError("factor analysis requires a llama-bench microbenchmark run")
    observations = _read_jsonl(measurements_path)
    if not observations or any(
        observation.get("dry_run") for observation in observations
    ):
        raise ValueError("factor analysis requires non-empty measured observations")
    effects = analyze_effects(observations)
    frontier = pareto_frontier(observations)
    model = manifest["experiment"]["model"]
    result = FactorAnalysisResult(
        run_dir=run_dir,
        output_dir=output_dir,
        quantization=str(model["quantization"]),
        model_sha256=str(model["sha256"]),
        manifest_sha256=_sha256(manifest_path),
        measurements_sha256=_sha256(measurements_path),
        effects=effects,
        pareto_frontier=frontier,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "analysis.json").write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "analysis.html").write_text(
        _render_html(result.quantization, effects, frontier), encoding="utf-8"
    )
    return result
