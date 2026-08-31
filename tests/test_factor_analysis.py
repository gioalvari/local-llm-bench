import json
import statistics
from pathlib import Path

import pytest

from localllm_bench.factor_analysis import (
    analyze_effects,
    analyze_factor_run,
    bootstrap_geometric_mean_interval,
    paired_ratios,
    pareto_frontier,
)


def _observations() -> list[dict[str, object]]:
    observations: list[dict[str, object]] = []
    cell_id = 0
    for workload, prompt_tokens, generation_tokens in (
        ("short", 128, 64),
        ("standard", 512, 128),
    ):
        for batch_size in (128, 512):
            for gpu_layers in (0, -1):
                for flash_attention in ("off", "on"):
                    cell_id += 1
                    base = 100.0
                    multiplier = (
                        (1.2 if batch_size == 512 else 1.0)
                        * (2.0 if gpu_layers == -1 else 1.0)
                        * (1.1 if flash_attention == "on" else 1.0)
                    )
                    cell = {
                        "cell_id": str(cell_id),
                        "workload_name": workload,
                        "prompt_tokens": prompt_tokens,
                        "generation_tokens": generation_tokens,
                        "batch_size": batch_size,
                        "ubatch_size": 128,
                        "threads": 10,
                        "gpu_layers": gpu_layers,
                        "flash_attention": flash_attention,
                    }
                    rss = 500 if gpu_layers == -1 else 700
                    observations.extend(
                        [
                            {
                                "cell": cell,
                                "metrics": {
                                    "n_prompt": prompt_tokens,
                                    "n_gen": 0,
                                    "avg_ts": base * multiplier,
                                },
                                "peak_process_tree_rss_bytes": rss,
                            },
                            {
                                "cell": cell,
                                "metrics": {
                                    "n_prompt": 0,
                                    "n_gen": generation_tokens,
                                    "avg_ts": base * multiplier / 2,
                                },
                                "peak_process_tree_rss_bytes": rss,
                            },
                        ]
                    )
    return observations


def test_bootstrap_interval_is_deterministic() -> None:
    first = bootstrap_geometric_mean_interval([1.0, 2.0, 4.0], iterations=100)
    second = bootstrap_geometric_mean_interval([1.0, 2.0, 4.0], iterations=100)
    assert first == second
    assert first[0] <= 2.0 <= first[1]
    with pytest.raises(ValueError, match="positive and non-empty"):
        bootstrap_geometric_mean_interval([])
    with pytest.raises(ValueError, match="iterations"):
        bootstrap_geometric_mean_interval([1.0], iterations=0)


def test_paired_effects_recover_known_multipliers() -> None:
    effects = analyze_effects(_observations())
    by_factor_phase = {(effect.factor, effect.phase): effect for effect in effects}
    assert by_factor_phase[
        ("gpu_layers", "prompt")
    ].geometric_mean_ratio == pytest.approx(2.0)
    assert by_factor_phase[
        ("flash_attention", "generation")
    ].effect_percent == pytest.approx(10.0)
    assert by_factor_phase[
        ("batch_size", "prompt")
    ].geometric_mean_ratio == pytest.approx(1.2)
    assert all(effect.pair_count == 8 for effect in effects)


def test_paired_ratios_reject_incomplete_factorial() -> None:
    observations = _observations()
    observations = [
        item
        for item in observations
        if not (
            item["cell"]["cell_id"] == "1"  # type: ignore[index]
            and item["metrics"]["n_prompt"] > 0  # type: ignore[index]
        )
    ]
    with pytest.raises(ValueError, match="incomplete paired"):
        paired_ratios(observations, "gpu_layers", "prompt")
    with pytest.raises(ValueError, match="unsupported"):
        paired_ratios(observations, "unknown", "prompt")


def test_batch_levels_are_derived_from_observations() -> None:
    observations = _observations()
    updated_cells: set[str] = set()
    for observation in observations:
        cell = observation["cell"]
        cell_id = str(cell["cell_id"])  # type: ignore[index]
        if cell_id in updated_cells:
            continue
        cell["batch_size"] = 256 if cell["batch_size"] == 128 else 1024  # type: ignore[index]
        updated_cells.add(cell_id)
    ratios = paired_ratios(observations, "batch_size", "prompt")
    assert statistics.geometric_mean(ratios) == pytest.approx(1.2)


def test_pareto_frontier_removes_dominated_cpu_points() -> None:
    frontier = pareto_frontier(_observations())
    assert frontier
    assert all(point.gpu_layers == -1 for point in frontier)
    assert {point.workload_name for point in frontier} == {"short", "standard"}


def test_analyze_factor_run_writes_outputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest = {"experiment": {"model": {"quantization": "Q4_K_M", "sha256": "a" * 64}}}
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "measurements.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in _observations()), encoding="utf-8"
    )
    result = analyze_factor_run(run_dir, tmp_path / "analysis")
    assert result.quantization == "Q4_K_M"
    assert len(result.effects) == 6
    assert (result.output_dir / "analysis.json").is_file()
    assert "Pareto frontier" in (result.output_dir / "analysis.html").read_text(
        encoding="utf-8"
    )
