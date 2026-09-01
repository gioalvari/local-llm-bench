import json
from pathlib import Path

import pytest

from localllm_bench.comparison import VariantRuns, compare_variants, summarize_variant


def _write_variant(
    root: Path,
    quantization: str,
    *,
    dataset_hash: str = "data",
    architecture: str = "arm64",
    zero_shot_accuracy: float = 0.75,
    quality_per_second: float = 3.0,
    peak_rss_bytes: int = 1024**3,
) -> VariantRuns:
    micro = root / "micro"
    serving = root / "serving"
    quality = root / "quality"
    micro.mkdir(parents=True)
    serving.mkdir()
    quality.mkdir()
    experiment = {
        "model": {
            "quantization": quantization,
            "sha256": quantization.lower().ljust(64, "0"),
            "source_repo": "owner/model",
            "source_revision": "revision",
        },
        "matrix": {"repetitions": 3},
        "server": {"context_size": 2048},
        "evaluation": {"dataset": "dataset.jsonl"},
    }
    capabilities = {
        "architecture": architecture,
        "logical_cpus": 14,
        "memory_bytes": 48 * 1024**3,
        "os": "Darwin",
        "os_release": "25.4.0",
        "physical_cpus": 14,
        "unified_memory": True,
        "llama_bench": {"sha256": "bench-hash"},
        "llama_server": {"sha256": "server-hash"},
    }
    (micro / "manifest.json").write_text(
        json.dumps({"experiment": experiment, "capabilities": capabilities}),
        encoding="utf-8",
    )
    cell = {
        "workload_name": "standard",
        "batch_size": 512,
        "gpu_layers": -1,
        "flash_attention": "on",
    }
    observations = [
        {
            "cell": cell,
            "metrics": {
                "n_prompt": 512,
                "n_gen": 0,
                "avg_ts": 5000.0,
                "model_size": 400,
            },
        },
        {
            "cell": cell,
            "metrics": {
                "n_prompt": 0,
                "n_gen": 128,
                "avg_ts": 250.0,
                "model_size": 400,
            },
        },
    ]
    (micro / "measurements.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in observations), encoding="utf-8"
    )
    serving_manifest = {
        "run_type": "llama-server",
        "experiment": experiment,
        "capabilities": capabilities,
        "peak_process_tree_rss_bytes": peak_rss_bytes,
    }
    (serving / "manifest.json").write_text(
        json.dumps(serving_manifest), encoding="utf-8"
    )
    (serving / "requests.jsonl").write_text(
        json.dumps({"ttft_ns": 10_000_000, "e2e_latency_ns": 200_000_000}) + "\n",
        encoding="utf-8",
    )
    arm = {
        "answer_accuracy": zero_shot_accuracy,
        "schema_valid_rate": 0.5,
        "quality_adjusted_answers_per_second": quality_per_second,
    }
    quality_manifest = {
        "run_type": "quality-evaluation",
        "experiment": experiment,
        "capabilities": capabilities,
        "dataset_sha256": dataset_hash,
        "summary": {"zero-shot": arm, "engineered": arm},
    }
    (quality / "manifest.json").write_text(
        json.dumps(quality_manifest), encoding="utf-8"
    )
    return VariantRuns(microbenchmark=micro, serving=serving, quality=quality)


def test_summarize_variant(tmp_path: Path) -> None:
    summary = summarize_variant(_write_variant(tmp_path, "Q4_K_M"))
    assert summary.quantization == "Q4_K_M"
    assert summary.prompt_tokens_per_second == 5000.0
    assert summary.generation_tokens_per_second == 250.0
    assert summary.zero_shot_quality_per_gib == 0.75


@pytest.mark.parametrize(
    ("run_name", "field_path", "value", "message"),
    [
        ("serving", ("capabilities", "architecture"), "x86_64", "hardware"),
        (
            "quality",
            ("experiment", "server", "context_size"),
            4096,
            "benchmark protocols",
        ),
        (
            "quality",
            ("capabilities", "llama_server", "sha256"),
            "other-server",
            "llama-server binaries",
        ),
    ],
)
def test_summarize_variant_rejects_mismatched_run_trio(
    tmp_path: Path,
    run_name: str,
    field_path: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    runs = _write_variant(tmp_path, "Q4_K_M")
    run_dir = getattr(runs, run_name)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    target = manifest
    for field in field_path[:-1]:
        target = target[field]
    target[field_path[-1]] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        summarize_variant(runs)


def test_summarize_variant_rejects_missing_rss(tmp_path: Path) -> None:
    runs = _write_variant(tmp_path, "Q4_K_M", peak_rss_bytes=0)
    with pytest.raises(ValueError, match="positive peak process RSS"):
        summarize_variant(runs)


def test_compare_variants_writes_aggregate_artifacts(tmp_path: Path) -> None:
    variants = [
        _write_variant(tmp_path / "q4", "Q4_K_M"),
        _write_variant(tmp_path / "q8", "Q8_0"),
    ]
    result = compare_variants(variants, tmp_path / "comparison")
    assert len(result.variants) == 2
    assert len(result.quality_pareto_frontier) == 2
    assert (result.output_dir / "comparison.json").is_file()
    assert (result.output_dir / "quality-pareto.json").is_file()
    html = (result.output_dir / "comparison.html").read_text(encoding="utf-8")
    assert "Q8_0" in html
    assert "quality-efficiency Pareto frontier" in html


def test_compare_variants_supports_three_quantizations(tmp_path: Path) -> None:
    variants = [
        _write_variant(tmp_path / "q4", "Q4_K_M"),
        _write_variant(tmp_path / "q5", "Q5_K_M"),
        _write_variant(tmp_path / "q8", "Q8_0"),
    ]
    result = compare_variants(variants, tmp_path / "comparison")
    assert [variant.quantization for variant in result.variants] == [
        "Q4_K_M",
        "Q5_K_M",
        "Q8_0",
    ]


def test_quality_pareto_frontier_removes_dominated_variants(tmp_path: Path) -> None:
    variants = [
        _write_variant(
            tmp_path / "q4",
            "Q4_K_M",
            zero_shot_accuracy=0.9,
            quality_per_second=3.0,
            peak_rss_bytes=2 * 1024**3,
        ),
        _write_variant(
            tmp_path / "q5",
            "Q5_K_M",
            zero_shot_accuracy=0.8,
            quality_per_second=2.5,
        ),
        _write_variant(
            tmp_path / "q8",
            "Q8_0",
            zero_shot_accuracy=0.6,
            quality_per_second=2.0,
            peak_rss_bytes=2 * 1024**3,
        ),
    ]
    result = compare_variants(variants, tmp_path / "comparison")
    assert [point.quantization for point in result.quality_pareto_frontier] == [
        "Q4_K_M",
        "Q5_K_M",
    ]
    artifact = json.loads(
        (result.output_dir / "quality-pareto.json").read_text(encoding="utf-8")
    )
    assert [point["quantization"] for point in artifact] == ["Q4_K_M", "Q5_K_M"]


def test_quality_pareto_frontier_keeps_exact_ties(tmp_path: Path) -> None:
    variants = [
        _write_variant(tmp_path / "q4", "Q4_K_M"),
        _write_variant(tmp_path / "q8", "Q8_0"),
    ]
    result = compare_variants(variants, tmp_path / "comparison")
    assert [point.quantization for point in result.quality_pareto_frontier] == [
        "Q4_K_M",
        "Q8_0",
    ]


def test_compare_variants_rejects_dataset_mismatch(tmp_path: Path) -> None:
    variants = [
        _write_variant(tmp_path / "q4", "Q4_K_M", dataset_hash="first"),
        _write_variant(tmp_path / "q8", "Q8_0", dataset_hash="second"),
    ]
    with pytest.raises(ValueError, match="different datasets"):
        compare_variants(variants, tmp_path / "comparison")


def test_compare_variants_requires_two_variants(tmp_path: Path) -> None:
    variant = _write_variant(tmp_path / "q4", "Q4_K_M")
    with pytest.raises(ValueError, match="at least two"):
        compare_variants([variant], tmp_path / "comparison")


def test_compare_variants_rejects_hardware_mismatch(tmp_path: Path) -> None:
    variants = [
        _write_variant(tmp_path / "q4", "Q4_K_M", architecture="arm64"),
        _write_variant(tmp_path / "q8", "Q8_0", architecture="x86_64"),
    ]
    with pytest.raises(ValueError, match="different hardware"):
        compare_variants(variants, tmp_path / "comparison")
