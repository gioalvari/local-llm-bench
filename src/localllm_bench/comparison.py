"""Cross-quantization aggregation for completed benchmark runs."""

import hashlib
import html
import json
import statistics
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class VariantRuns(BaseModel):
    """Run directories required to compare one model variant."""

    microbenchmark: Path
    serving: Path
    quality: Path


class VariantSummary(BaseModel):
    """Comparable performance, memory, and quality metrics."""

    quantization: str
    model_sha256: str
    source_repo: str
    source_revision: str
    dataset_sha256: str
    protocol_sha256: str
    hardware_sha256: str
    llama_bench_sha256: str
    llama_server_sha256: str
    model_size_bytes: int
    prompt_tokens_per_second: float
    generation_tokens_per_second: float
    median_ttft_ms: float
    median_e2e_ms: float
    peak_process_rss_bytes: int
    zero_shot_accuracy: float
    engineered_accuracy: float
    zero_shot_schema_validity: float
    engineered_schema_validity: float
    zero_shot_quality_per_second: float
    zero_shot_quality_per_gib: float


class ComparisonResult(BaseModel):
    """Output files and summaries from one comparison."""

    output_dir: Path
    variants: list[VariantSummary]


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


def _model_identity(manifest: dict[str, Any]) -> tuple[str, str]:
    model = manifest["experiment"]["model"]
    return str(model["quantization"]), str(model["sha256"])


def _protocol_sha256(manifest: dict[str, Any]) -> str:
    experiment = manifest["experiment"]
    protocol = {
        "matrix": experiment["matrix"],
        "server": experiment["server"],
        "evaluation": experiment["evaluation"],
    }
    payload = json.dumps(protocol, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _hardware_sha256(manifest: dict[str, Any]) -> str:
    capabilities = manifest["capabilities"]
    hardware = {
        "architecture": capabilities["architecture"],
        "logical_cpus": capabilities["logical_cpus"],
        "memory_bytes": capabilities["memory_bytes"],
        "os": capabilities["os"],
        "os_release": capabilities["os_release"],
        "physical_cpus": capabilities["physical_cpus"],
        "unified_memory": capabilities["unified_memory"],
    }
    payload = json.dumps(hardware, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _validate_run_types(
    micro: dict[str, Any], serving: dict[str, Any], quality: dict[str, Any]
) -> None:
    if serving.get("run_type") != "llama-server":
        raise ValueError("serving run does not have run_type llama-server")
    if quality.get("run_type") != "quality-evaluation":
        raise ValueError("quality run does not have run_type quality-evaluation")
    identities = {_model_identity(item) for item in (micro, serving, quality)}
    if len(identities) != 1:
        raise ValueError("variant run manifests refer to different model artifacts")


def summarize_variant(runs: VariantRuns) -> VariantSummary:
    """Extract one comparable row from three compatible run directories."""
    micro_manifest = _read_json(runs.microbenchmark / "manifest.json")
    serving_manifest = _read_json(runs.serving / "manifest.json")
    quality_manifest = _read_json(runs.quality / "manifest.json")
    _validate_run_types(micro_manifest, serving_manifest, quality_manifest)
    quantization, model_sha256 = _model_identity(micro_manifest)
    model = micro_manifest["experiment"]["model"]
    measurements = _read_jsonl(runs.microbenchmark / "measurements.jsonl")
    target = [
        item
        for item in measurements
        if item["cell"]["workload_name"] == "standard"
        and item["cell"]["batch_size"] == 512
        and item["cell"]["gpu_layers"] == -1
        and item["cell"]["flash_attention"] == "on"
    ]
    prompt_rates = [
        float(item["metrics"]["avg_ts"])
        for item in target
        if int(item["metrics"]["n_prompt"]) > 0
    ]
    generation_rates = [
        float(item["metrics"]["avg_ts"])
        for item in target
        if int(item["metrics"]["n_gen"]) > 0
    ]
    if len(prompt_rates) != 1 or len(generation_rates) != 1:
        raise ValueError("microbenchmark run lacks the canonical comparison cell")
    requests = _read_jsonl(runs.serving / "requests.jsonl")
    if not requests:
        raise ValueError("serving run contains no requests")
    summary = quality_manifest.get("summary", {})
    zero_shot = summary.get("zero-shot")
    engineered = summary.get("engineered")
    if not isinstance(zero_shot, dict) or not isinstance(engineered, dict):
        raise ValueError("quality run lacks required prompt-arm summaries")
    model_sizes = {int(item["metrics"]["model_size"]) for item in target}
    if len(model_sizes) != 1:
        raise ValueError("microbenchmark run has inconsistent model sizes")
    peak_rss = int(serving_manifest["peak_process_tree_rss_bytes"])
    zero_shot_accuracy = float(zero_shot["answer_accuracy"])
    return VariantSummary(
        quantization=quantization,
        model_sha256=model_sha256,
        source_repo=str(model["source_repo"]),
        source_revision=str(model["source_revision"]),
        dataset_sha256=str(quality_manifest["dataset_sha256"]),
        protocol_sha256=_protocol_sha256(quality_manifest),
        hardware_sha256=_hardware_sha256(micro_manifest),
        llama_bench_sha256=str(micro_manifest["capabilities"]["llama_bench"]["sha256"]),
        llama_server_sha256=str(
            serving_manifest["capabilities"]["llama_server"]["sha256"]
        ),
        model_size_bytes=model_sizes.pop(),
        prompt_tokens_per_second=prompt_rates[0],
        generation_tokens_per_second=generation_rates[0],
        median_ttft_ms=statistics.median(
            int(item["ttft_ns"]) / 1_000_000 for item in requests
        ),
        median_e2e_ms=statistics.median(
            int(item["e2e_latency_ns"]) / 1_000_000 for item in requests
        ),
        peak_process_rss_bytes=peak_rss,
        zero_shot_accuracy=zero_shot_accuracy,
        engineered_accuracy=float(engineered["answer_accuracy"]),
        zero_shot_schema_validity=float(zero_shot["schema_valid_rate"]),
        engineered_schema_validity=float(engineered["schema_valid_rate"]),
        zero_shot_quality_per_second=float(
            zero_shot["quality_adjusted_answers_per_second"]
        ),
        zero_shot_quality_per_gib=(
            zero_shot_accuracy / (peak_rss / (1024**3)) if peak_rss > 0 else 0.0
        ),
    )


def _render_html(variants: list[VariantSummary]) -> str:
    rows: list[str] = []
    for variant in variants:
        values = [
            variant.quantization,
            f"{variant.model_size_bytes / (1024**2):.1f}",
            f"{variant.prompt_tokens_per_second:.2f}",
            f"{variant.generation_tokens_per_second:.2f}",
            f"{variant.median_ttft_ms:.2f}",
            f"{variant.median_e2e_ms:.2f}",
            f"{variant.peak_process_rss_bytes / (1024**2):.1f}",
            f"{variant.zero_shot_accuracy:.1%}",
            f"{variant.engineered_schema_validity:.1%}",
            f"{variant.zero_shot_quality_per_second:.3f}",
            f"{variant.zero_shot_quality_per_gib:.3f}",
        ]
        rows.append(
            "<tr>"
            + "".join(f"<td>{html.escape(value)}</td>" for value in values)
            + "</tr>"
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LocalLLM Bench comparison</title>
<style>
body {{ font-family: ui-monospace, monospace; margin: 2rem; background: #101417;
color: #e8efe9; }} h1 {{ color: #9be15d; }} table {{ border-collapse: collapse;
width: 100%; }} th, td {{ padding: .7rem; border-bottom: 1px solid #344038;
text-align: right; }} th:first-child, td:first-child {{ text-align: left; }}
th {{ color: #101417; background: #9be15d; }} .panel {{ overflow-x: auto; }}
</style></head><body><h1>Quantization comparison</h1><div class="panel"><table>
<thead><tr><th>Quant</th><th>Model MiB</th><th>Prompt token/s</th>
<th>Generation token/s</th><th>TTFT ms</th><th>E2E ms</th><th>RSS MiB</th>
<th>Zero-shot accuracy</th><th>Engineered schema</th><th>Quality/s</th>
<th>Quality/GiB</th></tr></thead><tbody>{"".join(rows)}</tbody>
</table></div></body></html>"""


def compare_variants(variants: list[VariantRuns], output_dir: Path) -> ComparisonResult:
    """Compare two or more variants and write portable aggregate artifacts."""
    if len(variants) < 2:
        raise ValueError("at least two variants are required")
    summaries = [summarize_variant(variant) for variant in variants]
    quantizations = [item.quantization for item in summaries]
    if len(quantizations) != len(set(quantizations)):
        raise ValueError("comparison contains duplicate quantizations")
    checkpoints = {(item.source_repo, item.source_revision) for item in summaries}
    if len(checkpoints) != 1:
        raise ValueError("comparison variants use different source checkpoints")
    if len({item.dataset_sha256 for item in summaries}) != 1:
        raise ValueError("comparison variants use different datasets")
    if len({item.protocol_sha256 for item in summaries}) != 1:
        raise ValueError("comparison variants use different benchmark protocols")
    if len({item.hardware_sha256 for item in summaries}) != 1:
        raise ValueError("comparison variants use different hardware")
    if len({item.llama_bench_sha256 for item in summaries}) != 1:
        raise ValueError("comparison variants use different llama-bench binaries")
    if len({item.llama_server_sha256 for item in summaries}) != 1:
        raise ValueError("comparison variants use different llama-server binaries")
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "comparison.json").write_text(
        json.dumps(
            [item.model_dump(mode="json") for item in summaries],
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "comparison.html").write_text(
        _render_html(summaries), encoding="utf-8"
    )
    return ComparisonResult(output_dir=output_dir, variants=summaries)
