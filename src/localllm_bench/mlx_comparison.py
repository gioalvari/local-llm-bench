"""Strict comparison of frozen base and adapted MLX evaluations."""

import html
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class MlxComparisonResult(BaseModel):
    """Base/adapted objective metric deltas."""

    output_dir: Path
    items: int
    base_exact_match: float
    adapted_exact_match: float
    exact_match_delta: float
    base_token_f1: float
    adapted_token_f1: float
    token_f1_delta: float
    base_median_output_tokens: float
    adapted_median_output_tokens: float
    base_median_output_tokens_per_second: float
    adapted_median_output_tokens_per_second: float


def _read_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("MLX evaluation manifest must be a JSON object")
    return manifest


def _read_responses(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (path / "responses.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]


def compare_mlx_evaluations(
    base_dir: Path, adapted_dir: Path, output_dir: Path
) -> MlxComparisonResult:
    """Validate and compare one frozen base/adapted evaluation pair."""
    base = _read_manifest(base_dir)
    adapted = _read_manifest(adapted_dir)
    if base["result"]["arm"] != "base" or adapted["result"]["arm"] != "adapted":
        raise ValueError("comparison requires base then adapted evaluation arms")
    controls = (
        "mlx_lm_version",
        "model_directory_sha256",
        "source_dataset_sha256",
        "max_tokens",
        "seed",
    )
    if any(base[key] != adapted[key] for key in controls):
        raise ValueError("MLX evaluations use different model, dataset, or protocol")
    if adapted.get("adapter_weights_sha256") is None:
        raise ValueError("adapted evaluation has no adapter fingerprint")
    base_responses = _read_responses(base_dir)
    adapted_responses = _read_responses(adapted_dir)
    base_items = [
        (item["item_id"], item["source_document_id"], item["reference"])
        for item in base_responses
    ]
    adapted_items = [
        (item["item_id"], item["source_document_id"], item["reference"])
        for item in adapted_responses
    ]
    if base_items != adapted_items:
        raise ValueError("MLX evaluations contain different test items or references")
    base_result = base["result"]
    adapted_result = adapted["result"]
    if base_result["items"] != adapted_result["items"]:
        raise ValueError("MLX evaluations contain different item counts")
    if len(base_responses) != int(base_result["items"]):
        raise ValueError("MLX evaluation item count does not match response artifacts")
    result = MlxComparisonResult(
        output_dir=output_dir,
        items=int(base_result["items"]),
        base_exact_match=float(base_result["exact_match"]),
        adapted_exact_match=float(adapted_result["exact_match"]),
        exact_match_delta=float(adapted_result["exact_match"])
        - float(base_result["exact_match"]),
        base_token_f1=float(base_result["token_f1"]),
        adapted_token_f1=float(adapted_result["token_f1"]),
        token_f1_delta=float(adapted_result["token_f1"])
        - float(base_result["token_f1"]),
        base_median_output_tokens=float(base_result["median_output_tokens"]),
        adapted_median_output_tokens=float(adapted_result["median_output_tokens"]),
        base_median_output_tokens_per_second=float(
            base_result["median_output_tokens_per_second"]
        ),
        adapted_median_output_tokens_per_second=float(
            adapted_result["median_output_tokens_per_second"]
        ),
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "comparison.json").write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows = [
        ("Exact match", result.base_exact_match, result.adapted_exact_match),
        ("Token F1", result.base_token_f1, result.adapted_token_f1),
        (
            "Median output tokens",
            result.base_median_output_tokens,
            result.adapted_median_output_tokens,
        ),
        (
            "Median output token/s",
            result.base_median_output_tokens_per_second,
            result.adapted_median_output_tokens_per_second,
        ),
    ]
    table_rows = "".join(
        "<tr>"
        f"<td>{html.escape(label)}</td><td>{base_value:.3f}</td>"
        f"<td>{adapted_value:.3f}</td><td>{adapted_value - base_value:+.3f}</td>"
        "</tr>"
        for label, base_value, adapted_value in rows
    )
    (output_dir / "comparison.html").write_text(
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        "<title>MLX QLoRA comparison</title></head><body>"
        "<h1>MLX QLoRA smoke comparison</h1><table><thead><tr>"
        "<th>Metric</th><th>Base</th><th>Adapted</th><th>Delta</th></tr></thead>"
        f"<tbody>{table_rows}</tbody></table></body></html>",
        encoding="utf-8",
    )
    return result
