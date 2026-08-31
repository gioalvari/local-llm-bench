import json
from pathlib import Path

import pytest

from localllm_bench.mlx_comparison import compare_mlx_evaluations


def _manifest(arm: str) -> dict[str, object]:
    return {
        "adapter_directory_sha256": "adapter" if arm == "adapted" else None,
        "adapter_weights_sha256": "weights" if arm == "adapted" else None,
        "max_tokens": 64,
        "mlx_lm_version": "0.31.3",
        "model_directory_sha256": "model",
        "seed": 42,
        "source_dataset_sha256": "dataset",
        "result": {
            "arm": arm,
            "items": 6,
            "exact_match": 0.0,
            "token_f1": 0.2 if arm == "base" else 0.1,
            "median_output_tokens": 30,
            "median_output_tokens_per_second": 200,
        },
    }


def test_compare_mlx_evaluations(tmp_path: Path) -> None:
    base = tmp_path / "base"
    adapted = tmp_path / "adapted"
    base.mkdir()
    adapted.mkdir()
    (base / "manifest.json").write_text(json.dumps(_manifest("base")), encoding="utf-8")
    (adapted / "manifest.json").write_text(
        json.dumps(_manifest("adapted")), encoding="utf-8"
    )
    responses = "".join(
        json.dumps(
            {
                "item_id": f"item-{index}",
                "source_document_id": "doc",
                "reference": "answer",
            }
        )
        + "\n"
        for index in range(6)
    )
    (base / "responses.jsonl").write_text(responses, encoding="utf-8")
    (adapted / "responses.jsonl").write_text(responses, encoding="utf-8")
    result = compare_mlx_evaluations(base, adapted, tmp_path / "comparison")
    assert result.token_f1_delta == pytest.approx(-0.1)
    assert (result.output_dir / "comparison.json").is_file()
    assert "Adapted" in (result.output_dir / "comparison.html").read_text(
        encoding="utf-8"
    )


def test_compare_mlx_rejects_protocol_mismatch(tmp_path: Path) -> None:
    base = tmp_path / "base"
    adapted = tmp_path / "adapted"
    base.mkdir()
    adapted.mkdir()
    base_manifest = _manifest("base")
    adapted_manifest = _manifest("adapted")
    adapted_manifest["seed"] = 7
    (base / "manifest.json").write_text(json.dumps(base_manifest), encoding="utf-8")
    (adapted / "manifest.json").write_text(
        json.dumps(adapted_manifest), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="different model, dataset, or protocol"):
        compare_mlx_evaluations(base, adapted, tmp_path / "comparison")


def test_compare_mlx_rejects_different_items(tmp_path: Path) -> None:
    base = tmp_path / "base"
    adapted = tmp_path / "adapted"
    base.mkdir()
    adapted.mkdir()
    (base / "manifest.json").write_text(json.dumps(_manifest("base")), encoding="utf-8")
    (adapted / "manifest.json").write_text(
        json.dumps(_manifest("adapted")), encoding="utf-8"
    )
    base_response = {
        "item_id": "one",
        "source_document_id": "doc",
        "reference": "answer",
    }
    adapted_response = {**base_response, "item_id": "two"}
    (base / "responses.jsonl").write_text(
        json.dumps(base_response) + "\n", encoding="utf-8"
    )
    (adapted / "responses.jsonl").write_text(
        json.dumps(adapted_response) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="different test items"):
        compare_mlx_evaluations(base, adapted, tmp_path / "comparison")
