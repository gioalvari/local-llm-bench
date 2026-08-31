"""Base and adapter evaluation through the optional MLX-LM runtime."""

import hashlib
import importlib
import json
import statistics
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from localllm_bench.artifacts import directory_sha256, file_sha256
from localllm_bench.quality import normalized_exact_match, token_f1
from localllm_bench.training_data import TrainingItem, load_training_items


class MlxEvaluationResult(BaseModel):
    """Aggregate objective metrics for one MLX model arm."""

    arm: str
    items: int
    exact_match: float
    token_f1: float
    median_latency_ms: float
    median_output_tokens: float
    median_output_tokens_per_second: float
    responses_path: Path


def build_evaluation_prompt(tokenizer: Any, item: TrainingItem) -> str:
    """Apply the loaded model's chat template to one test question."""
    messages = [
        {
            "role": "system",
            "content": "Answer the energy-market question accurately and concisely.",
        },
        {"role": "user", "content": item.question},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if not isinstance(prompt, str):
        raise ValueError("chat template did not return text")
    return prompt


def evaluate_mlx_model(
    model_path: Path,
    source_dataset: Path,
    output_dir: Path,
    *,
    adapter_path: Path | None = None,
    expected_model_sha256: str | None = None,
    max_tokens: int = 64,
    seed: int = 42,
) -> MlxEvaluationResult:
    """Evaluate a base or adapted MLX model on the held-out test split."""
    try:
        mlx_lm = importlib.import_module("mlx_lm")
        sample_utils = importlib.import_module("mlx_lm.sample_utils")
        mx = importlib.import_module("mlx.core")
    except ImportError as error:
        raise FileNotFoundError(
            "MLX-LM is not installed; run make install-mlx"
        ) from error
    if not model_path.is_dir():
        raise FileNotFoundError(f"MLX model directory does not exist: {model_path}")
    if adapter_path is not None and not adapter_path.is_dir():
        raise FileNotFoundError(f"adapter directory does not exist: {adapter_path}")
    model_digest = directory_sha256(model_path)
    if (
        expected_model_sha256 is not None
        and model_digest.lower() != expected_model_sha256.lower()
    ):
        raise ValueError(
            f"MLX model SHA-256 mismatch: expected {expected_model_sha256}, "
            f"observed {model_digest}"
        )
    items = [
        item for item in load_training_items(source_dataset) if item.split == "test"
    ]
    if not items:
        raise ValueError("training dataset has no test items")
    output_dir.mkdir(parents=True, exist_ok=False)
    model, tokenizer = mlx_lm.load(
        str(model_path),
        adapter_path=str(adapter_path) if adapter_path is not None else None,
    )
    sampler = sample_utils.make_sampler(temp=0.0)
    records: list[dict[str, Any]] = []
    for item in items:
        mx.random.seed(seed)
        prompt = build_evaluation_prompt(tokenizer, item)
        started_ns = time.monotonic_ns()
        response = mlx_lm.generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=False,
        ).strip()
        latency_ns = time.monotonic_ns() - started_ns
        output_tokens = len(tokenizer.encode(response, add_special_tokens=False))
        records.append(
            {
                "item_id": item.item_id,
                "source_document_id": item.source_document_id,
                "prediction": response,
                "reference": item.answer,
                "exact_match": normalized_exact_match(response, [item.answer]),
                "token_f1": token_f1(response, [item.answer]),
                "latency_ns": latency_ns,
                "output_tokens": output_tokens,
                "output_tokens_per_second": (
                    output_tokens / (latency_ns / 1_000_000_000)
                    if latency_ns > 0
                    else 0.0
                ),
            }
        )
    responses_path = output_dir / "responses.jsonl"
    responses_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    arm = "adapted" if adapter_path is not None else "base"
    result = MlxEvaluationResult(
        arm=arm,
        items=len(records),
        exact_match=statistics.fmean(record["exact_match"] for record in records),
        token_f1=statistics.fmean(record["token_f1"] for record in records),
        median_latency_ms=statistics.median(
            record["latency_ns"] / 1_000_000 for record in records
        ),
        median_output_tokens=statistics.median(
            record["output_tokens"] for record in records
        ),
        median_output_tokens_per_second=statistics.median(
            record["output_tokens_per_second"] for record in records
        ),
        responses_path=responses_path,
    )
    manifest = {
        "mlx_lm_version": version("mlx-lm"),
        "result": result.model_dump(mode="json"),
        "model_path": str(model_path),
        "model_directory_sha256": model_digest,
        "adapter_path": str(adapter_path) if adapter_path is not None else None,
        "adapter_directory_sha256": (
            directory_sha256(adapter_path) if adapter_path is not None else None
        ),
        "adapter_weights_sha256": (
            file_sha256(adapter_path / "adapters.safetensors")
            if adapter_path is not None
            and (adapter_path / "adapters.safetensors").is_file()
            else None
        ),
        "source_dataset_sha256": hashlib.sha256(
            source_dataset.read_bytes()
        ).hexdigest(),
        "max_tokens": max_tokens,
        "seed": seed,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result
