"""Managed MLX QLoRA training runs."""

import json
import re
import shutil
import subprocess
import time
import uuid
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from localllm_bench.artifacts import directory_sha256, file_sha256
from localllm_bench.config import ExperimentSpec, TrainingSpec
from localllm_bench.doctor import inspect_capabilities
from localllm_bench.training_data import prepare_training_dataset


class TrainingRunResult(BaseModel):
    """Location and completion status for one managed QLoRA run."""

    run_id: str
    run_dir: Path
    adapter_dir: Path
    return_code: int
    duration_ns: int


def parse_training_metrics(log: str) -> dict[str, float]:
    """Extract final reported MLX train and validation metrics."""
    metrics: dict[str, float] = {}
    validation_losses = re.findall(r"Val loss ([0-9.]+)", log)
    train_losses = re.findall(r"Train loss ([0-9.]+)", log)
    peak_memory = re.findall(r"Peak mem ([0-9.]+) GB", log)
    if validation_losses:
        metrics["initial_validation_loss"] = float(validation_losses[0])
        metrics["final_validation_loss"] = float(validation_losses[-1])
    if train_losses:
        metrics["final_train_loss"] = float(train_losses[-1])
    if peak_memory:
        metrics["peak_memory_gb"] = max(float(value) for value in peak_memory)
    return metrics


def build_training_command(
    training: TrainingSpec,
    model_path: Path,
    data_dir: Path,
    adapter_dir: Path,
) -> list[str]:
    """Build an explicit MLX-LM QLoRA command."""
    command = [
        "mlx_lm.lora",
        "--model",
        str(model_path),
        "--train",
        "--data",
        str(data_dir),
        "--adapter-path",
        str(adapter_dir),
        "--iters",
        str(training.iterations),
        "--batch-size",
        str(training.batch_size),
        "--grad-accumulation-steps",
        str(training.gradient_accumulation_steps),
        "--num-layers",
        str(training.num_layers),
        "--learning-rate",
        str(training.learning_rate),
        "--max-seq-length",
        str(training.max_seq_length),
        "--seed",
        str(training.seed),
    ]
    if training.mask_prompt:
        command.append("--mask-prompt")
    return command


def run_training(experiment: ExperimentSpec, model_path: Path) -> TrainingRunResult:
    """Prepare data and execute MLX QLoRA in an isolated run directory."""
    if experiment.training is None:
        raise ValueError("experiment does not define a training section")
    training = experiment.training
    executable = shutil.which("mlx_lm.lora")
    if executable is None:
        raise FileNotFoundError("mlx_lm.lora is not installed; run make install-mlx")
    if not model_path.is_dir():
        raise FileNotFoundError(f"MLX model directory does not exist: {model_path}")
    model_digest = directory_sha256(model_path)
    if model_digest.lower() != training.model_sha256.lower():
        raise ValueError(
            f"MLX model SHA-256 mismatch: expected {training.model_sha256}, "
            f"observed {model_digest}"
        )
    run_id = (
        f"{experiment.experiment_id}-train-"
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    )
    run_dir = experiment.output_dir / run_id
    data_dir = run_dir / "data"
    adapter_dir = run_dir / "adapters"
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True)
    prepared = prepare_training_dataset(training.source_dataset, data_dir)
    command = build_training_command(training, model_path, data_dir, adapter_dir)
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "run_type": "mlx-qlora-training",
        "mlx_lm_version": version("mlx-lm"),
        "training": training.model_dump(mode="json"),
        "model_path": str(model_path),
        "model_directory_sha256": model_digest,
        "prepared_dataset": prepared.model_dump(mode="json"),
        "command": command,
        "capabilities": inspect_capabilities(run_dir).model_dump(mode="json"),
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    started_ns = time.monotonic_ns()
    with (
        (logs_dir / "training.stdout.log").open("w", encoding="utf-8") as stdout,
        (logs_dir / "training.stderr.log").open("w", encoding="utf-8") as stderr,
    ):
        result = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
    duration_ns = time.monotonic_ns() - started_ns
    training_log = logs_dir / "training.stdout.log"
    manifest["return_code"] = result.returncode
    manifest["duration_ns"] = duration_ns
    manifest["metrics"] = parse_training_metrics(
        training_log.read_text(encoding="utf-8")
    )
    manifest["adapter_directory_sha256"] = (
        directory_sha256(adapter_dir) if adapter_dir.is_dir() else None
    )
    adapter_weights = adapter_dir / "adapters.safetensors"
    manifest["adapter_weights_sha256"] = (
        file_sha256(adapter_weights) if adapter_weights.is_file() else None
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return TrainingRunResult(
        run_id=run_id,
        run_dir=run_dir,
        adapter_dir=adapter_dir,
        return_code=result.returncode,
        duration_ns=duration_ns,
    )
