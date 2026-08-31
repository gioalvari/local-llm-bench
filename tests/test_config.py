from pathlib import Path

import pytest
from pydantic import ValidationError

from localllm_bench.config import (
    BenchmarkMatrix,
    ExperimentSpec,
    LoadSpec,
    ModelSpec,
    ServerSpec,
    WorkloadSpec,
    load_experiment,
)


def test_model_requires_exactly_one_source() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        ModelSpec(name="test", quantization="Q4")
    with pytest.raises(ValidationError, match="exactly one"):
        ModelSpec(
            name="test",
            path=Path("model.gguf"),
            hf_repo="owner/model",
            quantization="Q4",
        )


def test_hf_file_requires_repository() -> None:
    with pytest.raises(ValidationError, match="hf_file requires"):
        ModelSpec(
            name="test",
            path=Path("model.gguf"),
            hf_file="model.gguf",
            quantization="Q4",
        )


def test_load_example() -> None:
    spec = load_experiment(Path("configs/experiments/qwen-0.5b-smoke.yaml"))
    assert isinstance(spec, ExperimentSpec)
    assert spec.model.quantization == "Q4_K_M"
    assert len(spec.matrix.workloads) == 2


def test_load_q8_example() -> None:
    spec = load_experiment(Path("configs/experiments/qwen-0.5b-q8-smoke.yaml"))
    assert spec.model.quantization == "Q8_0"
    assert spec.model.sha256 == (
        "25130a98aa782284a7dabea0c23245b2fd371ed47244e79d78b8ec23245fdf96"
    )


def test_workload_rejects_no_work() -> None:
    with pytest.raises(ValidationError, match="at least one token"):
        WorkloadSpec(name="empty", prompt_tokens=0, generation_tokens=0)


def test_matrix_rejects_invalid_gpu_layer_count() -> None:
    with pytest.raises(ValidationError, match="must be -1 or non-negative"):
        BenchmarkMatrix(
            workloads=[
                WorkloadSpec(name="short", prompt_tokens=8, generation_tokens=4)
            ],
            gpu_layers=[-2],
        )


def test_server_rejects_invalid_runtime_settings() -> None:
    with pytest.raises(ValidationError, match="must not exceed"):
        ServerSpec(prompt="test", batch_size=16, ubatch_size=32)
    with pytest.raises(ValidationError, match="must be -1"):
        ServerSpec(prompt="test", gpu_layers=-2)


def test_load_requires_unique_increasing_levels() -> None:
    with pytest.raises(ValidationError, match="unique and increasing"):
        LoadSpec(concurrency_levels=[2, 1])
    with pytest.raises(ValidationError, match="unique and increasing"):
        LoadSpec(concurrency_levels=[1, 1])
