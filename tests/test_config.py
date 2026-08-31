from pathlib import Path

import pytest
from pydantic import ValidationError

from localllm_bench.config import (
    BenchmarkMatrix,
    ContextSweepSpec,
    ExperimentSpec,
    LoadSpec,
    ModelSpec,
    OpenLoopSpec,
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


def test_load_q5_example() -> None:
    spec = load_experiment(Path("configs/experiments/qwen-0.5b-q5-smoke.yaml"))
    assert spec.model.quantization == "Q5_K_M"
    assert spec.model.sha256 == (
        "a0a413dcbb4676f21d4c951b98a393324694edb1a20a4f9547d1de8d2919ff3b"
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


def test_open_loop_requires_unique_increasing_rates() -> None:
    with pytest.raises(ValidationError, match="unique and increasing"):
        OpenLoopSpec(prompt_dataset=Path("prompts.jsonl"), arrival_rates_rps=[2, 1])
    with pytest.raises(ValidationError, match="unique and increasing"):
        OpenLoopSpec(prompt_dataset=Path("prompts.jsonl"), arrival_rates_rps=[1, 1])


def test_context_sweep_rejects_duplicates_and_undersized_windows() -> None:
    with pytest.raises(ValidationError, match="case names must be unique"):
        ContextSweepSpec.model_validate(
            {
                "corpus": "corpus.txt",
                "output_tokens": 8,
                "cases": [
                    {
                        "name": "same",
                        "series": ["window-size"],
                        "context_size": 32,
                        "prompt_tokens": 16,
                    },
                    {
                        "name": "same",
                        "series": ["window-size"],
                        "context_size": 64,
                        "prompt_tokens": 16,
                    },
                ],
            }
        )
    with pytest.raises(ValidationError, match="must fit"):
        ContextSweepSpec.model_validate(
            {
                "corpus": "corpus.txt",
                "output_tokens": 16,
                "cases": [
                    {
                        "name": "too-small",
                        "series": ["window-size"],
                        "context_size": 32,
                        "prompt_tokens": 24,
                    }
                ],
            }
        )


def test_context_sweep_rejects_confounded_series() -> None:
    with pytest.raises(ValidationError, match="keep prompt_tokens fixed"):
        ContextSweepSpec.model_validate(
            {
                "corpus": "corpus.txt",
                "output_tokens": 8,
                "cases": [
                    {
                        "name": "one",
                        "series": ["window-size"],
                        "context_size": 64,
                        "prompt_tokens": 16,
                    },
                    {
                        "name": "two",
                        "series": ["window-size"],
                        "context_size": 128,
                        "prompt_tokens": 32,
                    },
                ],
            }
        )


def test_context_sweep_accepts_controlled_two_series() -> None:
    sweep = ContextSweepSpec.model_validate(
        {
            "corpus": "corpus.txt",
            "output_tokens": 8,
            "cases": [
                {
                    "name": "small-window",
                    "series": ["window-size"],
                    "context_size": 64,
                    "prompt_tokens": 16,
                },
                {
                    "name": "shared-baseline",
                    "series": ["window-size", "prompt-length"],
                    "context_size": 128,
                    "prompt_tokens": 16,
                },
                {
                    "name": "long-prompt",
                    "series": ["prompt-length"],
                    "context_size": 128,
                    "prompt_tokens": 32,
                },
            ],
        }
    )
    assert len(sweep.cases) == 3
    with pytest.raises(ValidationError, match="keep context_size fixed"):
        ContextSweepSpec.model_validate(
            {
                "corpus": "corpus.txt",
                "output_tokens": 8,
                "cases": [
                    {
                        "name": "one",
                        "series": ["prompt-length"],
                        "context_size": 64,
                        "prompt_tokens": 16,
                    },
                    {
                        "name": "two",
                        "series": ["prompt-length"],
                        "context_size": 128,
                        "prompt_tokens": 32,
                    },
                ],
            }
        )
