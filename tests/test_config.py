from pathlib import Path

import pytest
from pydantic import ValidationError

from localllm_bench.config import (
    ArrivalProcess,
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
    with pytest.raises(ValidationError, match="must be increasing"):
        OpenLoopSpec(prompt_dataset=Path("prompts.jsonl"), arrival_rates_rps=[2, 1])
    with pytest.raises(ValidationError, match="must be unique"):
        OpenLoopSpec(prompt_dataset=Path("prompts.jsonl"), arrival_rates_rps=[1, 1])


def test_open_loop_arrival_process_requires_schema_v2() -> None:
    spec = load_experiment(
        Path("configs/experiments/qwen-0.5b-q4-poisson-open-loop.yaml")
    )
    assert spec.schema_version == "2"
    assert spec.open_loop is not None
    assert spec.open_loop.arrival_process is ArrivalProcess.POISSON
    assert spec.open_loop.independent_runs == 8
    assert spec.open_loop.bootstrap_iterations == 10_000
    payload = spec.model_dump(mode="json")
    payload["schema_version"] = "1"
    with pytest.raises(ValidationError, match="require schema_version 2"):
        ExperimentSpec.model_validate(payload)


def test_load_low_rate_poisson_study() -> None:
    spec = load_experiment(
        Path("configs/experiments/qwen-0.5b-q4-poisson-low-rate.yaml")
    )
    assert spec.open_loop is not None
    assert [float(rate) for rate in spec.open_loop.arrival_rates_rps] == [
        0.5,
        1.0,
        1.5,
        2.0,
    ]
    assert spec.open_loop.independent_runs == 8


def test_open_loop_defaults_to_fixed_arrivals() -> None:
    load = OpenLoopSpec(prompt_dataset=Path("prompts.jsonl"), arrival_rates_rps=[1])
    assert load.arrival_process is ArrivalProcess.FIXED
    assert load.arrival_seed == 42
    assert load.independent_runs == 1
    assert load.bootstrap_iterations == 10_000
    assert load.bootstrap_seed == 42
    with pytest.raises(ValidationError):
        OpenLoopSpec(
            prompt_dataset=Path("prompts.jsonl"),
            arrival_rates_rps=[1],
            arrival_seed=-1,
        )
    for field, value in (
        ("arrival_rates_rps", [float("inf")]),
        ("duration_seconds", float("inf")),
    ):
        payload = {"prompt_dataset": "prompts.jsonl", "arrival_rates_rps": [1]}
        payload[field] = value
        with pytest.raises(ValidationError, match="finite number"):
            OpenLoopSpec.model_validate(payload)
    with pytest.raises(ValidationError, match="at most 1000000"):
        OpenLoopSpec(
            prompt_dataset=Path("prompts.jsonl"),
            arrival_rates_rps=[1_000_001],
            duration_seconds=1,
        )


def test_repeated_open_loop_requires_schema_v2() -> None:
    spec = load_experiment(Path("configs/experiments/qwen-0.5b-smoke.yaml"))
    payload = spec.model_dump(mode="json")
    assert isinstance(payload["open_loop"], dict)
    payload["open_loop"]["independent_runs"] = 5
    with pytest.raises(ValidationError, match="require schema_version 2"):
        ExperimentSpec.model_validate(payload)
    payload["schema_version"] = "2"
    payload["open_loop"]["independent_runs"] = 2
    with pytest.raises(ValidationError, match="at least 5"):
        ExperimentSpec.model_validate(payload)
    payload["open_loop"]["independent_runs"] = 5
    with pytest.raises(ValidationError, match="multiple of the rate count"):
        ExperimentSpec.model_validate(payload)


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
