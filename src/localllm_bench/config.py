"""Validated experiment configuration."""

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, Field, PositiveInt, model_validator

NonNegativeInt = Annotated[int, Field(ge=0)]


class FlashAttention(StrEnum):
    """Supported llama.cpp Flash Attention modes."""

    OFF = "off"
    ON = "on"
    AUTO = "auto"


class PromptArm(StrEnum):
    """Frozen prompt variants used for quality comparisons."""

    ZERO_SHOT = "zero-shot"
    ENGINEERED = "engineered"


class ModelSpec(BaseModel):
    """Location and identity of one model artifact."""

    name: str = Field(min_length=1)
    path: Path | None = None
    hf_repo: str | None = None
    hf_file: str | None = None
    source_repo: str | None = None
    source_revision: str | None = None
    sha256: str | None = Field(default=None, pattern=r"^[a-fA-F0-9]{64}$")
    quantization: str = Field(min_length=1)
    offline: bool = False

    @model_validator(mode="after")
    def validate_source(self) -> "ModelSpec":
        """Require exactly one local or remote model source."""
        if (self.path is None) == (self.hf_repo is None):
            raise ValueError("set exactly one of model.path or model.hf_repo")
        if self.hf_file is not None and self.hf_repo is None:
            raise ValueError("model.hf_file requires model.hf_repo")
        return self


class WorkloadSpec(BaseModel):
    """Prompt and generation sizes for a microbenchmark workload."""

    name: str = Field(min_length=1)
    prompt_tokens: NonNegativeInt
    generation_tokens: NonNegativeInt

    @model_validator(mode="after")
    def validate_work(self) -> "WorkloadSpec":
        """Reject workloads that perform no work."""
        if self.prompt_tokens == 0 and self.generation_tokens == 0:
            raise ValueError("a workload must process or generate at least one token")
        return self


class BenchmarkMatrix(BaseModel):
    """Dimensions expanded into concrete llama.cpp benchmark cells."""

    repetitions: PositiveInt = 5
    workloads: list[WorkloadSpec] = Field(min_length=1)
    batch_sizes: list[PositiveInt] = Field(default_factory=lambda: [512], min_length=1)
    ubatch_sizes: list[PositiveInt] = Field(default_factory=lambda: [128], min_length=1)
    threads: list[PositiveInt] = Field(default_factory=lambda: [1], min_length=1)
    gpu_layers: list[int] = Field(default_factory=lambda: [-1], min_length=1)
    flash_attention: list[FlashAttention] = Field(
        default_factory=lambda: [FlashAttention.AUTO], min_length=1
    )

    @model_validator(mode="after")
    def validate_gpu_layers(self) -> "BenchmarkMatrix":
        """Allow full offload (-1) or a non-negative layer count."""
        if any(value < -1 for value in self.gpu_layers):
            raise ValueError("gpu_layers values must be -1 or non-negative")
        return self


class ServerSpec(BaseModel):
    """Configuration for an end-to-end llama.cpp serving benchmark."""

    prompt: str = Field(min_length=1)
    repetitions: PositiveInt = 5
    output_tokens: PositiveInt = 64
    context_size: PositiveInt = 2048
    batch_size: PositiveInt = 512
    ubatch_size: PositiveInt = 128
    threads: PositiveInt = 1
    gpu_layers: int = -1
    flash_attention: FlashAttention = FlashAttention.AUTO
    parallel: PositiveInt = 1
    port: int | None = Field(default=None, ge=1024, le=65535)
    startup_timeout_seconds: PositiveInt = 120
    request_timeout_seconds: PositiveInt = 120

    @model_validator(mode="after")
    def validate_runtime(self) -> "ServerSpec":
        """Reject serving settings that llama.cpp cannot execute safely."""
        if self.ubatch_size > self.batch_size:
            raise ValueError("server ubatch_size must not exceed batch_size")
        if self.gpu_layers < -1:
            raise ValueError("server gpu_layers must be -1 or non-negative")
        return self


class EvaluationSpec(BaseModel):
    """Objective quality evaluation configuration."""

    dataset: Path
    prompt_arms: list[PromptArm] = Field(
        default_factory=lambda: [PromptArm.ZERO_SHOT, PromptArm.ENGINEERED],
        min_length=1,
    )
    output_tokens: PositiveInt = 96


class LoadSpec(BaseModel):
    """Closed-loop concurrent serving workload."""

    concurrency_levels: list[PositiveInt] = Field(min_length=1)
    waves_per_level: PositiveInt = 3
    warmup_requests: NonNegativeInt = 1

    @model_validator(mode="after")
    def validate_levels(self) -> "LoadSpec":
        """Require unique concurrency levels in increasing order."""
        levels = [int(value) for value in self.concurrency_levels]
        if levels != sorted(set(levels)):
            raise ValueError("concurrency_levels must be unique and increasing")
        return self


class ExperimentSpec(BaseModel):
    """Top-level benchmark experiment specification."""

    schema_version: Literal["1"] = "1"
    experiment_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    output_dir: Path = Path("runs")
    llama_bench_binary: str = "llama-bench"
    llama_server_binary: str = "llama-server"
    sample_interval_ms: int = Field(default=100, ge=20, le=10_000)
    fail_fast: bool = True
    model: ModelSpec
    matrix: BenchmarkMatrix
    server: ServerSpec | None = None
    evaluation: EvaluationSpec | None = None
    load: LoadSpec | None = None


def load_experiment(path: Path) -> ExperimentSpec:
    """Load and validate an experiment YAML document.

    Parameters
    ----------
    path
        YAML configuration path.

    Returns
    -------
    ExperimentSpec
        Validated experiment definition.
    """
    with path.open(encoding="utf-8") as stream:
        content = yaml.safe_load(stream)
    return ExperimentSpec.model_validate(content)
