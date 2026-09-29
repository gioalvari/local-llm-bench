"""Validated experiment configuration."""

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, Field, PositiveFloat, PositiveInt, model_validator

NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveFiniteFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]
MAX_ARRIVALS_PER_WINDOW = 1_000_000


class FlashAttention(StrEnum):
    """Supported llama.cpp Flash Attention modes."""

    OFF = "off"
    ON = "on"
    AUTO = "auto"


class PromptArm(StrEnum):
    """Frozen prompt variants used for quality comparisons."""

    ZERO_SHOT = "zero-shot"
    ENGINEERED = "engineered"


class ArrivalProcess(StrEnum):
    """Supported open-loop arrival processes."""

    FIXED = "fixed"
    POISSON = "poisson"


class RateOrderProtocol(StrEnum):
    """Supported open-loop rate execution orders."""

    CONFIGURED = "configured-order-v1"
    CYCLIC = "cyclic-rotation-v1"


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


class OpenLoopSpec(BaseModel):
    """Open-loop serving workload with a reproducible arrival process."""

    prompt_dataset: Path
    arrival_rates_rps: list[PositiveFiniteFloat] = Field(min_length=1)
    arrival_process: ArrivalProcess = ArrivalProcess.FIXED
    arrival_seed: NonNegativeInt = 42
    duration_seconds: PositiveFiniteFloat = 3.0
    warmup_requests: NonNegativeInt = 1
    server_slots: PositiveInt = 4
    max_client_workers: PositiveInt = 32
    latency_slo_ms: PositiveFloat = 500.0
    cooldown_seconds: Annotated[float, Field(ge=0)] = 0.5
    independent_runs: PositiveInt = 1
    bootstrap_iterations: PositiveInt = 10_000
    bootstrap_seed: NonNegativeInt = 42
    rate_order_protocol: RateOrderProtocol = RateOrderProtocol.CONFIGURED
    rate_order_offset: NonNegativeInt = 0

    @model_validator(mode="after")
    def validate_rates(self) -> "OpenLoopSpec":
        """Require a unique valid rate execution order."""
        rates = [float(value) for value in self.arrival_rates_rps]
        if len(rates) != len(set(rates)):
            raise ValueError("arrival_rates_rps must be unique")
        if self.rate_order_protocol is RateOrderProtocol.CONFIGURED and rates != sorted(
            rates
        ):
            raise ValueError("configured arrival_rates_rps must be increasing")
        if self.rate_order_offset >= len(rates):
            raise ValueError("rate_order_offset must index arrival_rates_rps")
        canonical_rates = sorted(rates)
        if self.rate_order_protocol is RateOrderProtocol.CYCLIC:
            expected = (
                canonical_rates[self.rate_order_offset :]
                + canonical_rates[: self.rate_order_offset]
            )
            if rates != expected:
                raise ValueError(
                    "cyclic arrival rates must match the declared rotation offset"
                )
        elif self.rate_order_offset != 0:
            raise ValueError("configured rate order requires rate_order_offset 0")
        if any(
            rate * float(self.duration_seconds) > MAX_ARRIVALS_PER_WINDOW
            for rate in rates
        ):
            raise ValueError(
                f"open-loop windows support at most {MAX_ARRIVALS_PER_WINDOW} "
                "expected arrivals"
            )
        return self


class ContextCaseSpec(BaseModel):
    """One configured context window and exact prompt length."""

    name: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    series: list[Literal["window-size", "prompt-length"]] = Field(min_length=1)
    context_size: PositiveInt
    prompt_tokens: PositiveInt


class ContextSweepSpec(BaseModel):
    """Exact-token context-length benchmark."""

    corpus: Path
    repetitions: PositiveInt = 3
    warmup_requests: NonNegativeInt = 1
    output_tokens: PositiveInt = 64
    cases: list[ContextCaseSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_cases(self) -> "ContextSweepSpec":
        """Require unique cases with enough context for prompt and output."""
        names = [case.name for case in self.cases]
        if len(names) != len(set(names)):
            raise ValueError("context sweep case names must be unique")
        for case in self.cases:
            if len(case.series) != len(set(case.series)):
                raise ValueError("context sweep case series must be unique")
            if case.context_size < case.prompt_tokens + self.output_tokens:
                raise ValueError(
                    "context_size must fit prompt_tokens and output_tokens"
                )
        window_cases = [case for case in self.cases if "window-size" in case.series]
        prompt_cases = [case for case in self.cases if "prompt-length" in case.series]
        if window_cases and len({case.prompt_tokens for case in window_cases}) != 1:
            raise ValueError("window-size series must keep prompt_tokens fixed")
        if prompt_cases and len({case.context_size for case in prompt_cases}) != 1:
            raise ValueError("prompt-length series must keep context_size fixed")
        return self


class TrainingSpec(BaseModel):
    """Apple-native MLX QLoRA smoke-training configuration."""

    source_dataset: Path
    model: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    model_sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    iterations: PositiveInt = 30
    batch_size: PositiveInt = 1
    gradient_accumulation_steps: PositiveInt = 1
    num_layers: PositiveInt = 4
    learning_rate: PositiveFloat = 1e-5
    max_seq_length: PositiveInt = 512
    mask_prompt: bool = True
    seed: int = 42


class ExperimentSpec(BaseModel):
    """Top-level benchmark experiment specification."""

    schema_version: Literal["1", "2"] = "1"
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
    open_loop: OpenLoopSpec | None = None
    context_sweep: ContextSweepSpec | None = None
    training: TrainingSpec | None = None

    @model_validator(mode="after")
    def validate_schema_features(self) -> "ExperimentSpec":
        """Require schema v2 for expanded open-loop protocols."""
        if (
            self.open_loop is not None
            and (
                self.open_loop.arrival_process is ArrivalProcess.POISSON
                or self.open_loop.independent_runs > 1
                or self.open_loop.rate_order_protocol is RateOrderProtocol.CYCLIC
            )
            and self.schema_version != "2"
        ):
            raise ValueError("expanded open_loop protocols require schema_version 2")
        if self.open_loop is not None and 1 < self.open_loop.independent_runs < 5:
            raise ValueError("repeated open_loop studies require at least 5 runs")
        if self.open_loop is not None and 1 < self.open_loop.independent_runs < len(
            self.open_loop.arrival_rates_rps
        ):
            raise ValueError(
                "repeated open_loop studies must cover every rate position"
            )
        if (
            self.open_loop is not None
            and self.open_loop.independent_runs > 1
            and self.open_loop.independent_runs % len(self.open_loop.arrival_rates_rps)
            != 0
        ):
            raise ValueError(
                "repeated open_loop run count must be a multiple of the rate count"
            )
        return self


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
