"""Deterministic experiment matrix planning."""

import hashlib
import itertools
import json

from pydantic import BaseModel

from localllm_bench.config import ExperimentSpec, FlashAttention


class RunCell(BaseModel):
    """One concrete llama.cpp benchmark configuration."""

    cell_id: str
    workload_name: str
    prompt_tokens: int
    generation_tokens: int
    batch_size: int
    ubatch_size: int
    threads: int
    gpu_layers: int
    flash_attention: FlashAttention
    repetitions: int


class SkippedCell(BaseModel):
    """A matrix combination excluded before execution."""

    reason: str
    batch_size: int
    ubatch_size: int


class Plan(BaseModel):
    """Expanded executable cells and excluded combinations."""

    experiment_id: str
    cells: list[RunCell]
    skipped: list[SkippedCell]


def _cell_id(values: dict[str, object]) -> str:
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def expand_plan(spec: ExperimentSpec) -> Plan:
    """Expand an experiment matrix into deterministic run cells.

    Parameters
    ----------
    spec
        Validated experiment specification.

    Returns
    -------
    Plan
        Concrete cells and combinations skipped for safety.
    """
    cells: list[RunCell] = []
    skipped: list[SkippedCell] = []
    dimensions = itertools.product(
        spec.matrix.workloads,
        spec.matrix.batch_sizes,
        spec.matrix.ubatch_sizes,
        spec.matrix.threads,
        spec.matrix.gpu_layers,
        spec.matrix.flash_attention,
    )
    for workload, batch, ubatch, threads, gpu_layers, flash_attention in dimensions:
        if ubatch > batch:
            skipped.append(
                SkippedCell(
                    reason="ubatch_size exceeds batch_size",
                    batch_size=batch,
                    ubatch_size=ubatch,
                )
            )
            continue
        values = {
            "workload_name": workload.name,
            "prompt_tokens": workload.prompt_tokens,
            "generation_tokens": workload.generation_tokens,
            "batch_size": batch,
            "ubatch_size": ubatch,
            "threads": threads,
            "gpu_layers": gpu_layers,
            "flash_attention": flash_attention.value,
            "repetitions": spec.matrix.repetitions,
        }
        cells.append(
            RunCell(
                cell_id=_cell_id(values),
                workload_name=workload.name,
                prompt_tokens=workload.prompt_tokens,
                generation_tokens=workload.generation_tokens,
                batch_size=batch,
                ubatch_size=ubatch,
                threads=threads,
                gpu_layers=gpu_layers,
                flash_attention=flash_attention,
                repetitions=spec.matrix.repetitions,
            )
        )
    return Plan(experiment_id=spec.experiment_id, cells=cells, skipped=skipped)
