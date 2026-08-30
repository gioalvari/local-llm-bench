from localllm_bench.config import ExperimentSpec
from localllm_bench.planner import expand_plan


def _spec() -> ExperimentSpec:
    return ExperimentSpec.model_validate(
        {
            "experiment_id": "test",
            "model": {
                "name": "tiny",
                "hf_repo": "owner/model:Q4_K_M",
                "quantization": "Q4_K_M",
            },
            "matrix": {
                "repetitions": 2,
                "workloads": [
                    {"name": "short", "prompt_tokens": 16, "generation_tokens": 8}
                ],
                "batch_sizes": [32],
                "ubatch_sizes": [16, 64],
                "threads": [1, 2],
                "gpu_layers": [0],
                "flash_attention": ["off"],
            },
        }
    )


def test_plan_filters_invalid_batch_combinations() -> None:
    plan = expand_plan(_spec())
    assert len(plan.cells) == 2
    assert len(plan.skipped) == 2
    assert all(cell.ubatch_size == 16 for cell in plan.cells)


def test_plan_ids_are_stable() -> None:
    first = expand_plan(_spec())
    second = expand_plan(_spec())
    assert [cell.cell_id for cell in first.cells] == [
        cell.cell_id for cell in second.cells
    ]
