from pathlib import Path

from localllm_bench.config import TrainingSpec
from localllm_bench.training import build_training_command, parse_training_metrics


def test_build_training_command_is_explicit() -> None:
    spec = TrainingSpec(
        source_dataset=Path("source.jsonl"),
        model="owner/model",
        model_revision="revision",
        model_sha256="a" * 64,
        iterations=10,
        batch_size=1,
        gradient_accumulation_steps=2,
        num_layers=4,
        learning_rate=1e-5,
        max_seq_length=256,
        mask_prompt=True,
        seed=42,
    )
    command = build_training_command(
        spec, Path("model"), Path("data"), Path("adapters")
    )
    assert command[:4] == ["mlx_lm.lora", "--model", "model", "--train"]
    assert command[command.index("--iters") + 1] == "10"
    assert command[command.index("--grad-accumulation-steps") + 1] == "2"
    assert "--mask-prompt" in command


def test_parse_training_metrics() -> None:
    metrics = parse_training_metrics(
        "Iter 1: Val loss 4.577\n"
        "Iter 10: Train loss 3.803, Peak mem 0.452 GB\n"
        "Iter 30: Val loss 2.631\n"
        "Iter 30: Train loss 1.820, Peak mem 0.446 GB\n"
    )
    assert metrics == {
        "initial_validation_loss": 4.577,
        "final_validation_loss": 2.631,
        "final_train_loss": 1.82,
        "peak_memory_gb": 0.452,
    }
