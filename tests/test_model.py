import hashlib
from pathlib import Path

import pytest

from localllm_bench.config import ModelSpec
from localllm_bench.model import model_args, validate_model


def test_model_args_support_remote_file() -> None:
    model = ModelSpec(
        name="tiny",
        hf_repo="owner/repo",
        hf_file="tiny.gguf",
        quantization="Q4",
        offline=True,
    )
    assert model_args(model) == [
        "--hf-repo",
        "owner/repo",
        "--hf-file",
        "tiny.gguf",
        "--offline",
    ]


def test_validate_model_checks_hash(tmp_path: Path) -> None:
    path = tmp_path / "model.gguf"
    path.write_bytes(b"model")
    validate_model(
        ModelSpec(
            name="tiny",
            path=path,
            sha256=hashlib.sha256(b"model").hexdigest(),
            quantization="Q4",
        )
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_model(
            ModelSpec(
                name="tiny",
                path=path,
                sha256="0" * 64,
                quantization="Q4",
            )
        )
