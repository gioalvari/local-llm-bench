"""Model artifact validation and backend arguments."""

import hashlib

from localllm_bench.config import ModelSpec


def model_args(model: ModelSpec) -> list[str]:
    """Build common llama.cpp model source arguments."""
    if model.path is not None:
        arguments = ["--model", str(model.path)]
    else:
        arguments = ["--hf-repo", str(model.hf_repo)]
        if model.hf_file is not None:
            arguments.extend(["--hf-file", model.hf_file])
    if model.offline:
        arguments.append("--offline")
    return arguments


def validate_model(model: ModelSpec) -> None:
    """Check local model presence and its optional SHA-256 digest."""
    if model.path is not None and not model.path.is_file():
        raise FileNotFoundError(f"model file does not exist: {model.path}")
    if model.path is None or model.sha256 is None:
        return
    digest = hashlib.sha256()
    with model.path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    observed = digest.hexdigest()
    if observed.lower() != model.sha256.lower():
        raise ValueError(
            f"model SHA-256 mismatch: expected {model.sha256}, observed {observed}"
        )
