from pathlib import Path

from localllm_bench.artifacts import directory_sha256, file_sha256


def test_directory_sha256_ignores_transient_cache(tmp_path: Path) -> None:
    (tmp_path / "model.bin").write_bytes(b"weights")
    first = directory_sha256(tmp_path)
    cache = tmp_path / ".cache"
    cache.mkdir()
    (cache / "metadata.json").write_text("changed", encoding="utf-8")
    assert directory_sha256(tmp_path) == first
    (tmp_path / "model.bin").write_bytes(b"new weights")
    assert directory_sha256(tmp_path) != first


def test_file_sha256(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"artifact")
    assert file_sha256(path) == (
        "c7c5c1d70c5dec4416ab6158afd0b223ef40c29b1dc1f97ed9428b94d4cadb1c"
    )
