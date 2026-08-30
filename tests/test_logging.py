from localllm_bench.logging import get_logger


def test_get_logger_uses_source_stem() -> None:
    assert get_logger("/tmp/worker.py").name == "localllm_bench.worker"
