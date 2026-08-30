from pathlib import Path

from localllm_bench.doctor import (
    _executable,
    _first_line,
    inspect_capabilities,
    report_as_dict,
)


def test_inspect_capabilities_reports_host(tmp_path: Path) -> None:
    report = inspect_capabilities(tmp_path)
    serialized = report_as_dict(report)
    assert report.logical_cpus >= 1
    assert report.memory_bytes > 0
    assert report.disk_free_bytes > 0
    assert isinstance(serialized["llama_bench"]["available"], bool)


def test_missing_executable_is_reported() -> None:
    result = _executable("an-executable-that-does-not-exist")
    assert result.available is False
    assert result.path is None


def test_first_line_handles_missing_command() -> None:
    assert _first_line(["an-executable-that-does-not-exist"]) is None
