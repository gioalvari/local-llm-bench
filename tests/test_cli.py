from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from localllm_bench import cli


@pytest.mark.parametrize(
    ("independent_runs", "expected"),
    [(1, "single"), (2, "study")],
)
def test_open_loop_dispatches_by_independent_run_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    independent_runs: int,
    expected: str,
) -> None:
    experiment = SimpleNamespace(
        open_loop=SimpleNamespace(independent_runs=independent_runs)
    )
    calls: list[str] = []

    def result(name: str) -> SimpleNamespace:
        return SimpleNamespace(model_dump=lambda **_: {"mode": name})

    def single(_: Any) -> SimpleNamespace:
        calls.append("single")
        return result("single")

    def study(_: Any) -> SimpleNamespace:
        calls.append("study")
        return result("study")

    monkeypatch.setattr(cli, "load_experiment", lambda _: experiment)
    monkeypatch.setattr(cli, "run_open_loop_benchmark", single)
    monkeypatch.setattr(cli, "run_repeated_open_loop_benchmark", study)
    cli.open_loop(tmp_path / "config.yaml")
    assert calls == [expected]
    assert f'"mode": "{expected}"' in capsys.readouterr().out
