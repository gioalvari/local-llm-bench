"""Command-line interface for LocalLLM Bench."""

import json
from pathlib import Path
from typing import Annotated

import typer

from localllm_bench.config import load_experiment
from localllm_bench.doctor import inspect_capabilities
from localllm_bench.planner import expand_plan
from localllm_bench.reporting import generate_report
from localllm_bench.runner import run_experiment

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)


@app.command()
def doctor() -> None:
    """Inspect local inference capabilities and storage."""
    report = inspect_capabilities()
    typer.echo(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))


@app.command()
def plan(
    config: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
) -> None:
    """Validate an experiment and display its concrete cells."""
    result = expand_plan(load_experiment(config))
    typer.echo(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))


@app.command("run")
def run_command(
    config: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Write commands without executing them.")
    ] = False,
) -> None:
    """Run all valid cells in an experiment."""
    result = run_experiment(load_experiment(config), dry_run=dry_run)
    typer.echo(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))


@app.command()
def report(
    run_dir: Annotated[
        Path, typer.Argument(exists=True, file_okay=False, readable=True)
    ],
) -> None:
    """Generate a static report for a completed run."""
    typer.echo(str(generate_report(run_dir)))
