"""Command-line interface for LocalLLM Bench."""

import json
from pathlib import Path
from typing import Annotated

import typer

from localllm_bench.comparison import VariantRuns, compare_variants
from localllm_bench.config import load_experiment
from localllm_bench.context_sweep import run_context_sweep
from localllm_bench.doctor import inspect_capabilities
from localllm_bench.evaluation import run_evaluation
from localllm_bench.factor_analysis import analyze_factor_run
from localllm_bench.load import run_load_benchmark
from localllm_bench.mlx_comparison import compare_mlx_evaluations
from localllm_bench.mlx_evaluation import evaluate_mlx_model
from localllm_bench.open_loop import run_open_loop_benchmark
from localllm_bench.planner import expand_plan
from localllm_bench.reporting import generate_report
from localllm_bench.rescore import rescore_run
from localllm_bench.runner import run_experiment
from localllm_bench.server import run_server_benchmark
from localllm_bench.training import run_training
from localllm_bench.training_data import prepare_training_dataset

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


@app.command()
def serve(
    config: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
) -> None:
    """Measure streaming latency through a managed llama-server process."""
    result = run_server_benchmark(load_experiment(config))
    typer.echo(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))


@app.command()
def evaluate(
    config: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
) -> None:
    """Run source-grounded, judge-independent quality evaluation."""
    result = run_evaluation(load_experiment(config))
    typer.echo(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))


@app.command()
def rescore(
    run_dir: Annotated[
        Path, typer.Argument(exists=True, file_okay=False, readable=True)
    ],
) -> None:
    """Recompute quality metrics from persisted model responses."""
    typer.echo(json.dumps(rescore_run(run_dir), indent=2, sort_keys=True))


@app.command()
def compare(
    micro_run: Annotated[
        list[Path], typer.Option("--micro-run", exists=True, file_okay=False)
    ],
    serving_run: Annotated[
        list[Path], typer.Option("--serving-run", exists=True, file_okay=False)
    ],
    quality_run: Annotated[
        list[Path], typer.Option("--quality-run", exists=True, file_okay=False)
    ],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
) -> None:
    """Compare matched microbenchmark, serving, and quality runs."""
    if not (len(micro_run) == len(serving_run) == len(quality_run)):
        raise typer.BadParameter("run options must have matching counts")
    variants = [
        VariantRuns(microbenchmark=micro, serving=serving, quality=quality)
        for micro, serving, quality in zip(
            micro_run, serving_run, quality_run, strict=True
        )
    ]
    result = compare_variants(variants, output_dir)
    typer.echo(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))


@app.command()
def load(
    config: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
) -> None:
    """Run synchronized closed-loop concurrent request waves."""
    result = run_load_benchmark(load_experiment(config))
    typer.echo(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))


@app.command("open-loop")
def open_loop(
    config: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
) -> None:
    """Run deterministic fixed-rate open-loop request traffic."""
    result = run_open_loop_benchmark(load_experiment(config))
    typer.echo(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))


@app.command("context")
def context_sweep(
    config: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
) -> None:
    """Measure exact prompt lengths across configured context windows."""
    result = run_context_sweep(load_experiment(config))
    typer.echo(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))


@app.command("analyze-factors")
def analyze_factors(
    run_dir: Annotated[
        Path, typer.Argument(exists=True, file_okay=False, readable=True)
    ],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
) -> None:
    """Analyze paired offload, Flash Attention, and batch effects."""
    result = analyze_factor_run(run_dir, output_dir)
    typer.echo(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))


@app.command("prepare-training-data")
def prepare_training_data(
    source: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
) -> None:
    """Validate and export document-disjoint MLX chat splits."""
    result = prepare_training_dataset(source, output_dir)
    typer.echo(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))


@app.command("train")
def train(
    config: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    model_path: Annotated[Path, typer.Option("--model-path")],
) -> None:
    """Run a managed local MLX QLoRA experiment."""
    result = run_training(load_experiment(config), model_path)
    typer.echo(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))


@app.command("evaluate-mlx")
def evaluate_mlx(
    config: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    model_path: Annotated[Path, typer.Option("--model-path")],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    adapter_path: Annotated[Path | None, typer.Option("--adapter-path")] = None,
) -> None:
    """Evaluate a base or adapted MLX model on the held-out split."""
    experiment = load_experiment(config)
    if experiment.training is None:
        raise typer.BadParameter("experiment does not define a training section")
    result = evaluate_mlx_model(
        model_path,
        experiment.training.source_dataset,
        output_dir,
        adapter_path=adapter_path,
        expected_model_sha256=experiment.training.model_sha256,
    )
    typer.echo(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))


@app.command("compare-mlx")
def compare_mlx(
    base_dir: Annotated[Path, typer.Option("--base-dir")],
    adapted_dir: Annotated[Path, typer.Option("--adapted-dir")],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
) -> None:
    """Compare frozen base and adapted MLX evaluation artifacts."""
    result = compare_mlx_evaluations(base_dir, adapted_dir, output_dir)
    typer.echo(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))
