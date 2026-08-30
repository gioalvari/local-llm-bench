"""Static HTML reporting for benchmark run artifacts."""

import html
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _format_rate(value: object) -> str:
    return f"{float(value):.2f}" if isinstance(value, int | float) else "n/a"


def _format_gib(value: object) -> str:
    return f"{int(value) / (1024**3):.3f}" if isinstance(value, int | float) else "n/a"


def generate_report(run_dir: Path) -> Path:
    """Generate a self-contained HTML summary from raw run artifacts.

    Parameters
    ----------
    run_dir
        Directory containing a run manifest and measurements.

    Returns
    -------
    Path
        Generated report path.
    """
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing run manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    observations = _read_jsonl(run_dir / "measurements.jsonl")
    failures = _read_jsonl(run_dir / "failures.jsonl")
    rows: list[str] = []
    for observation in observations:
        if observation.get("dry_run"):
            continue
        cell = observation.get("cell", {})
        metrics = observation.get("metrics", {})
        peak_rss = observation.get("peak_process_tree_rss_bytes", 0)
        rate = metrics.get("avg_ts")
        speed_per_gib = (
            float(rate) / (int(peak_rss) / (1024**3))
            if isinstance(rate, int | float)
            and isinstance(peak_rss, int)
            and peak_rss > 0
            else None
        )
        values = [
            cell.get("cell_id"),
            cell.get("workload_name"),
            metrics.get("n_prompt"),
            metrics.get("n_gen"),
            cell.get("batch_size"),
            cell.get("ubatch_size"),
            cell.get("threads"),
            cell.get("gpu_layers"),
            cell.get("flash_attention"),
            metrics.get("backends"),
            _format_rate(rate),
            _format_gib(peak_rss),
            _format_rate(speed_per_gib),
        ]
        rows.append(
            "<tr>"
            + "".join(f"<td>{html.escape(str(value))}</td>" for value in values)
            + "</tr>"
        )
    run_id = html.escape(str(manifest.get("run_id", run_dir.name)))
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LocalLLM Bench - {run_id}</title>
  <style>
    :root {{ color-scheme: light dark; font-family: ui-monospace, monospace; }}
    body {{ margin: 0; padding: 2rem; background: #101417; color: #e8efe9; }}
    h1 {{ color: #9be15d; letter-spacing: -.04em; }}
    .meta {{ color: #9ea9a1; }}
    .panel {{ overflow-x: auto; border: 1px solid #344038; border-radius: 8px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: .82rem; }}
    th, td {{ padding: .65rem; border-bottom: 1px solid #2a332d; text-align: right; }}
    th:first-child, td:first-child,
    th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
    th {{ color: #101417; background: #9be15d; position: sticky; top: 0; }}
  </style>
</head>
<body>
  <h1>LocalLLM Bench</h1>
  <p class="meta">Run {run_id} |
    observations: {len(rows)} | failures: {len(failures)}</p>
  <p>Rates are llama.cpp microbenchmarks and exclude tokenization and sampling.</p>
  <div class="panel"><table>
    <thead><tr><th>Cell</th><th>Workload</th><th>Prompt</th><th>Gen</th>
    <th>Batch</th><th>Ubatch</th><th>Threads</th><th>GPU layers</th>
    <th>FA</th><th>Backend</th><th>tokens/s</th><th>Peak GiB</th>
    <th>tokens/s/GiB</th></tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table></div>
</body>
</html>
"""
    report_path = run_dir / "report.html"
    report_path.write_text(document, encoding="utf-8")
    return report_path
