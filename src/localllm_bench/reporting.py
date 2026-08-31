"""Static HTML reporting for benchmark run artifacts."""

import html
import json
import statistics
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


def _metric_card(label: str, value: str) -> str:
    return (
        '<div class="metric"><span class="metric-value">'
        f"{html.escape(value)}</span><span>{html.escape(label)}</span></div>"
    )


def _generate_server_report(
    run_dir: Path,
    manifest: dict[str, Any],
    failures: list[dict[str, Any]],
) -> str:
    requests = _read_jsonl(run_dir / "requests.jsonl")
    ttft_ms = [float(item["ttft_ns"]) / 1_000_000 for item in requests]
    e2e_ms = [float(item["e2e_latency_ns"]) / 1_000_000 for item in requests]
    decode_rates = [
        float(item["client_decode_tokens_per_second"])
        for item in requests
        if isinstance(item.get("client_decode_tokens_per_second"), int | float)
    ]
    cards = [
        _metric_card(
            "Model load",
            f"{float(manifest.get('model_load_time_ns', 0)) / 1_000_000:.2f} ms",
        ),
        _metric_card(
            "Median TTFT",
            f"{statistics.median(ttft_ms):.2f} ms" if ttft_ms else "n/a",
        ),
        _metric_card(
            "Median end-to-end",
            f"{statistics.median(e2e_ms):.2f} ms" if e2e_ms else "n/a",
        ),
        _metric_card(
            "Median decode",
            f"{statistics.median(decode_rates):.2f} token/s" if decode_rates else "n/a",
        ),
        _metric_card(
            "Peak process RSS",
            f"{_format_gib(manifest.get('peak_process_tree_rss_bytes'))} GiB",
        ),
    ]
    rows: list[str] = []
    for request in requests:
        values = [
            request.get("request_index"),
            f"{float(request['ttft_ns']) / 1_000_000:.2f}",
            f"{float(request['e2e_latency_ns']) / 1_000_000:.2f}",
            request.get("output_tokens"),
            _format_rate(request.get("client_decode_tokens_per_second")),
            request.get("event_count"),
        ]
        rows.append(
            "<tr>"
            + "".join(f"<td>{html.escape(str(value))}</td>" for value in values)
            + "</tr>"
        )
    return f"""
  <p class="meta">Requests: {len(requests)} | failures: {len(failures)}</p>
  <div class="metrics">{"".join(cards)}</div>
  <p>Client timings use monotonic timestamps from streamed completion events.</p>
  <div class="panel"><table>
    <thead><tr><th>Request</th><th>TTFT ms</th><th>E2E ms</th>
    <th>Output tokens</th><th>Client token/s</th><th>Events</th></tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table></div>"""


def _generate_microbenchmark_report(
    run_dir: Path,
    failures: list[dict[str, Any]],
) -> str:
    observations = _read_jsonl(run_dir / "measurements.jsonl")
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
    return f"""
  <p class="meta">Observations: {len(rows)} | failures: {len(failures)}</p>
  <p>Rates are llama.cpp microbenchmarks and exclude tokenization and sampling.</p>
  <div class="panel"><table>
    <thead><tr><th>Cell</th><th>Workload</th><th>Prompt</th><th>Gen</th>
    <th>Batch</th><th>Ubatch</th><th>Threads</th><th>GPU layers</th>
    <th>FA</th><th>Backend</th><th>tokens/s</th><th>Peak GiB</th>
    <th>tokens/s/GiB</th></tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table></div>"""


def _generate_quality_report(
    run_dir: Path,
    manifest: dict[str, Any],
    failures: list[dict[str, Any]],
) -> str:
    records = _read_jsonl(run_dir / "evaluations.jsonl")
    summary = manifest.get("summary", {})
    cards: list[str] = []
    rows: list[str] = []
    for arm, metrics in summary.items():
        cards.extend(
            [
                _metric_card(
                    f"{arm} accuracy", f"{float(metrics['answer_accuracy']):.1%}"
                ),
                _metric_card(
                    f"{arm} schema", f"{float(metrics['schema_valid_rate']):.1%}"
                ),
            ]
        )
        values = [
            arm,
            metrics.get("items"),
            f"{float(metrics['answer_accuracy']):.3f}",
            f"{float(metrics['exact_match']):.3f}",
            f"{float(metrics['token_f1']):.3f}",
            f"{float(metrics['scorable_response_rate']):.3f}",
            f"{float(metrics['schema_valid_rate']):.3f}",
            f"{float(metrics['numeric_accuracy']):.3f}",
            f"{float(metrics['unit_accuracy']):.3f}",
            f"{float(metrics['median_ttft_ms']):.2f}",
            f"{float(metrics['median_e2e_ms']):.2f}",
            f"{float(metrics['quality_adjusted_answers_per_second']):.3f}",
        ]
        rows.append(
            "<tr>"
            + "".join(f"<td>{html.escape(str(value))}</td>" for value in values)
            + "</tr>"
        )
    return f"""
  <p class="meta">Evaluations: {len(records)} | failures: {len(failures)}</p>
  <div class="metrics">{"".join(cards)}</div>
  <p>All acceptance metrics are deterministic and judge-independent.</p>
  <div class="panel"><table>
    <thead><tr><th>Prompt arm</th><th>Items</th><th>Accuracy</th>
    <th>Exact match</th><th>Token F1</th><th>Scorable</th><th>Schema</th>
    <th>Numeric</th><th>Unit</th><th>TTFT ms</th><th>E2E ms</th>
    <th>Correct/s</th></tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table></div>"""


def _generate_load_report(
    run_dir: Path,
    manifest: dict[str, Any],
    _failures: list[dict[str, Any]],
) -> str:
    records = _read_jsonl(run_dir / "requests.jsonl")
    failed_requests = sum("error" in record for record in records)
    summary = manifest.get("summary", [])
    rows: list[str] = []
    for level in summary:
        values = [
            level.get("concurrency"),
            level.get("requests"),
            f"{float(level['error_rate']):.1%}",
            f"{float(level['aggregate_output_tokens_per_second']):.2f}",
            f"{float(level['requests_per_second']):.2f}",
            f"{float(level['median_ttft_ms']):.2f}",
            f"{float(level['p95_ttft_ms']):.2f}",
            f"{float(level['median_e2e_ms']):.2f}",
            f"{float(level['p95_e2e_ms']):.2f}",
            f"{float(level['max_wave_launch_spread_ms']):.2f}",
            _format_gib(level.get("peak_process_tree_rss_bytes")),
        ]
        rows.append(
            "<tr>"
            + "".join(f"<td>{html.escape(str(value))}</td>" for value in values)
            + "</tr>"
        )
    peak_throughput = max(
        (float(level["aggregate_output_tokens_per_second"]) for level in summary),
        default=0.0,
    )
    cards = [
        _metric_card("Measured requests", str(len(records))),
        _metric_card("Peak aggregate throughput", f"{peak_throughput:.2f} token/s"),
        _metric_card(
            "Peak process RSS",
            f"{_format_gib(manifest.get('peak_process_tree_rss_bytes'))} GiB",
        ),
    ]
    return f"""
  <p class="meta">Requests: {len(records)} | failures: {failed_requests}</p>
  <div class="metrics">{"".join(cards)}</div>
  <p>Closed-loop synchronized waves; warm-up and model load are excluded.</p>
  <div class="panel"><table>
    <thead><tr><th>Concurrency</th><th>Requests</th><th>Error rate</th>
    <th>Output token/s</th><th>Request/s</th><th>Median TTFT ms</th>
    <th>P95 TTFT ms</th><th>Median E2E ms</th><th>P95 E2E ms</th>
    <th>Max launch spread ms</th><th>Peak RSS GiB</th></tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table></div>"""


def _generate_open_loop_report(
    run_dir: Path,
    manifest: dict[str, Any],
    _failures: list[dict[str, Any]],
) -> str:
    records = _read_jsonl(run_dir / "requests.jsonl")
    failed_requests = sum("error" in record for record in records)
    summary = manifest.get("summary", [])
    rows: list[str] = []
    for level in summary:
        values = [
            f"{float(level['offered_requests_per_second']):.2f}",
            level.get("requests"),
            f"{float(level['achieved_requests_per_second']):.2f}",
            f"{float(level['aggregate_output_tokens_per_second']):.2f}",
            f"{float(level['goodput_requests_per_second']):.2f}",
            f"{float(level['slo_attainment_rate']):.1%}",
            f"{float(level['error_rate']):.1%}",
            f"{float(level['median_ttft_ms']):.2f}",
            f"{float(level['p95_ttft_ms']):.2f}",
            f"{float(level['median_e2e_ms']):.2f}",
            f"{float(level['p95_e2e_ms']):.2f}",
            f"{float(level['p95_client_schedule_delay_ms']):.2f}",
            level.get("max_client_in_flight"),
            _format_gib(level.get("peak_process_tree_rss_bytes")),
        ]
        rows.append(
            "<tr>"
            + "".join(f"<td>{html.escape(str(value))}</td>" for value in values)
            + "</tr>"
        )
    peak_throughput = max(
        (float(level["aggregate_output_tokens_per_second"]) for level in summary),
        default=0.0,
    )
    cards = [
        _metric_card("Measured requests", str(len(records))),
        _metric_card("Peak aggregate throughput", f"{peak_throughput:.2f} token/s"),
        _metric_card(
            "Peak process RSS",
            f"{_format_gib(manifest.get('peak_process_tree_rss_bytes'))} GiB",
        ),
    ]
    return f"""
  <p class="meta">Requests: {len(records)} | failures: {failed_requests}</p>
  <div class="metrics">{"".join(cards)}</div>
  <p>Deterministic fixed-rate arrivals; warm-up and model load are excluded.</p>
  <div class="panel"><table>
    <thead><tr><th>Offered req/s</th><th>Requests</th><th>Achieved req/s</th>
    <th>Output token/s</th><th>Goodput req/s</th><th>SLO attainment</th>
    <th>Error rate</th><th>Median TTFT ms</th><th>P95 TTFT ms</th>
    <th>Median E2E ms</th><th>P95 E2E ms</th><th>P95 schedule delay ms</th>
    <th>Max in-flight</th><th>Peak RSS GiB</th></tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table></div>"""


def _generate_context_report(
    run_dir: Path,
    manifest: dict[str, Any],
    _failures: list[dict[str, Any]],
) -> str:
    records = _read_jsonl(run_dir / "requests.jsonl")
    failed_requests = sum("error" in record for record in records)
    summary = manifest.get("summary", [])
    rows: list[str] = []
    for case in summary:
        values = [
            case.get("case"),
            case.get("series"),
            case.get("context_size"),
            case.get("prompt_tokens"),
            f"{float(case['model_load_time_ms']):.2f}",
            f"{float(case['median_prompt_eval_ms']):.2f}",
            f"{float(case['median_prompt_tokens_per_second']):.2f}",
            f"{float(case['median_ttft_ms']):.2f}",
            f"{float(case['p95_ttft_ms']):.2f}",
            f"{float(case['median_e2e_ms']):.2f}",
            f"{float(case['median_decode_tokens_per_second']):.2f}",
            _format_gib(case.get("peak_process_tree_rss_bytes")),
            f"{float(case['error_rate']):.1%}",
        ]
        rows.append(
            "<tr>"
            + "".join(f"<td>{html.escape(str(value))}</td>" for value in values)
            + "</tr>"
        )
    cards = [
        _metric_card("Measured requests", str(len(records))),
        _metric_card("Context cases", str(len(summary))),
        _metric_card("Failures", str(failed_requests)),
    ]
    return f"""
  <p class="meta">Requests: {len(records)} | failures: {failed_requests}</p>
  <div class="metrics">{"".join(cards)}</div>
  <p>Exact token-ID prompts; each context case uses a fresh server.</p>
  <div class="panel"><table>
    <thead><tr><th>Case</th><th>Series</th><th>Context</th><th>Prompt tokens</th>
    <th>Load ms</th><th>Prompt eval ms</th><th>Prompt token/s</th>
    <th>Median TTFT ms</th><th>P95 TTFT ms</th><th>Median E2E ms</th>
    <th>Decode token/s</th><th>Peak RSS GiB</th><th>Error rate</th></tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table></div>"""


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
    failures = _read_jsonl(run_dir / "failures.jsonl")
    run_type = manifest.get("run_type")
    if run_type == "llama-server":
        content = _generate_server_report(run_dir, manifest, failures)
    elif run_type == "quality-evaluation":
        content = _generate_quality_report(run_dir, manifest, failures)
    elif run_type == "concurrency-load":
        content = _generate_load_report(run_dir, manifest, failures)
    elif run_type == "open-loop-load":
        content = _generate_open_loop_report(run_dir, manifest, failures)
    elif run_type == "context-sweep":
        content = _generate_context_report(run_dir, manifest, failures)
    else:
        content = _generate_microbenchmark_report(run_dir, failures)
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
    .metrics {{ display: grid;
      grid-template-columns: repeat(auto-fit, minmax(11rem, 1fr));
      gap: .75rem; margin: 1.5rem 0; }}
    .metric {{ display: flex; flex-direction: column; gap: .25rem; padding: 1rem;
      border: 1px solid #344038; border-radius: 8px; background: #171d19; }}
    .metric-value {{ color: #9be15d; font-size: 1.35rem; font-weight: 700; }}
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
  <p class="meta">Run {run_id}</p>
  {content}
</body>
</html>
"""
    report_path = run_dir / "report.html"
    report_path.write_text(document, encoding="utf-8")
    return report_path
