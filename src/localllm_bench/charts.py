"""Dependency-free, deterministic SVG documentation charts."""

import html
import json
from pathlib import Path
from typing import Any

_BACKGROUND = "#0d1117"
_PANEL = "#161b22"
_GRID = "#30363d"
_TEXT = "#e6edf3"
_MUTED = "#8b949e"
_GREEN = "#9be15d"
_CYAN = "#56d4dd"
_AMBER = "#f0b35a"
_PINK = "#e782a9"


def _svg(title: str, description: str, body: str, width: int, height: int) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
<title id="title">{html.escape(title)}</title>
<desc id="desc">{html.escape(description)}</desc>
<rect width="{width}" height="{height}" rx="18" fill="{_BACKGROUND}"/>
<style>text{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;fill:{_TEXT}}}.muted{{fill:{_MUTED}}}.label{{font-size:13px}}.small{{font-size:11px}}.title{{font-size:24px;font-weight:700}}.value{{font-size:15px;font-weight:700}}</style>
{body}</svg>
"""


def _line_points(
    values: list[float], x: float, y: float, width: float, height: float, maximum: float
) -> str:
    if len(values) == 1:
        return f"{x + width / 2:.2f},{y + height - values[0] / maximum * height:.2f}"
    return " ".join(
        f"{x + index * width / (len(values) - 1):.2f},{y + height - value / maximum * height:.2f}"
        for index, value in enumerate(values)
    )


def render_architecture() -> str:
    """Render the experiment lifecycle and artifact boundaries."""
    nodes = [
        (50, 125, 170, 74, "Experiment YAML", "Pinned matrix + hashes", _GREEN),
        (275, 125, 170, 74, "Capability probe", "Host + binary digests", _CYAN),
        (500, 125, 170, 74, "Isolated runners", "llama.cpp / MLX", _AMBER),
        (725, 125, 170, 74, "Raw artifacts", "JSONL + telemetry", _PINK),
        (950, 125, 170, 74, "Analysis", "Quality + Pareto", _GREEN),
    ]
    body = [
        '<text x="50" y="52" class="title">One experiment, three objective axes</text>',
        '<text x="50" y="80" class="muted label">Same checkpoint. Same hardware. Measured speed, memory, and quality.</text>',
    ]
    for index, (x, y, width, height, heading, detail, color) in enumerate(nodes):
        body.append(
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="12" fill="{_PANEL}" stroke="{color}" stroke-width="2"/>'
        )
        body.append(f'<text x="{x + 16}" y="{y + 30}" class="value">{heading}</text>')
        body.append(
            f'<text x="{x + 16}" y="{y + 54}" class="muted small">{detail}</text>'
        )
        if index < len(nodes) - 1:
            body.append(
                f'<path d="M {x + width + 8} {y + height / 2} H {nodes[index + 1][0] - 12}" stroke="{_GRID}" stroke-width="3" marker-end="url(#arrow)"/>'
            )
    body.extend(
        [
            '<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="#8b949e"/></marker></defs>',
            f'<rect x="160" y="245" width="250" height="58" rx="12" fill="{_PANEL}" stroke="{_CYAN}"/>',
            '<text x="180" y="271" class="value">Performance</text><text x="180" y="291" class="muted small">TTFT · token/s · throughput</text>',
            f'<rect x="465" y="245" width="250" height="58" rx="12" fill="{_PANEL}" stroke="{_AMBER}"/>',
            '<text x="485" y="271" class="value">Memory</text><text x="485" y="291" class="muted small">RSS · host memory · model size</text>',
            f'<rect x="770" y="245" width="250" height="58" rx="12" fill="{_PANEL}" stroke="{_PINK}"/>',
            '<text x="790" y="271" class="value">Quality</text><text x="790" y="291" class="muted small">Exact match · F1 · schema</text>',
        ]
    )
    return _svg(
        "LocalLLM Bench experiment lifecycle",
        "A reproducible pipeline from pinned YAML configuration through host probing, isolated runners, raw artifacts, and quality-aware analysis.",
        "\n".join(body),
        1170,
        350,
    )


def render_quantization(data: dict[str, Any]) -> str:
    """Render quantization memory, generation, and structured-output tradeoffs."""
    labels = [str(value) for value in data["labels"]]
    rss = [float(value) for value in data["peak_rss_mib"]]
    generation = [float(value) for value in data["generation_tokens_per_second"]]
    schema = [float(value) for value in data["engineered_schema_percent"]]
    colors = [_GREEN, _CYAN, _AMBER]
    body = [
        '<text x="45" y="48" class="title">Quantization is a three-way tradeoff</text>',
        '<text x="45" y="74" class="muted label">Lower memory, faster decode, or stronger structured output: no variant wins every axis.</text>',
    ]
    panels = [
        ("Peak serving RSS", "MiB · lower is better", rss, max(rss) * 1.12),
        (
            "Token generation",
            "token/s · higher is better",
            generation,
            max(generation) * 1.12,
        ),
        ("Engineered schema validity", "% · higher is better", schema, 100.0),
    ]
    for panel_index, (heading, subtitle, values, maximum) in enumerate(panels):
        x = 45 + panel_index * 365
        body.append(
            f'<rect x="{x}" y="105" width="330" height="285" rx="14" fill="{_PANEL}"/>'
        )
        body.append(f'<text x="{x + 20}" y="137" class="value">{heading}</text>')
        body.append(f'<text x="{x + 20}" y="158" class="muted small">{subtitle}</text>')
        for index, (label, value) in enumerate(zip(labels, values, strict=True)):
            bar_y = 190 + index * 62
            bar_width = 205 * value / maximum
            body.append(
                f'<text x="{x + 20}" y="{bar_y + 17}" class="label">{label}</text>'
            )
            body.append(
                f'<rect x="{x + 100}" y="{bar_y}" width="205" height="22" rx="6" fill="{_GRID}"/>'
            )
            body.append(
                f'<rect x="{x + 100}" y="{bar_y}" width="{bar_width:.2f}" height="22" rx="6" fill="{colors[index]}"/>'
            )
            suffix = "%" if panel_index == 2 else ""
            body.append(
                f'<text x="{x + 305}" y="{bar_y + 17}" text-anchor="end" class="small">{value:.1f}{suffix}</text>'
            )
    return _svg(
        "Q4, Q5, and Q8 quantization tradeoffs",
        "Three grouped horizontal bar charts compare peak serving RSS, generation throughput, and engineered JSON schema validity.",
        "\n".join(body),
        1170,
        430,
    )


def render_open_loop(data: dict[str, Any]) -> str:
    """Render offered load, achieved rate, goodput, and SLO collapse."""
    offered = [float(value) for value in data["offered_rps"]]
    achieved = [float(value) for value in data["achieved_rps"]]
    goodput = [float(value) for value in data["goodput_rps"]]
    slo = [float(value) for value in data["slo_attainment_percent"]]
    x, y, width, height = 80.0, 115.0, 720.0, 265.0
    maximum = 12.0
    body = [
        '<text x="45" y="48" class="title">Usable capacity ends before raw throughput</text>',
        '<text x="45" y="74" class="muted label">At 8 offered req/s the server still works, but only 54.2% of requests meet the 500 ms SLO.</text>',
    ]
    for tick in (0, 3, 6, 9, 12):
        tick_y = y + height - tick / maximum * height
        body.append(
            f'<line x1="{x}" y1="{tick_y:.2f}" x2="{x + width}" y2="{tick_y:.2f}" stroke="{_GRID}"/>'
        )
        body.append(
            f'<text x="{x - 12}" y="{tick_y + 4:.2f}" text-anchor="end" class="muted small">{tick}</text>'
        )
    for values, color in ((offered, _MUTED), (achieved, _CYAN), (goodput, _GREEN)):
        points = _line_points(values, x, y, width, height, maximum)
        body.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="4" stroke-linejoin="round"/>'
        )
        for point in points.split():
            px, py = point.split(",")
            body.append(f'<circle cx="{px}" cy="{py}" r="5" fill="{color}"/>')
    for index, rate in enumerate(offered):
        tick_x = x + index * width / (len(offered) - 1)
        body.append(
            f'<text x="{tick_x:.2f}" y="405" text-anchor="middle" class="small">{rate:.0f}</text>'
        )
    legends = (("Offered", _MUTED), ("Achieved", _CYAN), ("SLO goodput", _GREEN))
    for index, (name, color) in enumerate(legends):
        legend_x = 95 + index * 145
        body.append(
            f'<line x1="{legend_x}" y1="430" x2="{legend_x + 28}" y2="430" stroke="{color}" stroke-width="4"/><text x="{legend_x + 36}" y="434" class="small">{name}</text>'
        )
    body.append(
        f'<rect x="850" y="115" width="270" height="265" rx="14" fill="{_PANEL}"/>'
    )
    body.append('<text x="875" y="148" class="value">500 ms SLO attainment</text>')
    for index, (rate, percentage) in enumerate(zip(offered, slo, strict=True)):
        bar_y = 178 + index * 48
        body.append(
            f'<text x="875" y="{bar_y + 16}" class="small">{rate:.0f} req/s</text>'
        )
        body.append(
            f'<rect x="940" y="{bar_y}" width="145" height="20" rx="5" fill="{_GRID}"/>'
        )
        body.append(
            f'<rect x="940" y="{bar_y}" width="{145 * percentage / 100:.2f}" height="20" rx="5" fill="{_GREEN if percentage == 100 else _PINK}"/>'
        )
        body.append(
            f'<text x="1095" y="{bar_y + 15}" class="small">{percentage:.1f}%</text>'
        )
    return _svg(
        "Open-loop saturation and SLO goodput",
        "Line chart of offered, achieved, and SLO-compliant request rates plus SLO attainment bars from 2 to 12 requests per second.",
        "\n".join(body),
        1170,
        470,
    )


def render_context(data: dict[str, Any]) -> str:
    """Render prompt-length latency and reserved-window memory effects."""
    prompt_lengths = [float(value) for value in data["prompt_lengths"]]
    ttft = [float(value) for value in data["ttft_ms"]]
    windows = [float(value) for value in data["window_sizes"]]
    rss = [float(value) for value in data["window_rss_mib"]]
    body = [
        '<text x="45" y="48" class="title">Prompt occupancy drives latency; reserved window drives memory</text>',
        '<text x="45" y="74" class="muted label">Two controlled series separate tokens processed from KV capacity allocated.</text>',
    ]
    panels = [
        (
            60.0,
            "Actual prompt length",
            "Median TTFT (ms)",
            prompt_lengths,
            ttft,
            420.0,
            _PINK,
        ),
        (
            620.0,
            "Configured context window",
            "Peak RSS (MiB)",
            windows,
            rss,
            560.0,
            _CYAN,
        ),
    ]
    for x, heading, subtitle, labels, values, maximum, color in panels:
        body.append(
            f'<rect x="{x}" y="105" width="500" height="310" rx="14" fill="{_PANEL}"/>'
        )
        body.append(f'<text x="{x + 22}" y="138" class="value">{heading}</text>')
        body.append(f'<text x="{x + 22}" y="160" class="muted small">{subtitle}</text>')
        plot_x, plot_y, plot_width, plot_height = x + 55, 190.0, 400.0, 165.0
        for fraction in (0.0, 0.5, 1.0):
            grid_y = plot_y + plot_height - fraction * plot_height
            body.append(
                f'<line x1="{plot_x}" y1="{grid_y}" x2="{plot_x + plot_width}" y2="{grid_y}" stroke="{_GRID}"/>'
            )
            body.append(
                f'<text x="{plot_x - 10}" y="{grid_y + 4}" text-anchor="end" class="muted small">{maximum * fraction:.0f}</text>'
            )
        points = _line_points(values, plot_x, plot_y, plot_width, plot_height, maximum)
        body.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="4"/>'
        )
        for index, point in enumerate(points.split()):
            px, py = point.split(",")
            body.append(f'<circle cx="{px}" cy="{py}" r="5" fill="{color}"/>')
            body.append(
                f'<text x="{px}" y="382" text-anchor="middle" class="small">{labels[index]:.0f}</text>'
            )
    return _svg(
        "Context window and prompt length effects",
        "Two line charts show median time to first token as prompt length increases and peak RSS as the configured context window increases.",
        "\n".join(body),
        1170,
        455,
    )


def generate_charts(data_path: Path, output_dir: Path) -> list[Path]:
    """Generate every documentation chart from one versioned JSON document."""
    data = json.loads(data_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("chart data must be a JSON object")
    required = {"quantization", "open_loop", "context", "sources"}
    if set(data) != required:
        raise ValueError(f"chart data keys must be exactly {sorted(required)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "architecture.svg": render_architecture(),
        "quantization-tradeoffs.svg": render_quantization(data["quantization"]),
        "open-loop-capacity.svg": render_open_loop(data["open_loop"]),
        "context-effects.svg": render_context(data["context"]),
    }
    paths: list[Path] = []
    for filename, content in charts.items():
        path = output_dir / filename
        path.write_text(content, encoding="utf-8")
        paths.append(path)
    return paths
