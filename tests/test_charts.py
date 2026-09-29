import json
from pathlib import Path

import pytest

from localllm_bench.charts import generate_charts


def _data() -> dict[str, object]:
    return {
        "quantization": {
            "labels": ["Q4", "Q5", "Q8"],
            "peak_rss_mib": [500, 550, 650],
            "generation_tokens_per_second": [280, 260, 255],
            "engineered_schema_percent": [40, 80, 60],
        },
        "open_loop": {
            "offered_rps": [2, 4, 8, 12],
            "achieved_rps": [2, 4, 7.5, 8],
            "goodput_rps": [2, 4, 4, 0.2],
            "slo_attainment_percent": [100, 100, 50, 2],
        },
        "context": {
            "prompt_lengths": [128, 512, 1024, 2048],
            "ttft_ms": [25, 93, 188, 395],
            "window_sizes": [256, 1024, 2048, 4096],
            "window_rss_mib": [495, 503, 516, 540],
        },
        "sources": ["results/source.md"],
    }


def test_generate_charts_is_deterministic_and_accessible(tmp_path: Path) -> None:
    data = tmp_path / "data.json"
    data.write_text(json.dumps(_data()), encoding="utf-8")
    first = generate_charts(data, tmp_path / "images")
    first_contents = [path.read_text(encoding="utf-8") for path in first]
    second = generate_charts(data, tmp_path / "images")
    assert [path.read_text(encoding="utf-8") for path in second] == first_contents
    assert {path.name for path in first} == {
        "architecture.svg",
        "quantization-tradeoffs.svg",
        "open-loop-capacity.svg",
        "context-effects.svg",
    }
    assert all('<title id="title">' in content for content in first_contents)
    assert all('<desc id="desc">' in content for content in first_contents)


def test_generate_charts_rejects_unexpected_data_keys(tmp_path: Path) -> None:
    data = tmp_path / "data.json"
    data.write_text(json.dumps({"unexpected": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="keys must be exactly"):
        generate_charts(data, tmp_path / "images")
