import hashlib
import json
import sys
from pathlib import Path

import pytest

from localllm_bench.config import ArrivalProcess, ExperimentSpec, OpenLoopSpec
from localllm_bench.open_loop import (
    LoadPrompt,
    arrival_offsets_ns,
    build_arrival_schedule,
    load_prompts,
    prompt_dataset_sha256,
    request_count,
    run_open_loop_benchmark,
    summarize_rate,
)


def _spec(
    tmp_path: Path,
    executable: str,
    *,
    arrival_process: ArrivalProcess = ArrivalProcess.FIXED,
    arrival_rate: float = 50,
    duration_seconds: float = 0.06,
) -> ExperimentSpec:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        '{"prompt_id":"first","prompt":"first prompt"}\n'
        '{"prompt_id":"second","prompt":"second prompt"}\n',
        encoding="utf-8",
    )
    return ExperimentSpec.model_validate(
        {
            "schema_version": (
                "2" if arrival_process is ArrivalProcess.POISSON else "1"
            ),
            "experiment_id": "open-loop-test",
            "output_dir": tmp_path / "runs",
            "llama_server_binary": executable,
            "sample_interval_ms": 20,
            "model": {"name": "tiny", "path": str(model), "quantization": "Q4"},
            "matrix": {
                "workloads": [
                    {"name": "short", "prompt_tokens": 8, "generation_tokens": 4}
                ]
            },
            "server": {
                "prompt": "hello",
                "output_tokens": 2,
                "context_size": 64,
                "batch_size": 32,
                "ubatch_size": 16,
                "threads": 1,
                "gpu_layers": 0,
                "flash_attention": "off",
                "startup_timeout_seconds": 5,
                "request_timeout_seconds": 5,
            },
            "open_loop": {
                "prompt_dataset": str(prompts),
                "arrival_rates_rps": [arrival_rate],
                "arrival_process": arrival_process.value,
                "duration_seconds": duration_seconds,
                "warmup_requests": 1,
                "server_slots": 2,
                "max_client_workers": 4,
                "latency_slo_ms": 100,
                "cooldown_seconds": 0,
            },
        }
    )


def test_load_prompts_and_digest(tmp_path: Path) -> None:
    path = tmp_path / "prompts.jsonl"
    path.write_text('{"prompt_id":"one","prompt":"hello"}\n', encoding="utf-8")
    assert load_prompts(path) == [LoadPrompt(prompt_id="one", prompt="hello")]
    assert len(prompt_dataset_sha256(path)) == 64
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_prompts(path)
    path.write_text(
        '{"prompt_id":"one","prompt":"hello"}\n{"prompt_id":"one","prompt":"again"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_prompts(path)


def test_request_count_has_at_least_one_arrival() -> None:
    assert request_count(8, 3) == 24
    assert request_count(2.5, 3) == 8
    assert request_count(0.1, 0.1) == 1


def test_fixed_arrival_offsets_preserve_existing_schedule() -> None:
    assert arrival_offsets_ns(2.5, 1.0, ArrivalProcess.FIXED, 42) == [
        0,
        400_000_000,
        800_000_000,
    ]


def test_poisson_arrival_offsets_are_seeded_and_bounded() -> None:
    first = arrival_offsets_ns(8, 3, ArrivalProcess.POISSON, 42)
    repeated = arrival_offsets_ns(8, 3, ArrivalProcess.POISSON, 42)
    changed = arrival_offsets_ns(8, 3, ArrivalProcess.POISSON, 43)
    assert first == repeated
    assert first != changed
    assert first == sorted(first)
    assert all(0 <= offset < 3_000_000_000 for offset in first)
    gaps = {right - left for left, right in zip(first, first[1:], strict=False)}
    assert len(gaps) > 1
    assert first[:5] == [
        20_907_706,
        186_119_522,
        204_945_892,
        206_686_488,
        230_942_040,
    ]


def test_poisson_rate_stream_is_independent_and_may_be_empty() -> None:
    smaller = OpenLoopSpec(
        prompt_dataset=Path("prompts.jsonl"),
        arrival_rates_rps=[2, 8],
        arrival_process=ArrivalProcess.POISSON,
        arrival_seed=7,
    )
    expanded = smaller.model_copy(update={"arrival_rates_rps": [2, 4, 8]})
    smaller_schedule = build_arrival_schedule(smaller)
    expanded_schedule = build_arrival_schedule(expanded)
    assert smaller_schedule["windows"][1] == expanded_schedule["windows"][2]
    assert arrival_offsets_ns(1e-6, 1e-6, ArrivalProcess.POISSON, 42) == []


def test_summarize_rate_reports_goodput_and_in_flight() -> None:
    records = [
        {
            "ttft_ns": 10_000_000,
            "e2e_latency_ns": 100_000_000,
            "output_tokens": 10,
            "client_schedule_delay_ns": 100_000,
            "client_started_offset_ns": 0,
            "client_completed_offset_ns": 100_000_000,
        },
        {
            "ttft_ns": 20_000_000,
            "e2e_latency_ns": 600_000_000,
            "output_tokens": 10,
            "client_schedule_delay_ns": 200_000,
            "client_started_offset_ns": 50_000_000,
            "client_completed_offset_ns": 650_000_000,
        },
        {
            "error": "failed",
            "client_schedule_delay_ns": 300_000,
            "client_started_offset_ns": 60_000_000,
            "client_completed_offset_ns": 70_000_000,
        },
    ]
    summary = summarize_rate(
        3,
        1_000_000_000,
        records,
        [{"process_tree_rss_bytes": 2048, "monotonic_offset_ns": 0}],
        500,
    )
    assert summary["achieved_requests_per_second"] == 2.0
    assert summary["goodput_requests_per_second"] == 1.0
    assert summary["slo_attainment_rate"] == pytest.approx(1 / 3)
    assert summary["max_client_in_flight"] == 3
    assert summary["p95_client_schedule_delay_ms"] == 0.3


def test_summarize_empty_poisson_window_marks_statistics_undefined() -> None:
    summary = summarize_rate(1, 1_000_000_000, [], [], 500, 1_000_000_000)
    assert summary["realized_offered_requests_per_second"] == 0.0
    assert summary["achieved_requests_per_second"] == 0.0
    assert summary["error_rate"] is None
    assert summary["slo_attainment_rate"] is None
    assert summary["median_ttft_ms"] is None
    assert summary["p95_client_schedule_delay_ms"] is None


def test_run_open_loop_with_threaded_fake_server(tmp_path: Path) -> None:
    script = tmp_path / "fake_server.py"
    script.write_text(
        """import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--port', type=int)
args, _ = parser.parse_known_args()

class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{\"status\":\"ok\"}')

    def do_POST(self):
        length = int(self.headers.get('Content-Length', '0'))
        self.rfile.read(length)
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.end_headers()
        self.wfile.write(b'data: {\"content\":\"A\",\"stop\":false}\\n\\n')
        self.wfile.flush()
        time.sleep(0.04)
        final = {'content': 'B', 'stop': True, 'tokens_predicted': 2,
                 'timings': {'predicted_n': 2}}
        self.wfile.write(('data: ' + json.dumps(final) + '\\n\\n').encode())
        self.wfile.flush()

ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()
""",
        encoding="utf-8",
    )
    launcher = tmp_path / "fake-server"
    launcher.write_text(
        f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8"
    )
    launcher.chmod(0o755)
    result = run_open_loop_benchmark(_spec(tmp_path, str(launcher)))
    assert result.completed_requests == 3
    assert result.failed_requests == 0
    max_in_flight = result.summary[0]["max_client_in_flight"]
    assert isinstance(max_in_flight, int)
    assert max_in_flight >= 2
    assert result.summary[0]["realized_offered_requests_per_second"] == 50.0
    manifest = json.loads(
        (result.run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["artifact_schema_version"] == "2"
    assert manifest["warmup_completed"] == 1
    assert manifest["prompt_count"] == 2
    assert manifest["context_tokens_per_slot"] == 64
    assert manifest["effective_server"]["context_size"] == 128
    schedule_path = result.run_dir / "arrival_schedule.json"
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    assert schedule["algorithm"] == "fixed-spacing-v1"
    assert schedule["windows"][0]["scheduled_offsets_ns"] == [
        0,
        20_000_000,
        40_000_000,
    ]
    assert (
        manifest["arrival_schedule_sha256"]
        == hashlib.sha256(schedule_path.read_bytes()).hexdigest()
    )
    requests = [
        json.loads(line)
        for line in (result.run_dir / "requests.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [record["scheduled_offset_ns"] for record in requests] == [
        0,
        20_000_000,
        40_000_000,
    ]
    assert [record["rate_request_index"] for record in requests] == [0, 1, 2]
    assert [record["prompt_id"] for record in requests] == ["first", "second", "first"]

    poisson = run_open_loop_benchmark(
        _spec(
            tmp_path,
            str(launcher),
            arrival_process=ArrivalProcess.POISSON,
        )
    )
    poisson_schedule = json.loads(
        (poisson.run_dir / "arrival_schedule.json").read_text(encoding="utf-8")
    )
    poisson_requests = [
        json.loads(line)
        for line in (poisson.run_dir / "requests.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert poisson_schedule["algorithm"] == "poisson-exponential-v1"
    assert [record["scheduled_offset_ns"] for record in poisson_requests] == (
        poisson_schedule["windows"][0]["scheduled_offsets_ns"]
    )
    assert poisson.completed_requests == len(poisson_requests) == 7

    empty = run_open_loop_benchmark(
        _spec(
            tmp_path,
            str(launcher),
            arrival_process=ArrivalProcess.POISSON,
            arrival_rate=1e-6,
            duration_seconds=1e-6,
        )
    )
    assert empty.completed_requests == 0
    assert empty.failed_requests == 0
    assert empty.summary[0]["slo_attainment_rate"] is None
    assert (empty.run_dir / "requests.jsonl").read_text(encoding="utf-8") == ""
