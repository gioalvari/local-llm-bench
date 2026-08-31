import json
import sys
from pathlib import Path

import pytest

from localllm_bench.config import ExperimentSpec
from localllm_bench.open_loop import (
    LoadPrompt,
    load_prompts,
    prompt_dataset_sha256,
    request_count,
    run_open_loop_benchmark,
    summarize_rate,
)


def _spec(tmp_path: Path, executable: str) -> ExperimentSpec:
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
                "arrival_rates_rps": [50],
                "duration_seconds": 0.06,
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
    assert result.summary[0]["max_client_in_flight"] >= 2
    manifest = json.loads(
        (result.run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["warmup_completed"] == 1
    assert manifest["prompt_count"] == 2
    assert manifest["context_tokens_per_slot"] == 64
    assert manifest["effective_server"]["context_size"] == 128
