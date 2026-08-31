import json
import sys
from pathlib import Path

import pytest

from localllm_bench.config import ExperimentSpec
from localllm_bench.load import percentile, run_load_benchmark, summarize_level


def _spec(tmp_path: Path, executable: str) -> ExperimentSpec:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    return ExperimentSpec.model_validate(
        {
            "experiment_id": "load-test",
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
            "load": {
                "concurrency_levels": [1, 2],
                "waves_per_level": 2,
                "warmup_requests": 1,
            },
        }
    )


def test_percentile_uses_nearest_rank() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 95) == 4.0
    with pytest.raises(ValueError, match="at least one"):
        percentile([], 95)
    with pytest.raises(ValueError, match="interval"):
        percentile([1.0], 0)


def test_summarize_level_includes_failures_and_memory() -> None:
    records = [
        {
            "ttft_ns": 10_000_000,
            "e2e_latency_ns": 100_000_000,
            "output_tokens": 10,
            "wave_index": 0,
            "wave_started_ns": 1_000_000,
        },
        {
            "ttft_ns": 20_000_000,
            "e2e_latency_ns": 200_000_000,
            "output_tokens": 10,
            "wave_index": 0,
            "wave_started_ns": 1_500_000,
        },
        {"error": "failed"},
    ]
    summary = summarize_level(
        2,
        records,
        1_000_000_000,
        [{"process_tree_rss_bytes": 1024, "monotonic_offset_ns": 0}],
    )
    assert summary["error_rate"] == pytest.approx(1 / 3)
    assert summary["aggregate_output_tokens_per_second"] == 20.0
    assert summary["median_ttft_ms"] == 15.0
    assert summary["max_wave_launch_spread_ms"] == 0.5
    assert summary["peak_process_tree_rss_bytes"] == 1024


def test_run_load_benchmark_with_threaded_fake_server(tmp_path: Path) -> None:
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
        time.sleep(0.03)
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
    result = run_load_benchmark(_spec(tmp_path, str(launcher)))
    assert result.completed_requests == 6
    assert result.failed_requests == 0
    assert [level["concurrency"] for level in result.summary] == [1, 2]
    assert all(level["max_wave_launch_spread_ms"] >= 0 for level in result.summary)
    manifest = json.loads(
        (result.run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["warmup_completed"] == 1
    assert manifest["context_tokens_per_slot"] == 64
    assert manifest["effective_server"]["parallel"] == 2
    assert manifest["effective_server"]["context_size"] == 128
    assert (
        len(
            (result.run_dir / "requests.jsonl").read_text(encoding="utf-8").splitlines()
        )
        == 6
    )
