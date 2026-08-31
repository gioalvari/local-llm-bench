import json
import sys
from pathlib import Path

import pytest

from localllm_bench.config import ContextCaseSpec, ExperimentSpec
from localllm_bench.context_sweep import (
    calibrate_prompt_tokens,
    run_context_sweep,
    summarize_context_case,
)


def _spec(tmp_path: Path, executable: str) -> ExperimentSpec:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("energy market context", encoding="utf-8")
    return ExperimentSpec.model_validate(
        {
            "experiment_id": "context-test",
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
            "context_sweep": {
                "corpus": str(corpus),
                "repetitions": 2,
                "warmup_requests": 1,
                "output_tokens": 2,
                "cases": [
                    {
                        "name": "window-8-prompt-4",
                        "series": ["window-size"],
                        "context_size": 8,
                        "prompt_tokens": 4,
                    },
                    {
                        "name": "window-16-prompt-4",
                        "series": ["window-size", "prompt-length"],
                        "context_size": 16,
                        "prompt_tokens": 4,
                    },
                    {
                        "name": "window-16-prompt-8",
                        "series": ["prompt-length"],
                        "context_size": 16,
                        "prompt_tokens": 8,
                    },
                ],
            },
        }
    )


def test_calibrate_prompt_tokens_cycles_exactly() -> None:
    assert calibrate_prompt_tokens([1, 2, 3], 5) == [1, 2, 3, 1, 2]
    with pytest.raises(ValueError, match="empty"):
        calibrate_prompt_tokens([], 2)
    with pytest.raises(ValueError, match="positive"):
        calibrate_prompt_tokens([1], 0)


def test_summarize_context_case() -> None:
    case = ContextCaseSpec(
        name="prompt-4", series=["window-size"], context_size=8, prompt_tokens=4
    )
    records = [
        {
            "ttft_ns": 10_000_000,
            "e2e_latency_ns": 20_000_000,
            "backend_timings": {
                "prompt_ms": 5.0,
                "prompt_per_second": 800.0,
                "predicted_per_second": 200.0,
            },
        },
        {"error": "failed"},
    ]
    summary = summarize_context_case(
        case,
        records,
        100_000_000,
        [{"process_tree_rss_bytes": 2048, "monotonic_offset_ns": 0}],
    )
    assert summary["error_rate"] == 0.5
    assert summary["model_load_time_ms"] == 100.0
    assert summary["median_prompt_tokens_per_second"] == 800.0


def test_run_context_sweep_with_fake_server(tmp_path: Path) -> None:
    script = tmp_path / "fake_server.py"
    script.write_text(
        """import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

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
        payload = json.loads(self.rfile.read(length))
        self.send_response(200)
        if self.path == '/tokenize':
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{\"tokens\":[1,2,3]}')
            return
        prompt_n = len(payload['prompt'])
        self.send_header('Content-Type', 'text/event-stream')
        self.end_headers()
        self.wfile.write(b'data: {\"content\":\"A\",\"stop\":false}\\n\\n')
        final = {'content': 'B', 'stop': True, 'tokens_predicted': 2,
                 'timings': {'prompt_n': prompt_n, 'prompt_ms': 2.0,
                             'prompt_per_second': prompt_n / 0.002,
                             'predicted_n': 2, 'predicted_per_second': 100.0}}
        self.wfile.write(('data: ' + json.dumps(final) + '\\n\\n').encode())

HTTPServer(('127.0.0.1', args.port), Handler).serve_forever()
""",
        encoding="utf-8",
    )
    launcher = tmp_path / "fake-server"
    launcher.write_text(
        f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8"
    )
    launcher.chmod(0o755)
    result = run_context_sweep(_spec(tmp_path, str(launcher)))
    assert result.completed_requests == 6
    assert result.failed_requests == 0
    assert [item["prompt_tokens"] for item in result.summary] == [4, 4, 8]
    rows = [
        json.loads(line)
        for line in (result.run_dir / "requests.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["backend_timings"]["prompt_n"] for row in rows] == [
        4,
        4,
        4,
        4,
        8,
        8,
    ]
    manifest = json.loads(
        (result.run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert len(manifest["corpus_sha256"]) == 64
