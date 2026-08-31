import json
import sys
from io import BytesIO
from pathlib import Path

import pytest

from localllm_bench.config import ExperimentSpec
from localllm_bench.server import (
    build_server_command,
    parse_sse_line,
    run_server_benchmark,
)


def _spec(tmp_path: Path, executable: str = "llama-server") -> ExperimentSpec:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    return ExperimentSpec.model_validate(
        {
            "experiment_id": "serve-test",
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
                "repetitions": 2,
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
        }
    )


def test_build_server_command_is_local_and_disables_cache(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    assert spec.server is not None
    command = build_server_command(spec, spec.server, 8123)
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--port") + 1] == "8123"
    assert "--no-cache-prompt" in command
    assert "--no-webui" in command


def test_parse_sse_line() -> None:
    assert parse_sse_line(b"\n") is None
    assert parse_sse_line(b"data: [DONE]\n") is None
    assert parse_sse_line(b'data: {"content":"x"}\n') == {"content": "x"}
    with pytest.raises(ValueError, match="JSON object"):
        parse_sse_line(b"data: []\n")


def test_stream_parser_reads_openai_chat_delta() -> None:
    from localllm_bench.server import _read_stream

    response = BytesIO(
        b'data: {"choices":[{"delta":{"content":"{\\"answer\\""}}]}\n\n'
        b'data: {"choices":[{"delta":{"content":":\\"x\\",\\"unit\\":null}"}}]}\n\n'
        b"data: [DONE]\n\n"
    )
    result = _read_stream(response, 0)
    assert result["response_text"] == '{"answer":"x","unit":null}'


def test_run_server_benchmark_with_fake_server(tmp_path: Path) -> None:
    script = tmp_path / "fake_server.py"
    script.write_text(
        """import argparse
import json
import time
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
        self.rfile.read(length)
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.end_headers()
        self.wfile.write(b'data: {\"content\":\"A\",\"stop\":false}\\n\\n')
        self.wfile.flush()
        time.sleep(0.01)
        final = {'content': 'B', 'stop': True, 'tokens_predicted': 2,
                 'timings': {'predicted_n': 2, 'predicted_per_second': 10.0}}
        self.wfile.write(('data: ' + json.dumps(final) + '\\n\\n').encode())
        self.wfile.flush()

HTTPServer(('127.0.0.1', args.port), Handler).serve_forever()
""",
        encoding="utf-8",
    )
    launcher = tmp_path / "fake-server"
    launcher.write_text(
        f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8"
    )
    launcher.chmod(0o755)
    result = run_server_benchmark(_spec(tmp_path, str(launcher)))
    assert result.completed_requests == 2
    assert result.failed_requests == 0
    rows = [
        json.loads(line)
        for line in (result.run_dir / "requests.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert rows[0]["response_text"] == "AB"
    assert rows[0]["ttft_ns"] > 0
    assert rows[0]["e2e_latency_ns"] >= rows[0]["ttft_ns"]
    assert rows[0]["client_decode_tokens_per_second"] > 0
    manifest = json.loads(
        (result.run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["model_load_time_ns"] > 0
    assert manifest["peak_process_tree_rss_bytes"] > 0


def test_stream_parser_rejects_empty_content() -> None:
    from localllm_bench.server import _read_stream

    with pytest.raises(ValueError, match="no generated content"):
        _read_stream(BytesIO(b'data: {"stop":true}\n\n'), 0)
