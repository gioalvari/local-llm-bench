import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from localllm_bench.prefix_sharing import (
    PrefixSharingConfig,
    load_prefix_sharing_config,
    phase_for_request,
    prepare_workload,
    render_prefix_sharing_report,
    run_prefix_sharing,
    summarize_prefix_sharing,
    validate_completion,
)


def _config_value(tmp_path: Path, executable: str) -> dict[str, object]:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("wind solar storage grid", encoding="utf-8")
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        '{"prompt_id":"one","prompt":"Prompt one."}\n'
        '{"prompt_id":"two","prompt":"Prompt two."}\n',
        encoding="utf-8",
    )
    return {
        "name": "prefix-test",
        "seed": 7,
        "output_dir": str(tmp_path / "runs"),
        "model_path": str(model),
        "workload": {
            "agents": 2,
            "shared_prefix_words": 7,
            "turns": 2,
            "max_tokens": 4,
            "temperature": 0.0,
            "concurrency": [1, 2],
            "repetitions": 1,
            "corpus_path": str(corpus),
            "prompts_path": str(prompts),
        },
        "targets": [
            {
                "name": "server",
                "command": [executable, "--port", "{port}", "-m", "{model}"],
                "baseline": True,
            }
        ],
    }


def test_config_requires_one_baseline() -> None:
    value = {
        "name": "invalid",
        "seed": 1,
        "model_path": "model.gguf",
        "workload": {
            "agents": 1,
            "shared_prefix_words": 1,
            "turns": 1,
            "max_tokens": 1,
            "concurrency": [1],
            "repetitions": 1,
            "corpus_path": "corpus.txt",
            "prompts_path": "prompts.jsonl",
        },
        "targets": [{"name": "one", "command": ["server"]}],
    }
    with pytest.raises(ValidationError, match="exactly one"):
        PrefixSharingConfig.model_validate(value)


def test_config_resolves_paths_and_substitutes_command(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """name: resolved
seed: 1
model_path: models/model.gguf
workload:
  agents: 1
  shared_prefix_words: 1
  turns: 1
  max_tokens: 1
  concurrency: [1]
  repetitions: 1
  corpus_path: data/corpus.txt
  prompts_path: data/prompts.jsonl
targets:
  - name: one
    command: [bin/server, -m, '{model}', --port, '{port}']
    baseline: true
""",
        encoding="utf-8",
    )
    config = load_prefix_sharing_config(config_path)
    assert config.model_path == tmp_path / "models/model.gguf"
    assert config.workload.corpus_path == tmp_path / "data/corpus.txt"
    assert config.targets[0].command[0] == str(tmp_path / "bin/server")
    assert config.targets[0].command[2] == str(tmp_path / "models/model.gguf")


def test_workload_is_deterministic_and_has_shared_prefix(tmp_path: Path) -> None:
    config = PrefixSharingConfig.model_validate(_config_value(tmp_path, "server"))
    first = prepare_workload(config)
    second = prepare_workload(config)
    assert first.user_prompts == second.user_prompts
    assert (
        first.systems[0].split("\n\nAgent profile:")[0]
        == first.systems[1].split("\n\nAgent profile:")[0]
    )
    assert first.systems[0] != first.systems[1]


def test_phase_labelling() -> None:
    assert phase_for_request(1, 0, 0) == "cold"
    assert phase_for_request(1, 0, 1) == "fanout"
    assert phase_for_request(2, 0, 0) == "fanout"
    assert phase_for_request(2, 1, 0) == "continuation"


def test_summary_math_and_report_rendering(tmp_path: Path) -> None:
    records = [
        {
            "target": "baseline",
            "concurrency": 1,
            "phase": "cold",
            "repetition": 0,
            "ttft_ns": 20_000_000,
            "e2e_latency_ns": 30_000_000,
            "backend_timings": {"prompt_n": 3, "cache_n": 1},
            "error": None,
        },
        {
            "target": "candidate",
            "concurrency": 1,
            "phase": "cold",
            "repetition": 0,
            "ttft_ns": 10_000_000,
            "e2e_latency_ns": 20_000_000,
            "backend_timings": {"prompt_n": 2, "cache_n": 2},
            "error": None,
        },
        {
            "target": "candidate",
            "concurrency": 1,
            "phase": "cold",
            "repetition": 1,
            "error": "failed",
        },
    ]
    repetitions = [
        {
            "target": "baseline",
            "concurrency": 1,
            "status": "ok",
            "wall_ns": 1_000_000_000,
        },
        {
            "target": "candidate",
            "concurrency": 1,
            "status": "ok",
            "wall_ns": 500_000_000,
        },
    ]
    summary = summarize_prefix_sharing(records, repetitions, "baseline")
    candidate = next(
        item for item in summary["groups"] if item["target"] == "candidate"
    )
    assert candidate["ttft_p50_ms"] == 10.0
    assert candidate["speedup_vs_baseline"] == 2.0
    candidate_wall = next(
        item for item in summary["wall_clock"] if item["target"] == "candidate"
    )
    assert candidate_wall["cached_token_ratio"] == 0.5
    config = PrefixSharingConfig.model_validate(_config_value(tmp_path, "server"))
    report = render_prefix_sharing_report(config, summary)
    assert "# Prefix-sharing multi-agent benchmark" in report
    assert "| candidate | cold | 1/1 | 10.00" in report


def test_run_prefix_sharing_with_fake_streaming_server_and_start_failure(
    tmp_path: Path,
) -> None:
    script = tmp_path / "fake_server.py"
    script.write_text(
        """import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--port', type=int)
args, _ = parser.parse_known_args()

class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(b'{"ok":true}')

    def do_POST(self):
        length = int(self.headers.get('Content-Length', '0'))
        self.rfile.read(length)
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.end_headers()
        self.wfile.write(b'data: {"choices":[{"delta":{"content":"A"}}]}\\n\\n')
        final = {
            'choices': [{'delta': {'content': 'B'}}],
            'timings': {'prompt_n': 3, 'cache_n': 1, 'predicted_n': 2},
        }
        self.wfile.write(('data: ' + json.dumps(final) + '\\n\\n').encode())
        self.wfile.write(b'data: [DONE]\\n\\n')
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
    value = _config_value(tmp_path, str(launcher))
    value["targets"] = [
        {
            "name": "candidate",
            "command": [str(launcher), "--port", "{port}"],
            "baseline": False,
        },
        {
            "name": "baseline",
            "command": [str(launcher), "--port", "{port}"],
            "baseline": True,
        },
        {
            "name": "broken",
            "command": [str(tmp_path / "does-not-exist")],
            "baseline": False,
        },
    ]
    result = run_prefix_sharing(PrefixSharingConfig.model_validate(value))
    assert result.completed_requests == 16
    assert result.failed_requests == 0
    assert (result.run_dir / "manifest.json").exists()
    assert (result.run_dir / "requests.jsonl").exists()
    assert (result.run_dir / "summary.json").exists()
    assert (result.run_dir / "report.md").exists()
    summary = json.loads((result.run_dir / "summary.json").read_text(encoding="utf-8"))
    assert {(row["target"], row["concurrency"]) for row in summary["wall_clock"]} == {
        ("baseline", 1),
        ("baseline", 2),
        ("candidate", 1),
        ("candidate", 2),
        ("broken", 1),
        ("broken", 2),
    }
    repetitions = [
        json.loads(line)
        for line in (result.run_dir / "repetitions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert any(
        row["target"] == "broken" and row["status"] == "failed" for row in repetitions
    )


@pytest.mark.parametrize(
    "text", ["", "   ", "[ERROR: prefill failed]", " [ERROR: resource exhausted]"]
)
def test_validate_completion_rejects_failures(text: str) -> None:
    with pytest.raises(ValueError):
        validate_completion(text)


def test_validate_completion_accepts_normal_text() -> None:
    validate_completion("Prices rose because of [demand].")
