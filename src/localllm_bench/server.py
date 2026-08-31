"""End-to-end streaming benchmarks for llama.cpp server."""

import json
import socket
import subprocess
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import psutil
from pydantic import BaseModel

from localllm_bench.config import ExperimentSpec, ServerSpec
from localllm_bench.doctor import inspect_capabilities
from localllm_bench.model import model_args, validate_model
from localllm_bench.telemetry import ResourceMonitor


class ServerRunResult(BaseModel):
    """Location and request counts for one serving benchmark."""

    run_id: str
    run_dir: Path
    completed_requests: int
    failed_requests: int


def _available_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def build_server_command(
    experiment: ExperimentSpec, server: ServerSpec, port: int
) -> list[str]:
    """Build a local-only llama-server command."""
    return [
        experiment.llama_server_binary,
        *model_args(experiment.model),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ctx-size",
        str(server.context_size),
        "--batch-size",
        str(server.batch_size),
        "--ubatch-size",
        str(server.ubatch_size),
        "--threads",
        str(server.threads),
        "--n-gpu-layers",
        str(server.gpu_layers),
        "--flash-attn",
        server.flash_attention.value,
        "--parallel",
        str(server.parallel),
        "--no-cache-prompt",
        "--no-webui",
    ]


def parse_sse_line(line: bytes) -> dict[str, Any] | None:
    """Parse one llama.cpp server-sent event line."""
    decoded = line.decode("utf-8").strip()
    if not decoded.startswith("data:"):
        return None
    payload = decoded.removeprefix("data:").strip()
    if not payload or payload == "[DONE]":
        return None
    event = json.loads(payload)
    if not isinstance(event, dict):
        raise ValueError("stream event must be a JSON object")
    return event


def _wait_until_ready(
    base_url: str, process: subprocess.Popen[str], timeout_seconds: int
) -> int:
    started_ns = time.monotonic_ns()
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited with code {process.returncode}")
        try:
            with urlopen(f"{base_url}/health", timeout=1) as response:
                if response.status == 200:
                    return time.monotonic_ns() - started_ns
        except (HTTPError, URLError, TimeoutError):
            pass
        time.sleep(0.05)
    raise TimeoutError(f"llama-server was not ready after {timeout_seconds} seconds")


def _read_stream(response: BinaryIO, request_started_ns: int) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    content_parts: list[str] = []
    event_offsets_ns: list[int] = []
    first_content_ns: int | None = None
    last_content_ns: int | None = None
    final_event: dict[str, Any] = {}
    for line in response:
        event = parse_sse_line(line)
        if event is None:
            continue
        observed_ns = time.monotonic_ns()
        events.append(event)
        event_offsets_ns.append(observed_ns - request_started_ns)
        final_event = event
        content = event.get("content")
        if isinstance(content, str) and content:
            content_parts.append(content)
            first_content_ns = first_content_ns or observed_ns
            last_content_ns = observed_ns
    completed_ns = time.monotonic_ns()
    if first_content_ns is None:
        raise ValueError("completion stream contained no generated content")
    timings = final_event.get("timings", {})
    output_tokens_value = final_event.get("tokens_predicted")
    if not isinstance(output_tokens_value, int) and isinstance(timings, dict):
        output_tokens_value = timings.get("predicted_n")
    output_tokens = (
        output_tokens_value if isinstance(output_tokens_value, int) else None
    )
    decode_duration_ns = (
        last_content_ns - first_content_ns if last_content_ns is not None else 0
    )
    client_decode_rate = (
        (output_tokens - 1) * 1_000_000_000 / decode_duration_ns
        if output_tokens is not None and output_tokens > 1 and decode_duration_ns > 0
        else None
    )
    return {
        "ttft_ns": first_content_ns - request_started_ns,
        "e2e_latency_ns": completed_ns - request_started_ns,
        "decode_duration_ns": decode_duration_ns,
        "client_decode_tokens_per_second": client_decode_rate,
        "output_tokens": output_tokens,
        "response_text": "".join(content_parts),
        "event_offsets_ns": event_offsets_ns,
        "backend_timings": timings if isinstance(timings, dict) else {},
        "event_count": len(events),
    }


def stream_completion(
    base_url: str, server: ServerSpec, *, request_index: int
) -> dict[str, Any]:
    """Send one deterministic streaming request and measure client timings."""
    payload = {
        "prompt": server.prompt,
        "n_predict": server.output_tokens,
        "temperature": 0.0,
        "seed": 42,
        "stream": True,
        "cache_prompt": False,
        "ignore_eos": True,
    }
    request = Request(
        f"{base_url}/completion",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started_ns = time.monotonic_ns()
    with urlopen(request, timeout=server.request_timeout_seconds) as response:
        measurement = _read_stream(response, started_ns)
    return {"request_index": request_index, **measurement}


def _stop_server(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def run_server_benchmark(experiment: ExperimentSpec) -> ServerRunResult:
    """Start llama-server, issue streaming requests, and persist measurements."""
    if experiment.server is None:
        raise ValueError("experiment does not define a server section")
    validate_model(experiment.model)
    server = experiment.server
    port = server.port or _available_port()
    run_id = (
        f"{experiment.experiment_id}-serve-"
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    )
    run_dir = experiment.output_dir / run_id
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True)
    command = build_server_command(experiment, server, port)
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "run_type": "llama-server",
        "experiment": experiment.model_dump(mode="json"),
        "command": command,
        "capabilities": inspect_capabilities(run_dir).model_dump(mode="json"),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    completed = 0
    failed = 0
    started_ns = time.monotonic_ns()
    with (
        (logs_dir / "server.stdout.log").open("w", encoding="utf-8") as stdout,
        (logs_dir / "server.stderr.log").open("w", encoding="utf-8") as stderr,
    ):
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, text=True)
        monitor = ResourceMonitor(
            psutil.Process(process.pid), experiment.sample_interval_ms / 1000
        )
        monitor.start()
        try:
            base_url = f"http://127.0.0.1:{port}"
            load_time_ns = _wait_until_ready(
                base_url, process, server.startup_timeout_seconds
            )
            manifest["model_load_time_ns"] = load_time_ns
            (run_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with (run_dir / "requests.jsonl").open("w", encoding="utf-8") as requests:
                for request_index in range(server.repetitions):
                    try:
                        measurement = stream_completion(
                            base_url, server, request_index=request_index
                        )
                        requests.write(json.dumps(measurement, sort_keys=True) + "\n")
                        requests.flush()
                        completed += 1
                    except (HTTPError, URLError, TimeoutError, ValueError) as error:
                        failed += 1
                        with (run_dir / "failures.jsonl").open(
                            "a", encoding="utf-8"
                        ) as failures:
                            failures.write(
                                json.dumps(
                                    {
                                        "request_index": request_index,
                                        "error": str(error),
                                    },
                                    sort_keys=True,
                                )
                                + "\n"
                            )
                        if experiment.fail_fast:
                            break
        finally:
            _stop_server(process)
            samples = monitor.stop()
    with (run_dir / "resource_samples.jsonl").open("w", encoding="utf-8") as stream:
        for sample in samples:
            stream.write(json.dumps(sample, sort_keys=True) + "\n")
    manifest["process_wall_time_ns"] = time.monotonic_ns() - started_ns
    manifest["peak_process_tree_rss_bytes"] = max(
        (sample["process_tree_rss_bytes"] for sample in samples), default=0
    )
    initial_available = int(manifest["capabilities"]["available_memory_bytes"])
    manifest["max_host_memory_delta_bytes"] = max(
        [
            0,
            *(initial_available - sample["host_available_bytes"] for sample in samples),
        ]
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return ServerRunResult(
        run_id=run_id,
        run_dir=run_dir,
        completed_requests=completed,
        failed_requests=failed,
    )
