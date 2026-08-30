# LocalLLM Bench

LocalLLM Bench is a reproducible benchmark for finding the best local language
model configuration for a hardware budget. It keeps model quality, inference
speed, and memory consumption in the same experiment rather than treating
quantization as a speed-only decision.

The first release targets Apple Silicon with `llama.cpp`. It provides a small,
storage-conscious vertical slice before expanding to streaming serving,
Transformers/MPS, MLX fine-tuning, and a sealed energy-market QA benchmark.

## Current status

The MVP includes:

- A host and backend capability report with executable fingerprints.
- Validated YAML experiment specifications.
- Deterministic matrix expansion with invalid batch combinations skipped.
- Isolated `llama-bench` subprocesses for every configuration.
- Prompt-processing and token-generation throughput.
- Process-tree RSS and host-memory sampling.
- Immutable JSON/JSONL run artifacts and a self-contained HTML report.
- Judge-independent exact-match, token-F1, and numeric-answer metrics.

TTFT, request latency, concurrent serving, MLX training, and energy sampling are
explicitly outside this first vertical slice. They require `llama-server` or an
MLX training worker and must not be inferred from `llama-bench` timings.

## Why a tiny-model MVP

Model artifacts are deliberately not committed. The example starts with
Qwen2.5 0.5B Q4_K_M so that the benchmark engine can be validated with limited
disk space. The main study should move to Qwen2.5 3B only after at least 25 GiB
is available for source weights, quantized variants, adapters, and run output.

## Requirements

- Python 3.11 or newer
- `uv`
- A recent `llama.cpp` installation containing `llama-bench`
- Approximately 1 GiB free for the tiny-model smoke experiment

On macOS with Homebrew:

```bash
brew install llama.cpp
make install
```

MLX is an optional, larger environment installed only when beginning the
fine-tuning phase:

```bash
make install-mlx
```

## Usage

Inspect the machine before running experiments:

```bash
uv run llmb doctor
```

Validate and display the concrete benchmark matrix:

```bash
uv run llmb plan configs/experiments/qwen-0.5b-smoke.yaml
```

Run the experiment. The first invocation may download the model from Hugging
Face through `llama.cpp`:

```bash
uv run llmb run configs/experiments/qwen-0.5b-smoke.yaml
```

Generate a report from the run directory printed by the command:

```bash
uv run llmb report runs/<run-id>
```

Use `--dry-run` to inspect commands without downloading or loading a model:

```bash
uv run llmb run configs/experiments/qwen-0.5b-smoke.yaml --dry-run
```

## Artifacts

Each execution creates a new directory under `runs/`:

```text
runs/<run-id>/
├── manifest.json
├── measurements.jsonl
├── resource_samples.jsonl
├── failures.jsonl
├── logs/
└── report.html
```

Durations are stored as integer nanoseconds and memory values as integer bytes.
On Apple Silicon, memory is unified; the project therefore reports process RSS
and host available memory rather than presenting a misleading VRAM number.

## Benchmark interpretation

`llama-bench` excludes tokenization and sampling. Its prompt-processing and
generation rates are microbenchmark measurements, not end-to-end user latency.
Comparisons must use the same source checkpoint, prompt lengths, output lengths,
KV-cache types, and effective runtime settings.

The `llama-bench` executable has no version flag. Its SHA-256 is captured by
`doctor`, while its build number and commit are retained from every result row.

The planned serving phase will add:

- Time to first token and inter-token latency from streaming event timestamps.
- End-to-end latency and aggregate throughput.
- Concurrent and open-loop request workloads.
- Cold-load and first-request timing.
- Quality-per-second and quality-per-gigabyte Pareto analysis.

## Fine-tuning track

The intended training experiment uses MLX-LM LoRA on a narrow, public-source
energy-market QA dataset. Its four registered evaluation arms are:

1. Base model with a fixed zero-shot prompt.
2. Base model with a frozen prompt-engineered template.
3. Fine-tuned model with the zero-shot template.
4. Fine-tuned model quantized for deployment.

The held-out set must be split by source document or topic, not random rows.
Acceptance uses exact match, token F1, numeric accuracy with units and fixed
tolerances, abstention F1, and output-schema validity. An LLM judge is not an
acceptance metric.

## Development

```bash
make check
make full
```

Hardware integration tests are kept separate from unit tests. No model weights,
generated datasets, credentials, or machine-specific run artifacts belong in
version control.

## License

MIT. Model weights and datasets retain their own licenses.
