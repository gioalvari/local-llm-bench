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
- Streaming load time, TTFT, end-to-end latency, and client decode rate.
- Frozen zero-shot and engineered prompt arms over source-grounded JSONL data.

Concurrent serving, MLX training, and energy sampling remain outside this first
vertical slice. Serving latency comes from `llama-server` streaming events and
is never inferred from `llama-bench` timings.

## Why a tiny-model MVP

Model artifacts are deliberately not committed. The example starts with
Qwen2.5 0.5B Q4_K_M so that the benchmark engine can be validated with limited
disk space. The main study should move to Qwen2.5 3B only after at least 25 GiB
is available for source weights, quantized variants, adapters, and run output.
The example manifest pins both the upstream revision and model SHA-256.

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

Run the experiment after placing the pinned model in the local model directory:

```bash
mkdir -p models
curl --fail --location \
  --output models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf \
  https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF/resolve/41ba88dbac95fed2528c92514c131d73eb5a174b/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf
uv run llmb run configs/experiments/qwen-0.5b-smoke.yaml
```

For the pinned smoke configuration, place the model at
`models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf`. The runner verifies its SHA-256
before loading it.

The first measured Apple M4 Pro smoke run is summarized in
[`results/qwen-0.5b-m4-pro-smoke.md`](results/qwen-0.5b-m4-pro-smoke.md).

An equivalent pinned Q8_0 experiment is available at
`configs/experiments/qwen-0.5b-q8-smoke.yaml`.

```bash
curl --fail --location \
  --output models/Qwen2.5-0.5B-Instruct-Q8_0.gguf \
  https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF/resolve/41ba88dbac95fed2528c92514c131d73eb5a174b/Qwen2.5-0.5B-Instruct-Q8_0.gguf
uv run llmb run configs/experiments/qwen-0.5b-q8-smoke.yaml
```

The first controlled quantization comparison is documented in
[`results/qwen-0.5b-q4-vs-q8-m4-pro.md`](results/qwen-0.5b-q4-vs-q8-m4-pro.md).

Measure end-to-end streaming latency with a fresh local server:

```bash
uv run llmb serve configs/experiments/qwen-0.5b-smoke.yaml
```

Run objective quality evaluation on the included EIA smoke benchmark:

```bash
uv run llmb evaluate configs/experiments/qwen-0.5b-smoke.yaml
uv run llmb report runs/<quality-run-id>
uv run llmb rescore runs/<quality-run-id>
```

`evaluate` records schema validity, answer accuracy, exact match, token F1,
numeric and unit accuracy, TTFT, end-to-end latency, and quality-adjusted
answers per second. The included 12-item dataset validates the pipeline; it is
not a sealed benchmark for fine-tuning claims.

`rescore` recomputes deterministic metrics without running inference again and
refuses to proceed if the dataset SHA-256 differs from the run manifest.

Compare matched run trios after completing both variants:

```bash
uv run llmb compare \
  --micro-run runs/<q4-micro> --serving-run runs/<q4-serving> \
  --quality-run runs/<q4-quality> \
  --micro-run runs/<q8-micro> --serving-run runs/<q8-serving> \
  --quality-run runs/<q8-quality> \
  --output-dir runs/<comparison-id>
```

The comparator rejects different source checkpoints, dataset digests, benchmark
protocols, hardware fingerprints, `llama.cpp` binaries, and duplicate
quantizations.

The comparison's model-size field is the tensor size reported by `llama.cpp`;
peak RSS comes from the managed serving process tree. Neither value should be
interpreted as standalone VRAM on unified-memory hardware.

Schema validity is intentionally strict: the full response must be exactly the
requested JSON object. A single JSON object inside a Markdown fence may still
be parsed for semantic scoring, but it remains a schema failure. This keeps
format compliance separate from answer quality.

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
├── evaluations.jsonl
├── summary.json
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
Remote model configurations pin the exact GGUF filename instead of relying on
the backend's quantization-name discovery.

The planned serving phase will add:

- Concurrent and open-loop request workloads.
- Aggregate throughput and goodput under load.
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
