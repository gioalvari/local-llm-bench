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

Randomized-arrival serving and energy sampling remain outside this first
vertical slice. Serving latency comes from `llama-server` streaming events and
is never inferred from `llama-bench` timings. Apple-native MLX QLoRA is included
as an optional training extra.

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

The intermediate pinned Q5_K_M experiment is available at
`configs/experiments/qwen-0.5b-q5-smoke.yaml`.

```bash
curl --fail --location \
  --output models/Qwen2.5-0.5B-Instruct-Q5_K_M.gguf \
  https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF/resolve/41ba88dbac95fed2528c92514c131d73eb5a174b/Qwen2.5-0.5B-Instruct-Q5_K_M.gguf
uv run llmb run configs/experiments/qwen-0.5b-q5-smoke.yaml
```

```bash
curl --fail --location \
  --output models/Qwen2.5-0.5B-Instruct-Q8_0.gguf \
  https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF/resolve/41ba88dbac95fed2528c92514c131d73eb5a174b/Qwen2.5-0.5B-Instruct-Q8_0.gguf
uv run llmb run configs/experiments/qwen-0.5b-q8-smoke.yaml
```

The first controlled quantization comparison is documented in
[`results/qwen-0.5b-q4-vs-q8-m4-pro.md`](results/qwen-0.5b-q4-vs-q8-m4-pro.md).

The expanded three-level quantization comparison is documented in
[`results/qwen-0.5b-q4-q5-q8-m4-pro.md`](results/qwen-0.5b-q4-q5-q8-m4-pro.md).

The first controlled concurrency study is documented in
[`results/qwen-0.5b-q4-concurrency-m4-pro.md`](results/qwen-0.5b-q4-concurrency-m4-pro.md).

The first fixed-rate saturation study is documented in
[`results/qwen-0.5b-q4-open-loop-m4-pro.md`](results/qwen-0.5b-q4-open-loop-m4-pro.md).

The repeated 60-second seeded-Poisson study with run-level confidence intervals
is documented in
[`results/qwen-0.5b-q4-poisson-open-loop-m4-pro.md`](results/qwen-0.5b-q4-poisson-open-loop-m4-pro.md).

The follow-up low-rate study that brackets the 500 ms SLO boundary is documented
in
[`results/qwen-0.5b-q4-poisson-low-rate-m4-pro.md`](results/qwen-0.5b-q4-poisson-low-rate-m4-pro.md).

The controlled context-window and prompt-length study is documented in
[`results/qwen-0.5b-q4-context-m4-pro.md`](results/qwen-0.5b-q4-context-m4-pro.md).

The paired offload, Flash Attention, and batch-size analysis is documented in
[`results/qwen-0.5b-factor-effects-m4-pro.md`](results/qwen-0.5b-factor-effects-m4-pro.md).

The first Apple-native QLoRA smoke run, including its negative held-out result,
is documented in
[`results/qwen-0.5b-mlx-qlora-m4-pro.md`](results/qwen-0.5b-mlx-qlora-m4-pro.md).

Measure end-to-end streaming latency with a fresh local server:

```bash
uv run llmb serve configs/experiments/qwen-0.5b-smoke.yaml
```

Run closed-loop synchronized load at concurrency 1, 2, and 4:

```bash
uv run llmb load configs/experiments/qwen-0.5b-smoke.yaml
uv run llmb report runs/<load-run-id>
```

The load runner performs an excluded warm-up, keeps context capacity per slot
constant, and records aggregate output throughput, request throughput, median
and P95 TTFT/end-to-end latency, error rate, and process-tree memory.

Run deterministic fixed-rate open-loop traffic with a rotating prompt corpus:

```bash
uv run llmb open-loop configs/experiments/qwen-0.5b-smoke.yaml
uv run llmb report runs/<open-loop-run-id>
```

The open-loop runner records offered and achieved request rates, output token
throughput, latency-SLO goodput, median/P95 latency, scheduler delay, maximum
client in-flight requests, errors, and memory. Requests are emitted at fixed
intervals independently of previous completion, then fully drained before the
next rate.

Run a longer open-loop study with seeded Poisson inter-arrival times and eight
independent fresh-server repetitions:

```bash
uv run llmb open-loop configs/experiments/qwen-0.5b-q4-poisson-open-loop.yaml
```

Open `runs/<study-id>/analysis/analysis.html` for the aggregate study. To render
one child run separately, use
`uv run llmb report runs/<study-id>/repetitions/<open-loop-run-id>`.

Schema v2 Poisson runs derive an independent random stream for each offered
rate. Repetitions use consecutive arrival seeds and start a new `llama-server`
sequentially, avoiding host contention and server-state leakage. Rate order uses
balanced cyclic rotations, so every rate appears equally often in each execution
position instead of being confounded with server age. Every complete
precomputed schedule is persisted in `arrival_schedule.json` with its digest in
the child manifest, so matched model variants can replay the same traffic. The
configured rate remains the Poisson intensity; reports separately show the
realized request rate inside the finite arrival window. Open-loop manifests
carrying the expanded artifact contract declare `artifact_schema_version: "2"`;
reports continue to accept earlier manifests.

At least five independent runs are required. The study writes
`analysis/analysis.json` and a self-contained HTML report with
deterministic 95% percentile intervals for arithmetic means across runs. The
bootstrap resamples complete fresh-server runs, not individual requests. A
metric missing from any run, such as latency in an empty Poisson window, remains
undefined rather than silently dropping that run.

Reanalyze five or more compatible completed runs without starting a server:

```bash
uv run llmb analyze-open-loop \
  --run-dir runs/<open-loop-run-a> \
  --run-dir runs/<open-loop-run-b> \
  --run-dir runs/<open-loop-run-c> \
  --run-dir runs/<open-loop-run-d> \
  --run-dir runs/<open-loop-run-e> \
  --output-dir runs/<open-loop-analysis>
```

The analyzer rejects modified schedule digests, incomplete or reordered rate
grids, duplicate Poisson seeds, and differences in model, protocol, hardware,
`llama-server`, or prompt corpus.

Measure exact prompt lengths across context windows:

```bash
uv run llmb context configs/experiments/qwen-0.5b-smoke.yaml
uv run llmb report runs/<context-run-id>
```

The context runner calibrates prompts through the model's `/tokenize` endpoint,
sends exact token-ID arrays, validates backend prompt counts, and starts a fresh
one-slot server for every case. It records load time, prompt evaluation,
TTFT/end-to-end latency, decode rate, and memory. The configured sweep separates
window-size effects at a fixed prompt length from prompt-length effects at a
fixed context window.

Analyze the paired effects in a completed two-level microbenchmark matrix:

```bash
uv run llmb analyze-factors runs/<microbenchmark-run-id> \
  --output-dir runs/<factor-analysis-id>
```

The analysis pairs cells while holding every other setting fixed, reports
phase-specific geometric-mean speed ratios with deterministic bootstrap
intervals, and extracts a per-workload speed/process-RSS Pareto frontier. Zero
GPU layers means no model layers are offloaded; it is not labeled CPU-only
because backend operations may still use the accelerator.

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

The comparator also writes `quality-pareto.json` with the non-dominated
zero-shot variants that jointly maximize quality-adjusted answers per second and
answer accuracy per GiB of peak serving process RSS. Exact ties remain on the
frontier.

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
├── arrival_schedule.json
├── summary.json
├── resource_samples.jsonl
├── failures.jsonl
├── logs/
└── report.html
```

Repeated open-loop studies contain child runs and their aggregate analysis:

```text
runs/<study-id>/
├── repetitions/<child-run-id>/...
└── analysis/
    ├── analysis.json
    └── analysis.html
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

Prepare the document-disjoint MLX chat splits and run the optional Apple-native
4-bit QLoRA smoke experiment:

```bash
make install-mlx
uv run hf download mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --revision a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3 \
  --local-dir models/mlx-qwen2.5-0.5b-instruct-4bit
uv run llmb prepare-training-data \
  datasets/training/energy-market-qa.jsonl \
  --output-dir runs/prepared-training-data
uv run llmb train configs/experiments/qwen-0.5b-smoke.yaml \
  --model-path models/mlx-qwen2.5-0.5b-instruct-4bit
uv run llmb evaluate-mlx configs/experiments/qwen-0.5b-smoke.yaml \
  --model-path models/mlx-qwen2.5-0.5b-instruct-4bit \
  --output-dir runs/mlx-base-evaluation
uv run llmb evaluate-mlx configs/experiments/qwen-0.5b-smoke.yaml \
  --model-path models/mlx-qwen2.5-0.5b-instruct-4bit \
  --adapter-path runs/<training-run>/adapters \
  --output-dir runs/mlx-adapted-evaluation
```

Training data preparation rejects duplicate questions, source documents shared
between splits, and exact normalized answers shared between training and either
evaluation split. Adapter weights and prepared data remain under ignored
`runs/`.

MLX evaluation uses the model's chat template, greedy decoding, and the untouched
test split. It records normalized exact match, token F1, latency, model and
adapter digests, and individual responses under ignored run directories.

Compare the frozen evaluation artifacts:

```bash
uv run llmb compare-mlx \
  --base-dir runs/<base-evaluation> \
  --adapted-dir runs/<adapted-evaluation> \
  --output-dir runs/<mlx-comparison>
```

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
