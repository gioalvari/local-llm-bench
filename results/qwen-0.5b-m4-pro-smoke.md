# Qwen2.5 0.5B Q4_K_M smoke benchmark

This preliminary run validates the benchmark pipeline. It is not a model or
quantization comparison.

## Environment

| Item | Value |
| --- | --- |
| Host | Apple M4 Pro, 14 CPU cores, 48 GiB unified memory |
| Model | Qwen2.5-0.5B-Instruct Q4_K_M |
| Model parameters | 494,032,768 |
| Model size | 391,859,712 bytes |
| Model SHA-256 | `6eb923e7d26e9cea28811e1a8e852009b21242fb157b26149d3b188f3a8c8653` |
| llama.cpp | build 9860, commit `fdb1db877` |
| Benchmark repetitions | 3 per microbenchmark cell |

## Microbenchmark

The 16-cell matrix varied prompt length, batch size, CPU versus full Metal
offload, and Flash Attention. Each cell emitted separate prompt-processing and
generation measurements, for 32 observations total. There were no failures.

| Metric | Observed range |
| --- | ---: |
| Prompt processing | 523.86-5,573.20 token/s |
| Token generation | 233.42-284.49 token/s |
| Peak process-tree RSS | 758,611,968 bytes |

The best prompt-processing cell used full Metal offload, Flash Attention, a
128-token prompt, and batch size 128. The best generation cell used full Metal
offload, Flash Attention, a 512-token prompt, and batch size 512.

## Streaming serving

Five sequential requests used a fixed 14-token prompt, 64 generated tokens,
greedy decoding, disabled prompt caching, full Metal offload, and Flash
Attention. The model server was started fresh for the run.

| Metric | Result |
| --- | ---: |
| Model load time | 450.35 ms |
| Median TTFT | 12.43 ms |
| Median end-to-end latency | 246.89 ms |
| Median client decode rate | 268.89 token/s |
| Peak process-tree RSS | 534,986,752 bytes |
| Maximum host available-memory delta | 356,745,216 bytes |
| Failed requests | 0 of 5 |

`llama-bench` rates exclude tokenization and sampling. The serving measurements
use client-side monotonic timestamps over streamed events and therefore answer a
different question. Apple Silicon uses unified memory, so no standalone VRAM
number is reported.

## Objective quality smoke test

The model answered 12 source-grounded EIA questions under two frozen prompt
arms. All requests used the model's chat template, greedy decoding, disabled
prompt caching, and the same Q4_K_M artifact as the performance runs.

| Metric | Zero-shot | Engineered |
| --- | ---: | ---: |
| Answer accuracy | 91.7% | 50.0% |
| Exact match | 66.7% | 41.7% |
| Token F1 | 0.833 | 0.625 |
| Scorable response rate | 100.0% | 83.3% |
| Strict schema validity | 0.0% | 41.7% |
| Numeric accuracy | 85.7% | 57.1% |
| Unit accuracy | 100.0% | 28.6% |
| Median TTFT | 34.40 ms | 49.24 ms |
| Median end-to-end latency | 124.94 ms | 135.61 ms |
| Quality-adjusted answers/s | 7.486 | 2.902 |

Strict schema validity requires the entire output to be exactly one JSON
object. A single JSON object inside a Markdown fence is accepted for semantic
scoring but remains a schema failure. This explains why zero-shot can have
non-zero answer accuracy with zero strict schema validity.

The engineered prompt improved output-format compliance but did not improve
semantic accuracy on this small model and dataset. The result is preliminary:
the 12-item public smoke set validates the harness and is not a sealed benchmark
for fine-tuning claims.

Dataset SHA-256:
`ca0282158a697eaec43e992de3e22543dd3c191b87c72e37b0fbaa67b5a7a234`.
