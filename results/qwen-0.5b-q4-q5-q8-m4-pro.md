# Qwen2.5 0.5B Q4_K_M versus Q5_K_M versus Q8_0

This experiment compares three GGUF quantizations of the same Qwen2.5 0.5B
Instruct checkpoint on an Apple M4 Pro with 48 GiB unified memory.

## Controls

| Control | Value |
| --- | --- |
| Source checkpoint revision | `41ba88dbac95fed2528c92514c131d73eb5a174b` |
| Benchmark protocol SHA-256 | `7df88b09a68b9f3438a1d20a37f77f3852bf84ab9e0dcc8bf2c0813eb79f3f7b` |
| Quality dataset SHA-256 | `ca0282158a697eaec43e992de3e22543dd3c191b87c72e37b0fbaa67b5a7a234` |
| llama.cpp | build 9860, commit `fdb1db877` |
| Microbenchmark repetitions | 3 per cell |
| Serving requests | 5 per quantization |
| Quality questions | 12 per prompt arm and quantization |

The comparator verified identical hardware fingerprints, `llama-bench` and
`llama-server` binaries, source checkpoint, dataset, and benchmark protocol.
The canonical performance row uses a 512-token prompt, 128 generated tokens,
batch 512, microbatch 128, full Metal offload, and Flash Attention.

## Performance And Memory

| Metric | Q4_K_M | Q5_K_M | Q8_0 |
| --- | ---: | ---: | ---: |
| Model tensor size | 373.7 MiB | 395.0 MiB | 500.8 MiB |
| Peak serving RSS | 510.2 MiB | 533.2 MiB | 641.1 MiB |
| Prompt processing | 5,556.28 token/s | 5,437.04 token/s | 5,885.39 token/s |
| Token generation | 284.49 token/s | 253.12 token/s | 254.31 token/s |
| Median TTFT | 12.43 ms | 14.17 ms | 12.72 ms |
| Median end-to-end latency | 246.89 ms | 285.57 ms | 265.30 ms |

Relative to Q4_K_M, Q5_K_M uses 4.5% more serving RSS, processes prompts 2.1%
slower, generates 11.0% slower, and has 15.7% higher median end-to-end latency.
Q8_0 processes prompts fastest, but its generation rate is effectively tied
with Q5_K_M and it uses 20.2% more serving RSS than Q5_K_M.

## Objective Quality

| Metric | Q4_K_M | Q5_K_M | Q8_0 |
| --- | ---: | ---: | ---: |
| Zero-shot answer accuracy | 91.7% | 83.3% | 83.3% |
| Zero-shot exact match | 66.7% | 66.7% | 66.7% |
| Zero-shot token F1 | 0.833 | 0.833 | 0.833 |
| Zero-shot quality-adjusted answers/s | 7.486 | 6.953 | 6.879 |
| Zero-shot quality/GiB RSS | 1.840 | 1.600 | 1.331 |
| Engineered answer accuracy | 50.0% | 66.7% | 66.7% |
| Engineered exact match | 41.7% | 66.7% | 58.3% |
| Engineered token F1 | 0.625 | 0.778 | 0.750 |
| Engineered strict schema validity | 41.7% | 83.3% | 58.3% |

Q4_K_M wins the measured zero-shot efficiency objective: it has the highest
answer accuracy, quality per second, and quality per GiB. Q5_K_M is strongest
under the engineered structured-output prompt: it ties Q8_0 on answer accuracy,
exceeds it by 25 percentage points in strict schema validity, and uses 16.8%
less serving RSS. Q8_0's primary benefit in this comparison is prompt-processing
throughput, not measured generation quality or memory efficiency.

On the joint zero-shot quality-per-second and quality-per-GiB objective, Q4_K_M
is the sole non-dominated Pareto point; it dominates both Q5_K_M and Q8_0 on the
two measured efficiency axes.

## Recommendation

- Use Q4_K_M as the default for interactive, zero-shot local inference.
- Consider Q5_K_M when strict structured-output compliance with the engineered
  prompt is more important than latency or zero-shot efficiency.
- Use Q8_0 only when prefill throughput justifies its larger memory footprint;
  this smoke test found no answer-accuracy advantage over Q5_K_M.

The quality set contains only 12 public smoke questions. One item changes
accuracy by 8.3 percentage points, so the quality differences are descriptive,
not evidence of a general ranking. A larger sealed test set is required before
selecting quantization on quality grounds.
