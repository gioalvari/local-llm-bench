# Qwen2.5 0.5B Q4_K_M versus Q8_0

This experiment compares two GGUF quantizations of the same source checkpoint
on an Apple M4 Pro with 48 GiB unified memory.

## Controls

| Control | Value |
| --- | --- |
| Source checkpoint revision | `41ba88dbac95fed2528c92514c131d73eb5a174b` |
| Benchmark protocol SHA-256 | `7df88b09a68b9f3438a1d20a37f77f3852bf84ab9e0dcc8bf2c0813eb79f3f7b` |
| Quality dataset SHA-256 | `ca0282158a697eaec43e992de3e22543dd3c191b87c72e37b0fbaa67b5a7a234` |
| llama.cpp | build 9860, commit `fdb1db877` |
| Microbenchmark repetitions | 3 per cell |
| Serving requests | 5 per variant |
| Quality questions | 12 per prompt arm and variant |

Every canonical performance row uses a 512-token prompt, 128 generated tokens,
batch size 512, microbatch size 128, full Metal offload, and Flash Attention.
Quality runs use the same chat template, prompts, dataset, and deterministic
decoding settings.

## Results

| Metric | Q4_K_M | Q8_0 | Q8_0 change |
| --- | ---: | ---: | ---: |
| Model tensor size (`llama.cpp`) | 373.7 MiB | 500.8 MiB | +34.0% |
| Peak serving RSS | 510.2 MiB | 641.1 MiB | +25.6% |
| Prompt processing | 5,556.28 token/s | 5,885.39 token/s | +5.9% |
| Token generation | 284.49 token/s | 254.31 token/s | -10.6% |
| Median TTFT | 12.43 ms | 12.72 ms | +2.3% |
| Median end-to-end latency | 246.89 ms | 265.30 ms | +7.5% |
| Zero-shot answer accuracy | 91.7% | 83.3% | -8.3 pp |
| Engineered answer accuracy | 50.0% | 66.7% | +16.7 pp |
| Engineered schema validity | 41.7% | 58.3% | +16.7 pp |
| Zero-shot quality-adjusted answers/s | 7.486 | 6.879 | -8.1% |
| Zero-shot quality/GiB RSS | 1.840 | 1.331 | -27.6% |

## Interpretation

Q4_K_M is the stronger default for this hardware and smoke workload. It uses
less memory, generates faster, has lower end-to-end latency, and achieves higher
zero-shot quality per second and per GiB. Q8_0 improves prompt processing and
performs better under the engineered prompt, particularly for strict JSON
compliance, but those gains do not offset its memory and decode costs for the
zero-shot arm.

The quality set contains only 12 public smoke questions. An 8.3 percentage-point
difference is exactly one item and is not evidence of a general quality ranking.
The result demonstrates the harness and an observed trade-off; a sealed, larger
document-level test set is required for a quantization quality claim.
