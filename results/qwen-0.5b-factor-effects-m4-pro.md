# Qwen2.5 0.5B serving-factor effects

This analysis decomposes the existing Q4_K_M, Q5_K_M, and Q8_0 `llama-bench`
factorial runs on an Apple M4 Pro with 48 GiB unified memory.

## Method

Each factor is evaluated with matched cells that hold workload, prompt and
generation lengths, microbatch size, thread count, and every other tested factor
constant. Prompt processing and token generation are analyzed separately.

The reported effect is the geometric mean of eight treatment/reference speed
ratios. Fixed-seed percentile bootstrap intervals resample those eight matched
cells. These intervals describe sensitivity across the measured matrix; the
cells are fixed and correlated, so the intervals are not population confidence
intervals.

| Provenance | Q4_K_M | Q5_K_M | Q8_0 |
| --- | --- | --- | --- |
| Model SHA-256 | `6eb923e7...8c8653` | `a0a413dc...19ff3b` | `25130a98...5fdf96` |
| Measurements SHA-256 | `4cd37622...31f6c` | `9b152a74...bda38` | `c3c4af19...d593c` |
| Matched pairs per effect | 8 | 8 | 8 |

Zero model layers offloaded is the reference for the offload factor. It is not
labeled CPU-only because non-layer backend operations may still use Metal.

## Paired Effects

| Factor | Phase | Q4_K_M | Q5_K_M | Q8_0 |
| --- | --- | ---: | ---: | ---: |
| All vs zero model layers offloaded | Prompt | 8.621x | 7.148x | 4.738x |
| All vs zero model layers offloaded | Generation | 1.036x | 1.192x | 1.613x |
| Flash Attention on vs off | Prompt | 0.946x | 0.964x | 1.007x |
| Flash Attention on vs off | Generation | 1.099x | 1.094x | 1.121x |
| Batch 512 vs 128 | Prompt | 1.047x | 0.996x | 1.150x |
| Batch 512 vs 128 | Generation | 0.993x | 1.000x | 0.920x |

## Interpretation

Full model-layer offload is the dominant prompt-processing optimization. It
increases Q4_K_M prompt throughput by a geometric mean of 8.62 times, Q5_K_M by
7.15 times, and Q8_0 by 4.74 times. Its generation effect depends strongly on
quantization: Q4_K_M gains 3.6%, Q5_K_M 19.2%, and Q8_0 61.3% relative to
zero-layer offload.

Flash Attention's geometric-mean generation effect is 9.9% for Q4_K_M, 9.4%
for Q5_K_M, and 12.1% for Q8_0. The Q5_K_M descriptive interval spans 1.0,
however, so its gain is not consistent across all matched Q5 cells. The
prompt-processing effect is also workload-dependent for all three variants.

Increasing batch size from 128 to 512 has workload-dependent effects. Prompt
throughput changes by +4.7% for Q4_K_M, -0.4% for Q5_K_M, and +15.0% for Q8_0
on geometric average. Generation changes by -0.7%, approximately 0.0%, and
-8.0%, respectively. Every batch-effect descriptive interval spans 1.0, so
batch 512 should not be treated as an unconditional default for single-stream
generation.

## Process-RSS Pareto Frontier

Most non-dominated configurations use full model-layer offload. Q5_K_M retains
one zero-layer-offload point because its measured generation speed is high enough
to prevent strict three-objective dominance despite much slower prompt
processing and higher process RSS. This illustrates why Pareto membership is
not itself a recommendation: operational priorities still determine which
objective matters.

This frontier uses process-tree RSS, which does not fully account for Metal
allocations on unified-memory hardware. It is useful for eliminating clearly
dominated measured configurations, not for claiming complete VRAM efficiency.
