# Qwen2.5 0.5B serving-factor effects

This analysis decomposes the existing Q4_K_M and Q8_0 `llama-bench` factorial
runs on an Apple M4 Pro with 48 GiB unified memory.

## Method

Each factor is evaluated with matched cells that hold workload, prompt and
generation lengths, microbatch size, thread count, and every other tested factor
constant. Prompt processing and token generation are analyzed separately.

The reported effect is the geometric mean of eight treatment/reference speed
ratios. Fixed-seed percentile bootstrap intervals resample those eight matched
cells. These intervals describe sensitivity across the measured matrix; the
cells are fixed and correlated, so the intervals are not population confidence
intervals.

| Provenance | Q4_K_M | Q8_0 |
| --- | --- | --- |
| Model SHA-256 | `6eb923e7d26e9cea28811e1a8e852009b21242fb157b26149d3b188f3a8c8653` | `25130a98aa782284a7dabea0c23245b2fd371ed47244e79d78b8ec23245fdf96` |
| Measurements SHA-256 | `4cd3762281f3e645142b991e9766427c329de10edcf7a627a9df0f3d25331f6c` | `c3c4af1958d7b434075f7a54012d27701410d21eb167fc3373d7e5c94cfd593c` |
| Matched pairs per effect | 8 | 8 |

Zero model layers offloaded is the reference for the offload factor. It is not
labeled CPU-only because non-layer backend operations may still use Metal.

## Paired Effects

| Factor | Phase | Q4_K_M ratio | Q4 descriptive interval | Q8_0 ratio | Q8 descriptive interval |
| --- | --- | ---: | ---: | ---: | ---: |
| All vs zero model layers offloaded | Prompt | 8.621x | 7.847-9.468x | 4.738x | 4.057-5.476x |
| All vs zero model layers offloaded | Generation | 1.036x | 1.011-1.066x | 1.613x | 1.375-1.974x |
| Flash Attention on vs off | Prompt | 0.946x | 0.875-1.018x | 1.007x | 0.878-1.137x |
| Flash Attention on vs off | Generation | 1.099x | 1.067-1.140x | 1.121x | 1.032-1.235x |
| Batch 512 vs 128 | Prompt | 1.047x | 0.982-1.119x | 1.150x | 0.988-1.351x |
| Batch 512 vs 128 | Generation | 0.993x | 0.954-1.024x | 0.920x | 0.784-1.055x |

## Interpretation

Full model-layer offload is the dominant prompt-processing optimization. It
increases Q4_K_M prompt throughput by a geometric mean of 8.62 times and Q8_0
throughput by 4.74 times. Its generation effect depends strongly on
quantization: Q4_K_M gains 3.6%, while Q8_0 gains 61.3% relative to zero-layer
offload.

Flash Attention consistently benefits generation in this matrix: 9.9% for
Q4_K_M and 12.1% for Q8_0. Its prompt-processing effect is not consistent across
matched cells because both descriptive intervals include the neutral ratio of
1.0.

Increasing batch size from 128 to 512 has workload-dependent effects. Prompt
throughput rises by 4.7% for Q4_K_M and 15.0% for Q8_0 on geometric average, but
both descriptive intervals include 1.0. Generation is nearly unchanged for
Q4_K_M and 8.0% lower for Q8_0 on average, again with intervals spanning 1.0.
Batch 512 should therefore not be treated as an unconditional default for
single-stream generation.

## Process-RSS Pareto Frontier

All non-dominated configurations use full model-layer offload and Flash
Attention. For Q4_K_M, the short-workload frontier retains batch 128 and 512,
while the standard workload retains batch 512. For Q8_0, both batch sizes remain
on the frontier for both workloads because small speed and process-RSS tradeoffs
prevent strict dominance.

This frontier uses process-tree RSS, which does not fully account for Metal
allocations on unified-memory hardware. It is useful for eliminating clearly
dominated measured configurations, not for claiming complete VRAM efficiency.
