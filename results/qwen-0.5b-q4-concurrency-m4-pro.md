# Qwen2.5 0.5B Q4_K_M concurrency scaling

This smoke experiment measures closed-loop concurrent serving on an Apple M4
Pro with 48 GiB unified memory.

## Controls

| Control | Value |
| --- | --- |
| Model | Qwen2.5-0.5B-Instruct Q4_K_M |
| Model SHA-256 | `6eb923e7d26e9cea28811e1a8e852009b21242fb157b26149d3b188f3a8c8653` |
| llama.cpp | build 9860, commit `fdb1db877` |
| Server slots | 4 |
| Context per slot | 2,048 tokens |
| Output per request | 64 tokens |
| Prompt caching | disabled (`cache_n` was zero) |
| Warm-up | 1 excluded request |
| Measured waves | 3 per concurrency level |

Requests in each wave start behind a client-side barrier. Concurrency levels run
in increasing order against the same managed server. Model loading and warm-up
are excluded from level durations.

## Results

| Concurrency | Requests | Output token/s | Request/s | Median TTFT | P95 TTFT | Median E2E | P95 E2E | Max launch spread | Peak RSS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3 | 260.06 | 4.06 | 12.36 ms | 12.68 ms | 246.57 ms | 247.41 ms | 0.00 ms | 586.1 MiB |
| 2 | 6 | 416.94 | 6.51 | 18.46 ms | 27.98 ms | 306.15 ms | 311.57 ms | 0.10 ms | 588.2 MiB |
| 4 | 12 | 559.29 | 8.74 | 31.37 ms | 35.49 ms | 453.94 ms | 460.73 ms | 0.26 ms | 598.7 MiB |

There were no failed requests.

## Scaling

Relative to concurrency 1:

| Concurrency | Throughput multiplier | Scaling efficiency | Median TTFT increase | Median E2E increase |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 1.60x | 80.2% | 49.4% | 24.2% |
| 4 | 2.15x | 53.8% | 153.8% | 84.1% |

Concurrency 4 maximizes aggregate throughput in the measured range, but it does
not provide linear scaling. It more than doubles output throughput while median
request latency rises by 84.1%. Concurrency 2 is the more balanced operating
point when latency matters; concurrency 4 is preferable when aggregate token
throughput is the primary objective. Maximum measured client launch spread was
0.26 ms, small relative to request duration.

The P95 values are descriptive nearest-rank statistics over only 3, 6, and 12
requests. They must not be interpreted as production tail-latency estimates. A
longer open-loop test with randomized prompts and repeated independent runs is
required for capacity planning. The displayed precision supports artifact
recalculation and does not imply equivalent experimental certainty.
