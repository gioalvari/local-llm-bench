# Qwen2.5 0.5B Q4_K_M open-loop saturation

This experiment measures deterministic fixed-rate open-loop traffic on an Apple
M4 Pro with 48 GiB unified memory.

## Controls

| Control | Value |
| --- | --- |
| Model | Qwen2.5-0.5B-Instruct Q4_K_M |
| Model SHA-256 | `6eb923e7d26e9cea28811e1a8e852009b21242fb157b26149d3b188f3a8c8653` |
| Prompt corpus SHA-256 | `948faee8f9503b45e427bc49c5039150837bd8043ed6eeca483774dbfcd447b5` |
| llama.cpp | build 9860, commit `fdb1db877` |
| Server slots | 4 |
| Context per slot | 2,048 tokens |
| Output per request | 64 tokens |
| Prompt corpus | 8 prompts, deterministic round-robin order |
| Arrival process | Fixed spacing at each offered rate |
| Measurement window | 3 seconds of arrivals, followed by full drain |
| Latency SLO | End-to-end latency at or below 500 ms |
| Prompt caching | Disabled (`cache_n` was zero) |
| Warm-up | 1 excluded request |

## Results

| Offered req/s | Requests | Achieved req/s | Output token/s | Goodput req/s | SLO attainment | Median TTFT | P95 TTFT | Median E2E | P95 E2E | Max in-flight |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 6 | 2.00 | 127.83 | 2.00 | 100.0% | 13.53 ms | 17.80 ms | 253.45 ms | 259.58 ms | 1 |
| 4 | 12 | 4.00 | 255.97 | 4.00 | 100.0% | 14.37 ms | 20.12 ms | 250.32 ms | 276.55 ms | 2 |
| 8 | 24 | 7.42 | 474.69 | 4.02 | 54.2% | 25.37 ms | 37.09 ms | 499.58 ms | 512.56 ms | 5 |
| 12 | 36 | 7.82 | 500.61 | 0.22 | 2.8% | 679.04 ms | 1,310.74 ms | 1,149.83 ms | 1,734.40 ms | 15 |

All 78 measured requests completed without transport errors. Peak process-tree
RSS was 597.4 MiB. P95 client scheduling delay was at most 13.93 ms in the
per-rate summaries, well below the 83.33 ms shortest arrival interval.

## Interpretation

The tested configuration remains below the 500 ms end-to-end SLO through 4
offered requests/s. At 8 requests/s, aggregate output throughput rises to 474.69
token/s, but only 54.2% of requests meet the SLO and maximum client in-flight
work exceeds the four server slots. At 12 requests/s, achieved throughput
plateaus near 7.82 requests/s while queueing dominates TTFT and latency.

For this prompt and output distribution, 4 requests/s is the highest tested
rate with 100% SLO attainment. The true boundary lies somewhere between 4 and 8
requests/s and requires a finer rate sweep. Maximum raw throughput is not the
same as usable capacity: the 12 requests/s level produces the highest output
token rate but almost no SLO-compliant goodput.

These are descriptive results from one short run. Production capacity planning
requires longer windows, randomized inter-arrival times, repeated independent
runs, confidence intervals, and representative prompt/output distributions.
