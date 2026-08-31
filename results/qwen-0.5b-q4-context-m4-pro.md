# Qwen2.5 0.5B Q4_K_M context-length effects

This experiment separates configured context-window size from actual prompt
length on an Apple M4 Pro with 48 GiB unified memory.

## Controls

| Control | Value |
| --- | --- |
| Model | Qwen2.5-0.5B-Instruct Q4_K_M |
| Model SHA-256 | `6eb923e7d26e9cea28811e1a8e852009b21242fb157b26149d3b188f3a8c8653` |
| Corpus SHA-256 | `8c9f2404f98a9311c182ba23211db23a841c686d2853166220e9879e34a6814a` |
| llama.cpp | build 9860, commit `fdb1db877` |
| Output per request | 64 tokens |
| Repetitions | 3 measured plus 1 excluded warm-up per case |
| Server lifecycle | Fresh one-slot server for every case |
| Prompt construction | Exact token IDs from the loaded tokenizer |
| Prompt caching | Disabled (`cache_n` was zero) |

All 21 measured requests completed without errors or context truncation. Backend
prompt counts matched every target exactly.

## Configured Window Size

This series holds the actual prompt at 128 tokens.

| Context window | Prompt tokens | Prompt eval | Prompt token/s | Median TTFT | Median E2E | Decode token/s | Peak RSS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 128 | 24.10 ms | 5,312.08 | 24.97 ms | 254.86 ms | 278.67 | 495.3 MiB |
| 1,024 | 128 | 23.42 ms | 5,464.48 | 24.33 ms | 256.45 ms | 275.87 | 502.5 MiB |
| 2,048 | 128 | 24.13 ms | 5,305.48 | 24.96 ms | 254.71 ms | 278.83 | 516.2 MiB |
| 4,096 | 128 | 23.45 ms | 5,457.72 | 24.29 ms | 256.04 ms | 276.35 | 540.4 MiB |

Increasing the configured window from 256 to 4,096 tokens raises peak RSS by
9.1%, while median end-to-end latency changes by only 0.5%. Reserved KV capacity
therefore has a visible memory cost even when the request uses a short prompt,
but it does not materially change latency in this small-model test.

## Actual Prompt Length

This series holds the configured context window at 4,096 tokens.

| Context window | Prompt tokens | Prompt eval | Prompt token/s | Median TTFT | Median E2E | Decode token/s | Peak RSS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 128 | 23.45 ms | 5,457.72 | 24.29 ms | 256.04 ms | 276.35 | 540.4 MiB |
| 4,096 | 512 | 91.94 ms | 5,568.73 | 92.81 ms | 324.47 ms | 276.08 | 540.1 MiB |
| 4,096 | 1,024 | 186.76 ms | 5,482.86 | 187.65 ms | 421.23 ms | 273.46 | 535.6 MiB |
| 4,096 | 2,048 | 393.81 ms | 5,200.48 | 394.87 ms | 631.72 ms | 270.33 | 538.6 MiB |

At a fixed window, increasing the prompt from 128 to 2,048 tokens makes median
TTFT 16.26 times larger and median end-to-end latency 2.47 times larger. Prompt
processing remains above 5,200 token/s but falls 4.7% at the longest prompt;
decode throughput falls 2.2%.

The main latency effect comes from tokens actually processed, not merely from
reserving a larger context window. The memory effect is the opposite: configured
window size changes RSS, while actual prompt occupancy has little additional
impact once the KV cache is allocated.

These are descriptive medians from three requests per case. Longer independent
runs are required for confidence intervals and stable tail-latency estimates.
