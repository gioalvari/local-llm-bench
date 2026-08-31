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
