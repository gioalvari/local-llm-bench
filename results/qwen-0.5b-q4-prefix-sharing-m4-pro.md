# Qwen2.5 0.5B Q4_K_M multi-agent prefix sharing

This experiment measures a multi-agent workload in which every agent shares the
same long system prompt, on an Apple M4 Pro with 48 GiB unified memory. It
compares RadixForge (radix-tree KV sharing across sequences) with llama-server
with and without prompt caching.

Reproduce with:

```bash
uv run llmb prefix-sharing configs/experiments/prefix-sharing-qwen-0.5b.yaml
```

## Controls

| Control | Value |
| --- | --- |
| Model | Qwen2.5-0.5B-Instruct Q4_K_M |
| Model SHA-256 | `6eb923e7d26e9cea28811e1a8e852009b21242fb157b26149d3b188f3a8c8653` |
| llama-server | build 9860, commit `fdb1db877`, `--parallel 8`, `--ctx-size 32768` |
| RadixForge | llama.cpp `f728adab`, unified KV, `--max-seq 32`, `--ctx-size 32768` |
| Agents | 8, identical ~1,500-token system prefix + short per-agent suffix |
| Turns | 4 per agent; turn 0 = fan-out, turns 1–3 = continuation with real history |
| Output per request | 32 tokens, temperature 0 |
| Repetitions | 3, fresh server (empty cache) per repetition |
| Warm-up | 1 excluded unrelated request per server |

Phases: `cold` is the first request of a repetition at concurrency 1 (nothing
cached); `fanout` is the other agents' first turn (shared prefix possibly
cached); `continuation` is later turns (the agent's own history possibly
cached). Completions that are empty or carry an in-band server error count as
failures. All 576 requests succeeded.

## TTFT (median of pooled requests)

| Concurrency | Phase | llama-server no cache | llama-server cache | RadixForge |
| ---: | --- | ---: | ---: | ---: |
| 1 | cold | 245.2 ms | 244.4 ms | 228.9 ms |
| 1 | fanout | 243.9 ms | 27.5 ms | **17.0 ms** |
| 1 | continuation | 275.7 ms | 57.9 ms | **21.2 ms** |
| 8 | fanout | 1,608.1 ms | 1,856.8 ms | **288.3 ms** |
| 8 | continuation | 2,157.0 ms | 111.8 ms | **107.6 ms** |

## Throughput

| Concurrency | Target | Median repetition wall | Request/s | Cached-token ratio |
| ---: | --- | ---: | ---: | ---: |
| 1 | llama-server no cache | 12.37 s | 2.58 | 0.00 |
| 1 | llama-server cache | 5.54 s | 5.71 | 0.90 |
| 1 | RadixForge | 5.39 s | 5.91 | 0.94 |
| 8 | llama-server no cache | 10.44 s | 3.07 | 0.00 |
| 8 | llama-server cache | 4.08 s | 7.75 | 0.75 |
| 8 | RadixForge | 2.98 s | 10.74 | 0.94 |

## Findings

- **Concurrent fan-out is the main difference.** With 8 agents arriving
  together, llama-server spreads them across slots whose caches are per-slot, so
  the shared prefix is recomputed in each slot (`cache_n` = 0) and the prompt
  cache gives no benefit (1.86 s vs 1.61 s). RadixForge prefills the prefix once
  and shares it, reducing median fan-out TTFT to 288 ms (6.4x lower than
  llama-server with cache).
- **Sequential reuse is also faster**: 17 ms vs 27 ms fan-out TTFT and 21 ms vs
  58 ms continuation TTFT at concurrency 1.
- **Continuation at concurrency 8 is on par** (108 ms vs 112 ms): each agent's
  history lives in its own llama-server slot, which is the case llama-server's
  cache already handles well.
- **Decode is not faster.** E2E latency at concurrency 8 is higher for
  RadixForge in continuations (696 ms vs 525 ms median) because generation runs
  as one static batch without continuous batching; overall repetition wall time
  is still 27% lower thanks to fan-out.

## Limitations

- One small model (0.5B) on one machine; larger models increase prefill cost and
  should widen the fan-out gap, but this is not measured here.
- Nearest-rank percentiles over pooled requests; repetition medians are reported
  per cell in the run's `report.md`. No confidence intervals.
- RadixForge uses a fixed ChatML template, which matches Qwen2.5 but changes
  token counts relative to llama-server's GGUF template.
