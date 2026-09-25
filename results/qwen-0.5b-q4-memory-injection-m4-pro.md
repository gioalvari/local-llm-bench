# Memory injection position: system prompt vs. last user turn

This experiment puts the agent-memory-layer proxy in front of RadixForge and
llama-server and measures how much the position of the injected memory block
costs in prefill, on an Apple M4 Pro with 48 GiB unified memory.

Reproduce with (needs `scripts/stack.sh` from
[agent-memory-layer](https://github.com/gioalvari/agent-memory-layer)):

```bash
uv run llmb prefix-sharing configs/experiments/memory-injection-qwen-0.5b.yaml
```

## Controls

| Control | Value |
| --- | --- |
| LLM | Qwen2.5-0.5B-Instruct Q4_K_M (`6eb923e7…8c8653`) |
| Embedding model | nomic-embed-text (Ollama GGUF blob `970aa74c…`) |
| Proxy | agent-memory-layer, top_k 5, fresh SQLite database per repetition |
| Workload | as `prefix-sharing-qwen-0.5b`: 8 agents, ~1,500-token shared system prefix, 4 turns, 32 output tokens |
| Agent identity | `X-Agent-Id: agent-<i>` (per-agent memory namespaces) |
| Repetitions | 3, fresh servers and empty memory per repetition |

Targets:

- `radixforge-no-memory`: RadixForge without the proxy (reference).
- `*-memory-system`: memories appended to the end of the system prompt
  (`--inject-mode system`, the default).
- `*-memory-suffix`: memories prepended to the last user message
  (`--inject-mode suffix`), so the system prompt and the whole history stay
  byte-identical between turns.

All 1,440 requests succeeded, with no embedding warnings. Memories were
injected into every request that had earlier turns to retrieve from (93 of
96 at concurrency 1).

## TTFT (median, continuation turns)

| Concurrency | RadixForge, no memory | RadixForge, system | RadixForge, suffix | llama-server, system | llama-server, suffix |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 21.6 ms | 65.5 ms | **59.6 ms** | 91.4 ms | 91.3 ms |
| 8 | 110.2 ms | 317.4 ms | **227.5 ms** | 399.1 ms | 409.5 ms |

Fan-out turns (turn 0) do not change: no memories exist yet, so both modes are
identical.

## Tokens prefilled per request (median, RadixForge, concurrency 8)

| Turn | No memory | System | Suffix |
| ---: | ---: | ---: | ---: |
| 0 | 30 | 30 | 30 |
| 1 | 55 | 107 | 82 |
| 2 | 56 | 200 | 156 |
| 3 | 54 | 296 | 231 |

## Findings

- **The shared prefix is not lost in either mode.** The proxy appends memories
  to the *end* of the system prompt, so the ~1,500 shared tokens before it stay
  cached; the original concern (injection defeats prefix caching) holds only
  for the part after the insertion point.
- **Suffix injection saves the conversation history.** In system mode the
  history after the system prompt is recomputed every turn; in suffix mode only
  the memory block and the new user message are. At concurrency 8 this cuts
  prefilled tokens by 22–25% on turns 2–3 and median continuation TTFT by 28%
  (317 → 228 ms) on RadixForge.
- **The memory block itself is the main cost.** It changes on every request
  (new retrievals, growing up to top_k), so it is always recomputed: turn 3
  prefills 231 tokens instead of 54 without memory. The proxy roughly triples
  continuation TTFT versus no memory in both modes.
- **llama-server does not benefit.** Its per-slot cache already reuses the
  common prefix of each agent's slot, and at concurrency 8 its TTFT is
  dominated by queueing; the two modes are within noise.

## Implications for agent-memory-layer

- Use `--inject-mode suffix` in front of a prefix-caching backend.
- Next gains come from making the injected block cacheable: stable ordering,
  reusing the previous turn's block when retrievals do not change, or placing
  long-lived memories in a stable block and only new ones near the question.

## Limitations

- One small model and short histories (4 turns × 32 tokens); longer histories
  should widen the suffix advantage, which is not measured here.
- Placing memories next to the question may change answer quality; quality is
  not evaluated in this experiment.
