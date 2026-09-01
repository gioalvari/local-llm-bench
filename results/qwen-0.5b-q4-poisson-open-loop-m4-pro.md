# Qwen2.5 0.5B Q4_K_M repeated Poisson open-loop study

This experiment measures seeded Poisson open-loop traffic on an Apple M4 Pro
with 48 GiB unified memory. It uses eight independent fresh-server runs and a
balanced cyclic rate order rather than treating requests from one server process
as independent repetitions.

## Controls

| Control | Value |
| --- | --- |
| Model | Qwen2.5-0.5B-Instruct Q4_K_M |
| Model SHA-256 | `6eb923e7d26e9cea28811e1a8e852009b21242fb157b26149d3b188f3a8c8653` |
| Prompt corpus SHA-256 | `948faee8f9503b45e427bc49c5039150837bd8043ed6eeca483774dbfcd447b5` |
| llama.cpp | build 9860, commit `fdb1db877` |
| llama-server SHA-256 | `6e0b939f96d6407ce71dc27b74d686bfa7e72d478382c8cf2e963a09de30e903` |
| Hardware | Apple M4 Pro, model `Mac16,7`, 48 GiB unified memory |
| Independent runs | 8 fresh server processes |
| Arrival seeds | 42 through 49 |
| Rate order | Balanced cyclic rotations; each rate appears twice in each position |
| Arrival windows | 60 seconds per rate, followed by full drain |
| Offered rates | 2, 4, 8, and 12 requests/s |
| Server slots | 4 |
| Context per slot | 2,048 tokens |
| Output per request | 64 tokens |
| Client workers | 32 |
| Latency SLO | Post-dispatch client-observed latency at or below 500 ms; scheduling delay excluded |
| Bootstrap | 10,000 deterministic samples of complete-run arithmetic means |
| Analysis protocol SHA-256 | `c1cd0588eb8f7a1086f564d12fd31e1458405c5aa0a62f5323bd1404c3606f8c` |

The bootstrap resamples complete fresh-server runs, not individual requests.
Reported latency values are means of the per-run median or P95 estimates. The
study completed 12,372 measured requests, plus eight excluded warm-up requests,
without client-recorded transport failures. Because only eight same-host
repetitions were available, the percentile-bootstrap intervals are conditional,
small-sample uncertainty summaries for the run-level mean. They are not
prediction intervals or pooled-request confidence intervals.

## Results

| Offered req/s | Realized req/s | Achieved req/s, mean [95% CI] | Output token/s, mean [95% CI] | Goodput req/s, mean [95% CI] | SLO attainment, mean [95% CI] |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 1.867 | 1.860 [1.706, 2.023] | 119.1 [109.2, 129.5] | 1.462 [1.010, 1.770] | 80.8% [57.6%, 93.3%] |
| 4 | 4.054 | 3.893 [3.463, 4.235] | 249.2 [221.6, 271.1] | 1.999 [1.344, 2.544] | 49.6% [33.0%, 63.7%] |
| 8 | 7.944 | 6.780 [6.111, 7.318] | 433.9 [391.1, 468.4] | 0.221 [0.042, 0.462] | 3.87% [0.60%, 8.90%] |
| 12 | 11.910 | 6.766 [5.707, 7.486] | 433.0 [365.3, 479.1] | 0.013 [0.003, 0.026] | 0.19% [0.03%, 0.38%] |

| Offered req/s | Mean run median E2E [95% CI] | Mean run P95 E2E [95% CI] | Mean run P95 client schedule delay [95% CI] | Mean peak RSS |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 437.6 ms [283.6, 733.7] | 871.2 ms [509.8, 1,571.7] | 11.0 ms [5.8, 20.5] | 611.1 MiB |
| 4 | 1,766.1 ms [461.3, 4,337.1] | 2,576.1 ms [737.5, 5,862.4] | 1,858.5 ms [7.3, 5,560.0] | 611.5 MiB |
| 8 | 3,573.0 ms [2,451.5, 4,458.7] | 5,023.3 ms [3,846.8, 6,459.6] | 6,497.4 ms [2,632.2, 11,462.7] | 614.8 MiB |
| 12 | 4,913.8 ms [4,184.2, 6,134.4] | 5,616.5 ms [4,485.8, 7,524.5] | 45,080.5 ms [29,447.7, 70,599.9] | 614.2 MiB |

## Interpretation

Under this exact protocol, observed mean throughput changes negligibly between
8 and 12 offered requests/s: 6.780 versus 6.766 achieved requests/s and 433.94
versus 433.05 output token/s. These results are consistent with a pipeline
plateau, although no equivalence margin was prespecified. The higher offered
rate is accompanied by substantially greater client scheduling delay and drain
time rather than greater measured throughput.

The latency-SLO view is substantially stricter. Mean SLO attainment falls from
80.8% at 2 requests/s to 49.6% at 4, 3.9% at 8, and 0.2% at 12. None of the
tested rates demonstrates 95% mean run-level attainment of the 500 ms
post-dispatch SLO. A lower and finer rate sweep is needed to identify a
defensible SLO capacity boundary.

At 8 requests/s, seven of eight runs reach the 32-worker cap and have
second-scale P95 client scheduling delay; at 12 requests/s, all eight do. The
throughput and drain results at these levels therefore reflect client-plus-server
pipeline constraints rather than an isolated server limit. Reported SLO
attainment uses post-dispatch request latency and excludes the separately
reported client scheduling delay. A follow-up intended to identify server-only
capacity must use a higher-capacity or asynchronous load generator and verify
that scheduling delay remains negligible relative to the arrival process.

## Run Variability

The seed-44 repetition is a real, retained outlier. It ran for 455 seconds,
compared with 277 to 305 seconds for the other seven runs. Its order was 8, 12,
2, then 4 requests/s. After draining the two saturated windows, its 2 and 4
requests/s windows remained severely degraded, despite a fresh server at the
start of the repetition and no logged server or transport errors. The seed-48
run used the same rate rotation without reproducing the full degradation.

The cause was not identified, so the primary analysis includes the run. A
post-hoc sensitivity calculation excluding seed 44 gives 92.1% SLO attainment
with a 291.0 ms mean run median at 2 requests/s, and 56.5% with a 485.2 ms mean
run median at 4 requests/s. This does not change the conclusion that 4
requests/s and above miss the 500 ms objective, but it shows that the primary
2 requests/s interval is widened materially by one repetition.

The result shows that starting one fresh server per repetition does not rule out
possible within-run carryover or other persistent within-run degradation. Exact
cyclic counterbalancing removes simple position imbalance from the run-level
means, but it does not eliminate carryover or rate-by-position interactions. The
retained outlier illustrates operational variability that a short deterministic
run could miss.

## Recommendation

- Do not use the earlier single three-second fixed-spacing result as a production
  capacity estimate.
- Treat approximately 6.8 requests/s and 433 output token/s as the descriptive
  plateau observed for this exact 32-worker, four-slot pipeline protocol, not as
  an isolated server-capacity estimate.
- Do not claim a 500 ms SLO capacity from the tested grid; add rates below 2
  requests/s.
- Before a server-only saturation claim, remove client scheduling bottlenecks and
  define a schedule-delay validity threshold.
- Retain all eight runs in the primary result and investigate carryover with a
  fresh server per rate or a longer recovery criterion between rate windows.

## Follow-Up

The requested finer sweep below 2 requests/s is reported in
[`qwen-0.5b-q4-poisson-low-rate-m4-pro.md`](qwen-0.5b-q4-poisson-low-rate-m4-pro.md).
It identifies 0.5 requests/s as the highest tested rate whose 95% bootstrap
interval for mean run-level SLO attainment remains entirely above 95%.
