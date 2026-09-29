# Qwen2.5 0.5B Q4_K_M low-rate Poisson SLO study

This follow-up adds offered rates below 2 requests/s, retaining 2 requests/s as
a cross-study anchor, to bracket the 500 ms post-dispatch latency-SLO boundary
on the tested grid. It keeps the model, host, server, request protocol,
60-second windows, eight fresh-server repetitions, Poisson seeds, bootstrap, and
balanced cyclic ordering used by the broader saturation study.

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
| Offered rates | 0.5, 1.0, 1.5, and 2.0 requests/s |
| Server slots | 4 |
| Context per slot | 2,048 tokens |
| Output per request | 64 tokens |
| Client workers | 32 |
| Latency SLO | Post-dispatch client-observed latency at or below 500 ms; scheduling delay excluded |
| Bootstrap | 10,000 deterministic samples of complete-run arithmetic means |
| Analysis protocol SHA-256 | `82e5a3fd55a8a496156887ac4e439c2d5498f95da73da49fac7fd29fb23fd982` |

The study completed 2,288 measured requests, plus eight excluded warm-up
requests, without client-recorded transport failures. Intervals are conditional,
small-sample 95% percentile-bootstrap summaries of the equal-weight run-level
mean. They are not prediction intervals or pooled-request confidence intervals.

## Results

| Offered req/s | Realized req/s | Achieved req/s, mean [95% CI] | Output token/s, mean [95% CI] | Goodput req/s, mean [95% CI] | SLO attainment, mean [95% CI] |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 0.496 | 0.496 [0.429, 0.558] | 31.7 [27.5, 35.7] | 0.492 [0.425, 0.556] | 99.17% [97.50%, 100.00%] |
| 1.0 | 0.919 | 0.919 [0.770, 1.067] | 58.8 [49.3, 68.3] | 0.866 [0.744, 0.987] | 95.22% [90.25%, 98.62%] |
| 1.5 | 1.485 | 1.484 [1.394, 1.574] | 94.9 [89.2, 100.7] | 1.373 [1.289, 1.454] | 92.62% [90.82%, 94.46%] |
| 2.0 | 1.867 | 1.866 [1.708, 2.035] | 119.4 [109.3, 130.2] | 1.664 [1.500, 1.819] | 89.18% [84.91%, 92.91%] |

| Offered req/s | Mean run median E2E [95% CI] | Mean run P95 E2E [95% CI] | Mean run P95 client schedule delay [95% CI] | Mean max in-flight | Mean peak RSS |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 268.6 ms [262.9, 274.3] | 378.2 ms [324.5, 444.0] | 17.9 ms [6.6, 37.3] | 2.00 | 601.6 MiB |
| 1.0 | 270.4 ms [262.8, 279.5] | 466.8 ms [418.5, 521.7] | 7.5 ms [5.9, 9.5] | 3.00 | 604.6 MiB |
| 1.5 | 281.4 ms [273.0, 290.4] | 511.3 ms [500.1, 521.4] | 8.9 ms [7.3, 10.4] | 4.00 | 603.2 MiB |
| 2.0 | 296.0 ms [283.4, 309.4] | 530.0 ms [511.6, 548.8] | 7.7 ms [6.2, 9.2] | 4.75 | 604.0 MiB |

## SLO Boundary

The point estimate of mean run-level SLO attainment remains above 95% through
1.0 requests/s, where it is 95.22%. Its bootstrap interval extends down to
90.25%, so this rate does not provide a conservative above-95% claim with eight
runs.

At 0.5 requests/s, mean attainment is 99.17% and the lower bootstrap bound is
97.50%. It is therefore the highest tested rate whose interval remains entirely
above 95%. One individual run attained 93.33%, so this is evidence about the
mean across runs, not a guarantee that every future run or request population
will achieve 95%.

At 1.5 requests/s, both the point estimate and upper interval bound are below
95%. Among the tested rates, 0.5 requests/s is therefore the highest and only
rate whose interval remains entirely above 95%. The adjacent tested point
estimates are above 95% at 1.0 and below 95% at 1.5 requests/s; this brackets a
crossing on the tested grid but does not locate a continuous capacity boundary.
Rates below 0.5 requests/s were not tested.

## Load-Generator Validity

Unlike the 8 and 12 requests/s windows in the broader study, this sweep does not
saturate the 32-worker client. The largest per-run maximum in-flight count is 6,
and the largest P95 scheduling delay is 83.7 ms at 0.5 requests/s in one run.
For 1.0 through 2.0 requests/s, the largest per-run P95 scheduling delay remains
below 12 ms. No low-rate window approaches the 32-worker cap, so the observed
decline does not appear attributable to worker-cap exhaustion. Other client-side
or shared-host effects are not excluded.

All eight complete repetitions last between 242.7 and 243.2 seconds. No
counterpart to the 455-second seed-44 outlier from the broader saturation study
appears here.

## Cross-Study Sensitivity at 2 Requests/s

The 2 requests/s Poisson offsets are identical by seed across this study and the
broader 2, 4, 8, 12 requests/s study, although their rate positions and preceding
loads differ. The paired mean SLO difference for low-rate minus broad study is
+8.4 percentage points, with a 95% run-level bootstrap interval from -5.2 to
+30.4 points. The interval includes zero and the difference is driven by the
seed-44 degradation in the broader study. Excluding that run, the exploratory
mean difference is approximately -2.0 points.

The aggregate paired contrast is dominated by the seed-44 pair. Because the
studies were executed at different times and used different rate positions and
preceding loads, this post-hoc sensitivity comparison does not estimate a
causal study effect or identify whether the seed-44 observation resulted from
carryover, host state, or another mechanism.

## Recommendation

- Use 0.5 requests/s as the highest tested conservative operating point for a
  claim about mean run-level attainment of the 500 ms post-dispatch SLO.
- Treat 1.0 requests/s as a borderline point estimate, not a confidence-bounded
  SLO capacity claim.
- Add rates between 0.5 and 1.0 requests/s, or more independent repetitions, if
  a less conservative capacity target is operationally important.
- Continue reporting scheduling delay separately from post-dispatch latency.
- Do not generalize these results beyond this exact model, output length,
  four-slot server configuration, host, and llama.cpp build.
