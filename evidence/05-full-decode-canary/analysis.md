# E79 full-decode canary analysis

Sessions: 5 | classification: **pass_tier1_non_inferiority** | Tier-1 True | Tier-2 False

| contrast | sessions | point | lo | hi |
|---|---:|---:|---:|---:|
| hierarchical_stable_over_atomic.c32768 | 5 | 1.000580 | 0.997800 | 1.003368 |
| hierarchical_stable_over_atomic.c65536 | 5 | 1.002178 | 0.995750 | 1.008648 |
| row_stable_over_atomic.c32768 | 5 | 1.000031 | 0.997373 | 1.002696 |
| row_stable_over_atomic.c65536 | 5 | 1.014006 | 1.010326 | 1.017700 |

reference-executor complete-decode ratios under live DCP-2; Tier-1 supports the no-measurable-cost headline only; Tier-2 alone supports speedup wording; neither is a serving-runtime, capacity, or quality-benchmark result
