# E76-A trained selection-to-projection performance

Decision: **abstain_selection_to_projection_subpath**.
Stop reason: `precision_reached`.
Fresh-process sessions: 9.
CUDA-event observations: 11664.

## Primary DCP-2 selector-included row-stable contrast

| context | ratio | 98.75% interval | saving us | 98.75% interval | log half-width |
|---:|---:|---:|---:|---:|---:|
| 4096 | 0.998204 | [0.997672, 0.998736] | 1.762 | [1.241, 2.284] | 0.000533 |
| 8192 | 0.993412 | [0.992720, 0.994105] | 6.621 | [5.923, 7.318] | 0.000697 |
| 16384 | 0.998066 | [0.997432, 0.998701] | 2.018 | [1.356, 2.680] | 0.000635 |
| 32768 | 0.997376 | [0.996650, 0.998103] | 2.825 | [2.040, 3.610] | 0.000729 |

The comparator is the source-pinned vLLM atomic converter, not a Keye-native runtime baseline. Prefix cache construction, KV append, all later layers, communication, full decode, and quality are outside the measured denominator. This result cannot unlock the second tranche.
