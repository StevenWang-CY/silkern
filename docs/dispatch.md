# Choosing between the two implementations

Both satisfy the same contract and produce byte-identical output. They differ
only in how they get there, and therefore in where they are fast.

## The short answer

```
context ≤ 32K   →  localize_rowwise        (default)
context > 32K   →  localize_hierarchical
row width > 4096 →  localize_hierarchical  (row-wide is capped)
```

If you only remember one thing: **row-wide by default, hierarchical at long
context.** The rest of this page is why.

## How they differ

### `localize_rowwise`

One program per selection row. It loads the whole row (padded to a power of two),
computes ownership, deinterleave, and paged translation elementwise, then derives
each survivor's destination from a single row-wide `tl.cumsum`. One launch, no
cross-program communication, no atomics, no scratch.

The destination of every element is a pure function of the input. That is the
determinism argument, and it is a one-liner because the algorithm has no shared
mutable state at all.

Cost: the whole row must fit in one program's registers and shared memory. Row
width is capped at `MAX_ROW_WIDTH = 4096`, and occupancy falls as the row grows.

### `localize_hierarchical`

Four launches over caller-owned workspace:

1. **map tiles** — per-tile ownership, translation, tile-local prefix, tile count
2. **tile prefix** — per-row exclusive prefix over the tile counts
3. **fill** — initialize the output to `-1`
4. **scatter** — write each survivor to `tile_offset + local_position`

The tile prefix is computed in its own launch by a single program per row, so
tile bases are a deterministic function of the tile counts — the same thing the
atomic version computes, minus the race. Each program's working set is bounded by
`tile_size` regardless of row width.

Cost: four launches instead of one, and four workspace buffers you must size and
allocate before capture with `silkern.workspace_shapes()`.

## What the measurements say

Converter segment, timed in isolation as a captured graph, on two H100s
([`evidence/05`](../evidence/05-full-decode-canary/)):

| Implementation | 32K | 64K |
|---|---:|---:|
| `localize_rowwise` | 120 µs | 121 µs |
| atomic baseline | 194 µs | 194 µs |
| `localize_hierarchical` | 239 µs | 240 µs |

On the converter alone, row-wide wins outright and hierarchical is the most
expensive of the three. If converter time were the whole story, you would never
use hierarchical.

It is not the whole story. Complete 48-layer decode step, ratio against the
atomic baseline, 98.75% simultaneous intervals:

| Contrast | 32K | 64K |
|---|---|---|
| `localize_rowwise` / atomic | **1.00003** [0.99737, 1.00270] | **1.01401** [1.01033, 1.01770] |
| `localize_hierarchical` / atomic | 1.00058 [0.99780, 1.00337] | 1.00218 [0.99575, 1.00865] |

At 64K the row-wide arm costs about **1.4%** of the complete step — roughly 1 ms,
reproduced in all five sessions — while its own converter segment is unchanged at
121 µs. The cost is downstream, not in the converter. The hierarchical arm stays
at parity at both contexts.

**Read that carefully before optimizing.** A converter-only microbenchmark would
tell you to use row-wide everywhere, and at 64K it would be wrong. This is why
`bench/bench_converter.py` says in its own docstring that a converter that wins
there can still lose end to end.

### Why (as far as the evidence goes)

A frozen 8-arm decomposition
([`evidence/04`](../evidence/04-mechanism-decomposition/)) separates converter
work from consumed-prefix order. Converter work resolves a difference of about
2 µs at every context. Order alone does **not**: its effect is inconsistently
signed — −1.08 µs at 4K, −0.79 µs at 16K, +2.20 µs at 32K — which is the shape of
no effect, not of a small one.

So the 64K row-wide cost is not "the consumer dislikes this order." The most
likely remaining explanation is the row-wide kernel's own execution context: a
single program holding a 4096-wide row occupies registers and shared memory that
neighboring work wants, and at 64K there is more neighboring work. That is an
inference from what the counters could not see, not an observation, and it is
stated as such here and in the evidence.

## Picking `tile_size`

Supported: 64, 128, 256. Default 128.

Larger tiles mean fewer tiles per row, so the tile-prefix pass is cheaper and the
scatter is more coalesced — at the cost of a larger per-program working set,
which is the thing hierarchical exists to bound. 128 was the default throughout
the recorded evidence. If you change it, re-run `silkern.conformance(tile_size=...)`
on your geometry; the harness sweeps the boundary cases that a partial final tile
exposes.

## Before you trust any of this on your hardware

Everything above was measured on specific devices. Reproduce it:

```bash
python -m bench.bench_converter --with-atomic --width 2048 --dcp-size 2
```

and confirm correctness first:

```bash
python -m silkern
```

A converter-segment ranking that differs from the table above is a legitimate
result about your stack, not a bug. A `silkern.conformance()` failure is a bug —
please report it with the geometry line from `report.summary()`.
