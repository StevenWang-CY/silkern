# What is established, and what is not

Software claims are cheap. This page separates what was measured from what was
inferred from what was not tested at all, so you can decide how much of it
transfers to your stack.

Raw analyzer output is in [`evidence/`](../evidence/). Every number quoted
anywhere in this repository appears verbatim in one of those files.

## Established

**The implementations reproduce the contract.** Both kernels match the
independent CPU oracle elementwise — arrays and counts — across a geometry matrix
crossing widths 1–4096, page sizes 32 and 64, DCP degrees 1/2/4/8 including the
last rank of each, and grouped interleave. Checked on SM90, SM100, and SM120.

**The prefix is the selector's order.** Checked against an order derivation
written independently of the oracle, per row, with the tail asserted to be `-1`.

**Replay stability.** 10,000 fixed-address graph replays per arm with pointer
identity and allocator-growth gates. No compiled variant spilled registers.

**Memory safety.** Guarded buffers with zeroed canary regions around every output
and workspace; inputs verified unmodified.

**It holds under a real trained selector.** With real tokenized text and trained
layer-0 selector, QKV, and output weights, 24/24 cells pass at 4K/8K/16K/32K under
DCP-2 and DCP-4: each rank's array equals the oracle, each rank's order equals
filtering the selector array by ownership, the partitions are disjoint and their
union reconstructs the returned global set, and recombined attention matches the
unsharded result (worst post-projection difference `6.1e-05`). No ownership
balance was assumed — rank shares were measured, not stipulated.

**Determinism is not paid for at the converter.** Row-wide 120 µs vs. atomic
194 µs vs. hierarchical 239 µs, timed as captured graphs.

**Order itself carries no resolved cost.** An 8-arm decomposition resolves
converter work (~2 µs at every context) but finds the order-only contrast
inconsistently signed across contexts. That is counter-evidence against the
tempting story, and it is retained rather than dropped.

**Complete-step non-inferiority at 32K.** Five fresh two-process sessions on two
H100s, a real 48-layer model, live context parallelism, randomized paired blocks,
a 1.01 margin fixed before any observation, and a direction-blind precision stop.
Primary contrast `1.00003 [0.99737, 1.00270]`. In every session the two
deterministic arms produced bitwise-identical free-running generations while the
atomic arm's varied.

## Established, and unfavorable

Kept here because a repository that only reports its wins is not evidence.

**The row-wide arm costs 1.4% at 64K.** `1.01401 [1.01033, 1.01770]` — the whole
interval sits above the prefixed margin, reproduced in all five sessions, about
+1 ms per step. It arises downstream of the converter, whose segment timing is
unchanged. Mitigation: use `localize_hierarchical` at long context. See
[`dispatch.md`](dispatch.md).

**A subpath timing study abstained.** Nine sessions, four contrasts, all four
favoring the stable arm, **none** clearing its prefixed practical margin. The
preregistered decision was an abstention and it was honored. See
[`evidence/03`](../evidence/03-subpath-timing/).

**The reference executor is not fast.** Its per-step time is inflated by a
capture-safe dense-weighted mixture-of-experts that evaluates all experts rather
than the active ones. The inflation is identical across arms, so the *ratios* are
unaffected — but the absolute step time is not representative of a tuned serving
step, and a converter's share of a faster step is correspondingly larger.

## Inferred, not observed

Stated separately because the distinction is the point.

**Why the 64K row-wide cost exists.** Per-kernel counters show the downstream
kernels are serialized-identical, and cannot see inter-kernel effects. That the
remaining difference arises *between* kernels — scheduling, cache state,
contention from the row-wide kernel's larger per-program working set — is an
inference from what the instrumentation could not observe. It is consistent with
the data and is not a measurement.

## Not established

- **Any serving-grade result.** No TPOT under load, no throughput, no goodput, no
  queueing behavior. All timing is from a single-purpose reference executor.
- **Portability.** Three GPU generations were tested. Nothing is claimed about a
  fourth, about AMD, or about a different Triton version.
- **A speedup.** The 32K result is *non-inferiority* against a margin fixed in
  advance. The converter-segment win is a segment measurement, not an end-to-end
  claim.
- **Model quality.** Determinism is not quality. No claim is made that preserving
  the selector's order improves model output — only that it stops changing.
- **Generality across model families.** The trained-weight evidence is one family.
- **Capacity or memory headroom** under production batching.

## How to check any of this

```bash
shasum -a 256 -c evidence/SHA256SUMS           # the artifacts are what they claim
python -m silkern                                # the correctness claims, on your device
python -m bench.order_instability              # the instability claim, on your device
python -m bench.bench_converter --with-atomic  # the cost claim, on your device
```

If a claim on this page does not reproduce on your hardware, that is a result
worth reporting. Include the geometry line from `report.summary()`.
