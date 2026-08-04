# Evidence

Verbatim analyzer output for the measurements quoted in the top-level README.
Nothing here was re-typed for presentation: each `analysis.json` is the file an
independent analyzer wrote, and every number in the README appears in one of
them. If a claim in the README does not appear in this directory, treat it as
unsupported and open an issue.

Only identity strings were altered — absolute paths, email addresses, and
monetary amounts were redacted. Measurements, gates, decisions, and claim
boundaries are untouched, including the parts that are unflattering.

Two things you will notice and that were deliberately **not** rewritten, because
rewriting them would forfeit the verbatim guarantee that makes these files worth
shipping: the records carry the internal campaign identifiers they were collected
under (`E75`–`E80`, referenced in schema strings and claim boundaries), and they
name the source package as `regatta/` and the model family as Keye. None of that
identifies a person; all of it is needed to cross-reference an artifact against
its own analyzer.

Verify integrity with:

```
shasum -a 256 -c SHA256SUMS
```

## What is here

| Directory | What it establishes | Hardware |
|---|---|---|
| `01-oracle-conformance-b200/` | The two implementations reproduce the CPU oracle across the geometry matrix, keep fixed addresses through 10,000 graph replays, pass guarded-memory canaries, and spill no registers. Also records compiled-variant register/shared-memory counts. | 1× B200 |
| `02-trained-layer0-semantics/` | With a real trained selector and real tokenized text, rank-local arrays equal the oracle, each rank's order equals filtering the selector array by ownership, the DCP partitions are disjoint and reconstruct the global set, and recombined attention matches the unsharded result. 24/24 cells at 4K–32K, DCP-2 and DCP-4. | 1× RTX 5060 Ti |
| `03-subpath-timing/` | A 9-session paired timing study of a trained selection-to-projection subpath. **All four intervals favor the stable arm and none clears the prefixed practical margin, so the decision is an abstention.** Retained because a campaign that only publishes its wins is not evidence. | 1× RTX 5060 Ti |
| `04-mechanism-decomposition/` | An 8-arm decomposition of where the difference comes from. Converter work resolves (~2 µs); **consumed-prefix order alone does not** — its effect is inconsistently signed across contexts (−1.08 µs at 4K, −0.79 µs at 16K, +2.20 µs at 32K). This is counter-evidence against the tempting story that order itself is what costs. | 1× RTX 5060 Ti |
| `05-full-decode-canary/` | The complete per-token decode step of a real 48-layer model under live two-GPU context parallelism, three arms, randomized paired blocks, margins fixed before observation. `segments.json` carries the converter/attention/communication segment medians (converter at 32K: row-wide 120.0 µs, atomic 194.4 µs, hierarchical 239.7 µs — the source of the "38% less time" statement). | 2× H100 80GB |

## Reading `05-full-decode-canary/analysis.json`

This is the headline measurement, and it does not say only good things.

```
row_stable_over_atomic.c32768   1.000031  [0.997373, 1.002696]   <- primary
hierarchical_stable_over_atomic.c32768   1.000580  [0.997800, 1.003368]
hierarchical_stable_over_atomic.c65536   1.002178  [0.995750, 1.008648]
row_stable_over_atomic.c65536   1.014006  [1.010326, 1.017700]   <- a real cost
```

The primary contrast is the row-wide arm at 32K, prefixed before any
observation, with a 1.01 non-inferiority margin: it passes with room to spare.
The 64K replication of the same arm does not — that interval sits entirely
above the margin. It is a genuine ~1.4% cost, reproduced in all five sessions,
and it arises downstream of the converter, whose own segment timing is
unchanged. The hierarchical arm is at parity at both contexts.

Practical consequence, stated in `docs/dispatch.md`: use `localize_hierarchical`
at long context.

## What is deliberately not here

* Raw per-session records, session logs, protocol documents, preregistration
  records, retained failure records, and the profiler counter dumps. They exist
  and are checksummed, but they are large and belong with the research
  program rather than with a tool.
* Any serving-runtime result. Every timing here comes from a single-purpose
  reference executor, not a production serving path under load. The reference
  executor computes a dense-weighted mixture-of-experts to stay
  capture-safe, which inflates the per-step denominator identically for all
  arms. The ratios are unbiased by this; the absolute step time is not
  representative of a tuned serving step.
* Any claim of portability. Three GPU generations were tested. That is three,
  not all.
