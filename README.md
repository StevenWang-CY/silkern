<div align="center">

<img src="https://raw.githubusercontent.com/StevenWang-CY/silkern/main/assets/logo-text.svg" width="440" alt="SILKern — Sparse-Index Localization Kernels">

**Order, woven in.**

Deterministic sparse-index localization for context-parallel decode.

[![CI](https://github.com/StevenWang-CY/silkern/actions/workflows/ci.yml/badge.svg)](https://github.com/StevenWang-CY/silkern/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-2a78d6.svg)](https://github.com/StevenWang-CY/silkern/blob/main/pyproject.toml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-0b0b0b.svg)](https://github.com/StevenWang-CY/silkern/blob/main/LICENSE)
[![Evidence](https://img.shields.io/badge/evidence-checksummed-eb6834.svg)](https://github.com/StevenWang-CY/silkern/tree/main/evidence)

[**The contract**](docs/contract.md) · [**Determinism**](docs/determinism.md) · [**Choosing a kernel**](docs/dispatch.md) · [**Evidence**](docs/evidence.md) · [**vLLM integration**](docs/integration-vllm.md) · [**Citation**](#license-and-citation)

</div>

---

<img src="https://raw.githubusercontent.com/StevenWang-CY/silkern/main/assets/fig-problem.svg" alt="Three replays of byte-identical input. The atomic converter returns three different orders; SILKern returns one. Both return the same set and count." width="100%">

## Why SILKern?

Token-sparse decode selects **global logical** token positions. Context-parallel
attention consumes **rank-owned physical** KV-cache slots. Something has to
convert between them, and the usual converter reserves output segments with
atomic additions.

That converter is correct. It returns the right set and the right count. It just
never promised an *order* — and it doesn't deliver one. The prefix your attention
kernel consumes depends on which tile won a race, so it changes from run to run.
Floating-point reduction is not associative, so the logits change, and eventually
so does the token:

- **RL rollouts diverge** from the policy that generated them, so your advantage
  estimates are computed against a trajectory the model didn't actually take.
- **Evals move** between runs, and you cannot tell a real regression from noise.
- **Bisects find nothing.** The bug reproduces, then doesn't, then does.
- **Cached and uncached prefixes disagree**, because the tile geometry differs.

`silkern` is the same converter without that property. Same contract, same set,
same count, plus the selector's order preserved exactly. The usual fixes — sort
the prefix, add a barrier, re-scan afterward — all allocate, and allocation is
exactly what CUDA-graph capture forbids. SILKern allocates **nothing**, so the
whole thing captures and replays inside a CUDA graph.

## Highlights

- **Exact.** Both kernels reproduce an independent 40-line CPU oracle
  elementwise — set, count, *and* order — across the full geometry matrix.
- **Graph-native.** Zero device allocation, fixed addresses through
  10,000-replay gates with pointer and allocator-growth checks.
- **Faster where it works, honest where it doesn't.** The row-wide kernel does
  the atomic baseline's work in 38% less time; the one measured regression
  (1.4% at 64K, row-wide arm) is printed on this page, not in a footnote.
- **Free at full-model scale.** Complete 48-layer decode step under live
  two-GPU context parallelism: 1.000031 [0.997373, 1.002696] vs. the atomic
  baseline at 32K, under a margin fixed before any observation.
- **Evidence-first.** Every number in this README appears verbatim in
  checksummed analyzer output under [`evidence/`](evidence/) — including
  [an abstention](evidence/03-subpath-timing/) where a result did not clear
  its own prefixed margin.

## GPU support

| Architecture | Example GPU | What was verified here |
|---|---|---|
| SM90 | H100 | Live DCP-2 complete-decode canary, converter segment timing — [`evidence/05`](evidence/05-full-decode-canary/) |
| SM100 | B200 | Full oracle conformance matrix, 10,000-replay graph gates — [`evidence/01`](evidence/01-oracle-conformance-b200/) |
| SM120 | RTX 50-series | Trained-selector semantics 24/24 cells, mechanism decomposition — [`evidence/02`](evidence/02-trained-layer0-semantics/), [`evidence/04`](evidence/04-mechanism-decomposition/) |

Three generations is three, not all. **Portability is not claimed** — run
`silkern.conformance()` on your own stack before trusting any of it.

## News

- **2026-08** — `silkern` v0.1.0: initial public release (renamed from its
  working title *cleat*; same code, same evidence).
- **2026-08** — The accompanying manuscript is under anonymous peer review;
  author metadata lands here on de-anonymization.

## Install

```bash
pip install silkern            # contract + oracle, pure Python, works anywhere
pip install "silkern[gpu]"     # + torch and triton for the kernels
```

## Quick start

```python
import silkern

# 1. The contract, in pure Python. Slow, obvious, and normative.
out, counts = silkern.localize_reference(
    req_ids, block_table, rows,
    block_size=64, dcp_size=2, dcp_rank=0,
)

# 2. The GPU path. Allocate every buffer ONCE, before graph capture.
out     = torch.empty_like(token_indices)                    # (batch, top_k) int32
counts  = torch.empty(batch, dtype=torch.int32, device="cuda")

silkern.localize_rowwise(                                    # allocates nothing
    req_ids, block_table, token_indices, out, counts,
    block_size=64, dcp_size=2, dcp_rank=rank,
)

# 3. Prove it on your own stack. Do not take the numbers on this page on trust.
report = silkern.conformance()
assert report.ok, report.summary()
```

Two implementations share one contract. `localize_rowwise` scans a whole
selection row in one program — one launch, no cross-program communication.
`localize_hierarchical` bounds each program's scan to a tile and composes
deterministic tile-prefix offsets through caller-owned workspace:

```python
shapes = silkern.workspace_shapes(batch, width, tile_size=128)   # allocates nothing
ws = {n: torch.empty(s, dtype=torch.int32, device="cuda") for n, s in shapes.items()}

silkern.localize_hierarchical(
    req_ids, block_table, token_indices, out, counts,
    ws["mapped"], ws["local_positions"], ws["tile_counts"], ws["tile_offsets"],
    block_size=64, dcp_size=2, dcp_rank=rank, tile_size=128,
)
```

Pick between them with [`docs/dispatch.md`](docs/dispatch.md). Short version:
row-wide by default, hierarchical past 32K.

### See the problem yourself

```bash
python -m bench.order_instability              # counts distinct orders over identical input
python -m bench.bench_converter --with-atomic  # prices each converter on your device
```

## The contract

<img src="https://raw.githubusercontent.com/StevenWang-CY/silkern/main/assets/fig-contract.svg" alt="The localization contract in five stages, with one worked example carried through all five: request routing, ownership filter, deinterleave, paged translation, stable front compaction." width="100%">

Request routing, ownership filtering, deinterleaving, paged address translation,
and stable front compaction — with an exact valid count, fragmented non-identity
page tables, grouped interleave, and a `dcp_size == 1` bypass that matches the
converter it replaces. Stated precisely in [`docs/contract.md`](docs/contract.md);
defined normatively by `silkern.localize_reference`, which is 40 lines of
deliberately boring Python.

## What it costs

<img src="https://raw.githubusercontent.com/StevenWang-CY/silkern/main/assets/fig-cost.svg" alt="Panel a: converter segment time — row-wide 120.0 microseconds, atomic 194.4, hierarchical 239.7. Panel b: forest plot of complete decode step ratios versus the atomic baseline with 98.75 percent intervals; three contrasts sit on 1.00 inside the prespecified 1.01 margin, while row-wide at 64K exceeds it at 1.014." width="100%">

Less than nothing, at the converter. The row-wide kernel does the atomic
baseline's work in 38% less time, because a row-wide scan beats a scan *plus*
atomic traffic.

At the level of a complete 48-layer decode step under live two-GPU context
parallelism, with margins fixed before any observation, three of four contrasts
land on 1.00 — and **the fourth does not**. The row-wide arm at 64K is 1.4%
slower, reproduced in all five sessions, arising downstream of its unchanged
converter. That is why `docs/dispatch.md` says to use the hierarchical variant at
long context, and why the number is on this page instead of in a footnote.

## What is actually verified

| Claim | How it was checked | Where |
|---|---|---|
| Both kernels reproduce the contract exactly | Independent CPU oracle, elementwise, across the geometry matrix on SM90 / SM100 / SM120 | [`evidence/01`](evidence/01-oracle-conformance-b200/) |
| The prefix is the selector's order, not just its set | Order derived independently of the oracle and compared per row | `silkern.conformance()` |
| Fixed addresses, zero allocation under replay | 10,000 graph replays with pointer and allocator-growth gates | [`evidence/01`](evidence/01-oracle-conformance-b200/) |
| Nothing is written out of bounds | Guarded buffers with zeroed canary regions around every output | `silkern.conformance()` |
| It holds with a real trained selector on real text | 24/24 cells, 4K–32K, DCP-2 and DCP-4; rank-local arrays equal the oracle; recombined attention matches unsharded | [`evidence/02`](evidence/02-trained-layer0-semantics/) |
| Determinism is not paid for at the converter | Segment timing, three arms, graph-replayed | [`evidence/05`](evidence/05-full-decode-canary/) |
| Order *itself* carries no resolved cost | 8-arm decomposition; the order-only contrast is inconsistently signed | [`evidence/04`](evidence/04-mechanism-decomposition/) |
| It costs nothing on a complete decode step at 32K | 5 sessions, prefixed 1.01 non-inferiority margin, direct-blind precision stop | [`evidence/05`](evidence/05-full-decode-canary/) |

Every number quoted anywhere in this repository appears verbatim in
[`evidence/`](evidence/), including
[an abstention](evidence/03-subpath-timing/) where the result did not clear its
own prefixed margin. See [`docs/evidence.md`](docs/evidence.md).

## Scope and limits

Read this before deploying it, not after.

- **Tested geometry.** DCP-2 and DCP-4, interleave 1, page size 32/64, top-k 2048
  for the vLLM adapter. The kernels themselves handle grouped interleave and a
  wider matrix, but the *integration* has only been exercised on the narrow
  surface. Run `silkern.conformance()` on your geometry before widening it.
- **Tested hardware.** SM90 (H100), SM100 (B200), SM120 (RTX 50-series). Three
  generations is three, not all. **Portability is not claimed.**
- **The row-wide 64K cost is real** and is stated above rather than buried.
- **No serving-grade result.** All timing comes from a single-purpose reference
  executor, not a production serving path under load. There is no TPOT-under-load
  measurement here, and this repository does not claim one.
- **One model family** in the trained-weight evidence.
- Selection-row width is capped at 4096 for both kernels.

## Integrating with vLLM

[`silkern.integrations.vllm`](silkern/integrations/vllm.py) is a drop-in for an
allocation-owning `triton_filter_and_convert_dcp_index`. It is two-phase — bind
buffers once with `prepare()`, then call it on every step including inside a
replayed graph — and it **fails closed**: an unregistered geometry or a changed
input binding raises instead of reallocating, because a silent fallback inside a
captured graph writes to a stale address.

```python
from silkern.integrations.vllm import WorkspaceAdapter, install_converter

adapter = WorkspaceAdapter("row_stable")
adapter.prepare(req_id, block_table, token_indices, dcp_size=2, dcp_rank=rank)

with install_converter(sparse_backend_module, adapter):
    ...   # capture and replay; the original callable is restored on exit
```

No upstream file is modified. Details and the qualified surface:
[`docs/integration-vllm.md`](docs/integration-vllm.md).

## Contributing

The rule that matters: **the oracle is the specification.** A kernel change that
does not reproduce `localize_reference` elementwise is wrong, however fast it is.
See [`CONTRIBUTING.md`](CONTRIBUTING.md).

```bash
pip install -e ".[dev]"
python -m pytest -q -m "not gpu"      # runs anywhere
python -m pytest -q                   # adds the CUDA suites
python -m silkern                     # the conformance matrix on your device
```

## License and citation

Apache-2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

Author metadata is withheld pending peer review of the accompanying manuscript
and will be added on de-anonymization. Until then, please cite the software
release — see [`CITATION.cff`](CITATION.cff):

```bibtex
@software{silkern2026,
  title   = {SILKern: deterministic sparse-index localization for context-parallel decode},
  author  = {{The SILKern Authors}},
  year    = {2026},
  version = {0.1.0},
  license = {Apache-2.0},
  url     = {https://github.com/StevenWang-CY/silkern}
}
```
