# The localization contract

The normative definition is [`silkern/contract.py`](../silkern/contract.py) —
`localize_reference` is 40 lines of deliberately boring Python and is the
arbiter. This page states the same thing in prose and pins down the edge cases
that a GPU implementation is most likely to get wrong.

## Inputs

| Name | Shape | Meaning |
|---|---|---|
| `req_ids` | `(batch,)` | which request each selection row belongs to |
| `block_table` | `(requests, table_width)` | per-request page table: logical block → physical block |
| `rows` | `(batch, width)` | the selector's output: **global logical** token positions, in the selector's order |
| `block_size` | scalar | KV page size in tokens |
| `dcp_size`, `dcp_rank` | scalar | context-parallel degree and this rank's index |
| `dcp_interleave` | scalar | ownership granularity; must divide `block_size` |

## Outputs

| Name | Shape | Meaning |
|---|---|---|
| `out` | `(batch, width)` | **rank-local physical** KV slots, valid entries in a front prefix, tail `-1` |
| `counts` | `(batch,)` | the exact number of valid entries per row |

## The five stages

For each element of each row, in order:

**1. Request routing.** The row's `req_id` selects `block_table[req_id]`. Rows in
one batch may belong to different requests, and their page tables are unrelated.

**2. Ownership.**

```
owner = (token // dcp_interleave) % dcp_size
```

Elements with `owner != dcp_rank` are not this rank's problem and are dropped.
Negative tokens are invalid sentinels and are also dropped.

**3. Deinterleave.**

```
local = (token // (dcp_size * dcp_interleave)) * dcp_interleave
        + token % dcp_interleave
```

With `dcp_interleave == 1` this is just `token // dcp_size`. The general form
handles grouped interleave, where ownership rotates in runs of `dcp_interleave`
consecutive positions rather than one at a time.

**4. Paged translation.**

```
logical_block, offset = divmod(local, block_size)
physical = block_table[req_id][logical_block] * block_size + offset
```

If `logical_block >= table_width` the element is dropped (mapped to `-1`). A
block-table value that is in range is used **verbatim**, including if it is
negative — this matches the upstream precondition that a selected token addresses
a populated entry, and deliberately does not silently repair a malformed table.

**5. Stable front compaction.** Survivors are written to the front of the row,
retaining their relative input order. The tail is filled with `-1`. `counts[row]`
is the number of survivors.

## Edge cases that matter

These are where implementations diverge. All of them are covered by
`silkern.conformance()` and by `tests/test_contract.py`.

**`dcp_size == 1` bypasses compaction.** Mapped and invalid values stay in their
input columns even when `compact_valid_to_front=True`. This mirrors the
production converter's own fast path. It is not an oversight; changing it would
break drop-in compatibility.

**Valid lanes are not the first `tile_count` lanes.** In the hierarchical scatter,
a tile's survivors are scattered through the tile, not packed at its start. An
implementation that assumes otherwise passes on dense rows and silently corrupts
sparse ones. `silkern/kernels.py` carries a comment at exactly that line.

**Negative page-table entries are passed through.** In range → used verbatim.
The oracle does this; the kernels do this; the conformance harness generates them
(`rng.randrange(-2, 4096)`) specifically to check that both agree.

**Non-monotonic, fragmented page tables are the normal case.** The table is not an
identity map and is not sorted. Any implementation that works only on identity
tables is solving a different problem.

**Out-of-range global tokens.** Tokens past the end of the page table are dropped
rather than clamped or faulted. The harness generates these too.

**`block_size % dcp_interleave != 0` is rejected**, not accommodated. So is a row
width past `MAX_ROW_WIDTH` for the row-wide kernel, a non-contiguous buffer, a
wrong dtype, aliased input/output storage, and a mismatched workspace shape.
Every one raises `LocalizationError`.

## Order is a first-class output

The contract requires that the valid prefix equal the input row *filtered* by
ownership and validity, in input order. Set equality and count equality are not
sufficient, and an atomic-reservation converter satisfies both while violating
this. `silkern.conformance()` derives the expected order independently of the
oracle — a separate function, `_expected_order`, written as a filter over the
input rather than reusing the oracle's control flow — so agreement between them
is evidence rather than a tautology. `tests/test_verify.py` proves the two agree
on randomized cases, which is what makes a field disagreement meaningful.

## Guarantees the implementations add

Beyond the contract, both GPU implementations promise:

- **No device allocation.** Every buffer is caller-owned. Safe under CUDA-graph
  capture and replay at fixed addresses.
- **No writes outside the declared buffers.** Verified with zeroed guard regions
  around every output.
- **Inputs are not modified.**
- **Fail closed.** An unsupported geometry or binding raises; it never falls back
  to a different code path, because a fallback inside a captured graph writes to
  a stale address.
