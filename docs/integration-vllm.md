# Integrating with vLLM

[`silkern/integrations/vllm.py`](../silkern/integrations/vllm.py) replaces the
allocation-owning `triton_filter_and_convert_dcp_index` used by the sparse MLA
attention backend, without modifying any upstream file.

## Why an adapter is needed at all

The upstream converter returns freshly allocated tensors. Under eager execution
that is fine. Under CUDA-graph capture it is not: the capture records the
addresses that existed at capture time, and a replay that allocates fresh tensors
either fails to capture or silently writes somewhere the graph is not reading.

So the adapter presents the same callable signature while keeping every output
and scratch buffer caller-owned and fixed-address. It is two-phase:

```python
from silkern.integrations.vllm import WorkspaceAdapter, install_converter

# Phase 1 — once, before warmup or capture.
adapter = WorkspaceAdapter("row_stable")          # or "hierarchical_stable"
adapter.prepare(
    req_id, block_table, token_indices,
    dcp_size=2, dcp_rank=rank,
    block_size=64, num_topk_tokens=2048,
)

# Phase 2 — every step, including inside a replayed graph. Allocates nothing.
with install_converter(sparse_backend_module, adapter):
    ...  # warmup, capture, replay

# The original callable is restored on exit, including on exception.
```

`prepare()` allocates the output, counts, and (for the hierarchical arm) the four
workspace buffers, then records the address, shape, stride, dtype, and device of
every input. Afterward:

- `adapter.output` / `adapter.counts` — the bound buffers, for wiring downstream
- `adapter.fixed_buffer_signatures` — what your own capture gate should assert
- `adapter.calls` — an invocation counter, useful for confirming the swap took

## It fails closed, on purpose

If the call geometry or any input binding differs from what was registered, the
adapter **raises `AdapterError`**. It does not reallocate, and it does not fall
back to the native converter.

This is the single most important design decision in the module. A fallback
inside a captured graph would write to a stale address and corrupt results
silently. A raised exception is loud and recoverable; silent corruption is
neither.

Concretely it rejects:

- a `dcp_size` outside `QUALIFIED_DCP_SIZES`
- `cp_kv_cache_interleave_size != 1` (the upstream indexer itself fails closed here)
- a `block_size` outside `QUALIFIED_BLOCK_SIZES`
- a top-k width outside `QUALIFIED_TOPK_WIDTHS`
- a `BLOCK_N` that differs from the wrapper being replaced
- `return_valid_counts` or `compact_valid_to_front` not `True`
- any change to an input tensor's address, shape, stride, dtype, or device
- being called at all before `prepare()`

## The qualified surface

| Parameter | Qualified values |
|---|---|
| `dcp_size` | 2, 4 |
| `cp_kv_cache_interleave_size` | 1 |
| `block_size` | 32, 64 |
| `num_topk_tokens` | 2048 |
| `BLOCK_N` | 128 |

**This is narrower than what the kernels support.** `silkern.localize_rowwise` and
`silkern.localize_hierarchical` handle grouped interleave and a much wider geometry
matrix. The *integration* has only been exercised on the surface above, and
conflating "the kernel handles it" with "the integration was tested on it" is the
easiest way to ship a wrong index.

To widen it, run `silkern.conformance()` on the new geometry first, then edit the
`QUALIFIED_*` constants. In that order.

## Upstream pinning

The module records the upstream revision its call signature, geometry envelope,
and file digests were checked against:

```python
from silkern.integrations import vllm
vllm.UPSTREAM_COMMIT                  # revision
vllm.UPSTREAM_SPARSE_UTILS_SHA256     # sparse_utils.py
vllm.UPSTREAM_WRAPPER_SHA256          # the wrapper module
vllm.UPSTREAM_INDEXER_SHA256          # the indexer
```

These are informational — nothing enforces them at import. If you are on a
different revision, diff the converter's signature and semantics before
installing the adapter, and re-check the digests. `install_converter` does verify
that the target module actually has a `triton_filter_and_convert_dcp_index`
attribute and raises if it does not, which catches the grossest form of drift.

## A capture checklist

1. `python -m silkern` passes on the device you will deploy on.
2. `prepare()` is called once, on the real tensors, before warmup.
3. Every graph bucket gets its own adapter — a different batch size is a
   different binding.
4. Your capture gate asserts `adapter.fixed_buffer_signatures` is unchanged
   before and after capture.
5. `torch.cuda.memory_allocated()` does not grow across replays.
6. `adapter.calls` increased, i.e. the swap actually took effect and you are not
   quietly still running the native converter.

Steps 4–6 are what `silkern.conformance()`'s `replay` check does for the kernels
directly; do them again around your integration, because a correct kernel wired
up wrongly is still wrong.
