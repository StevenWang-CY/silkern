"""An atomic-reservation converter, included so you can watch it misbehave.

This is a baseline, not a product. It implements the same localization contract
as :mod:`silkern.contract` -- request routing, ownership, deinterleave, paged
translation, front compaction -- using the strategy production converters use:

    per tile:  scan the tile locally, then reserve a contiguous output segment
               for the whole tile with one `atomic_add` on a per-row counter,
               then scatter the tile's survivors into that segment.

Every part of that is correct except one thing nobody promised: *which* tile
wins the race for the first segment. The output holds the same set and the same
count on every run, and the elements land in a different order. Downstream
floating-point reduction is not associative, so a different order is a different
result.

Run ``python -m bench.order_instability`` to see the order change under your own
hands, and ``python -m bench.bench_converter --with-atomic`` to price it.

The kernel is allocation-free like the rest of the package: pass a ``scratch``
buffer of shape ``(batch,)`` for the reservation counters.
"""

from __future__ import annotations

from silkern.contract import _validate_dcp_config
from silkern.errors import LocalizationError

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # pragma: no cover - CPU-only environments
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _reset_kernel(counters_ptr, out_ptr, batch, elements, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offset = pid * BLOCK + tl.arange(0, BLOCK)
        tl.store(out_ptr + offset, -1, mask=offset < elements)
        tl.store(counters_ptr + offset, 0, mask=offset < batch)

    @triton.jit
    def _atomic_reserve_kernel(
        req_ids_ptr,
        block_table_ptr,
        tokens_ptr,
        out_ptr,
        counters_ptr,
        bt_stride0: tl.constexpr,
        bt_stride1: tl.constexpr,
        token_stride0: tl.constexpr,
        token_stride1: tl.constexpr,
        out_stride0: tl.constexpr,
        out_stride1: tl.constexpr,
        WIDTH: tl.constexpr,
        BLOCK_TABLE_WIDTH: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        DCP_SIZE: tl.constexpr,
        DCP_RANK: tl.constexpr,
        DCP_INTERLEAVE: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        tile = tl.program_id(1)
        row_i64 = row.to(tl.int64)
        lane = tl.arange(0, BLOCK_N)
        column = tile * BLOCK_N + lane
        in_width = column < WIDTH
        token = tl.load(
            tokens_ptr + row_i64 * token_stride0 + column * token_stride1,
            mask=in_width,
            other=-1,
        )
        valid_token = in_width & (token >= 0)
        owner = (token // DCP_INTERLEAVE) % DCP_SIZE
        owned = valid_token & (owner == DCP_RANK)
        local_token = (
            token // (DCP_SIZE * DCP_INTERLEAVE)
        ) * DCP_INTERLEAVE + token % DCP_INTERLEAVE
        logical_block = local_token // BLOCK_SIZE
        offset = local_token % BLOCK_SIZE
        in_table = owned & (logical_block >= 0) & (logical_block < BLOCK_TABLE_WIDTH)
        request = tl.load(req_ids_ptr + row)
        physical_block = tl.load(
            block_table_ptr
            + request.to(tl.int64) * bt_stride0
            + logical_block * bt_stride1,
            mask=in_table,
            other=-1,
        )
        physical = physical_block * BLOCK_SIZE + offset

        valid_i = in_table.to(tl.int32)
        tile_count = tl.sum(valid_i, axis=0)
        local_position = tl.cumsum(valid_i, axis=0) - valid_i

        # The whole story is this line. Tiles reserve their segment in whatever
        # order they reach the atomic, so a tile's base -- and therefore the
        # position of its elements in the output -- is not a function of the
        # input alone.
        base = tl.atomic_add(counters_ptr + row, tile_count)

        tl.store(
            out_ptr + row_i64 * out_stride0 + (base + local_position) * out_stride1,
            physical,
            mask=in_table,
        )


def launch_atomic_baseline(
    req_ids,
    block_table,
    tokens,
    out,
    counts,
    scratch,
    *,
    block_size: int,
    dcp_size: int,
    dcp_rank: int,
    dcp_interleave: int = 1,
    block_n: int = 128,
) -> None:
    """Run the atomic-reservation baseline. Correct set, unspecified order.

    ``scratch`` is an int32 ``(batch,)`` buffer used for the reservation
    counters; ``counts`` receives the final per-row valid count, which is
    exactly the total the counters accumulated.
    """
    if triton is None:
        raise LocalizationError("triton is required for the atomic baseline")
    import torch

    tensors = (req_ids, block_table, tokens, out, counts, scratch)
    if any(not isinstance(t, torch.Tensor) for t in tensors):
        raise LocalizationError("all baseline buffers must be torch tensors")
    if any(not t.is_cuda for t in tensors):
        raise LocalizationError("all baseline buffers must be CUDA tensors")
    if any(t.dtype != torch.int32 for t in tensors):
        raise LocalizationError("all baseline buffers must use int32")
    if any(not t.is_contiguous() for t in tensors):
        raise LocalizationError("the baseline requires contiguous tensors")
    if tokens.ndim != 2 or out.shape != tokens.shape:
        raise LocalizationError("tokens and out must have the same 2D shape")
    _validate_dcp_config(dcp_size, dcp_rank, dcp_interleave)

    batch, width = tokens.shape
    if counts.shape != (batch,) or scratch.shape != (batch,):
        raise LocalizationError("counts and scratch must hold one element per row")

    elements = batch * width
    reset_block = 256
    _reset_kernel[(triton.cdiv(max(elements, batch), reset_block),)](
        scratch,
        out,
        batch,
        elements,
        BLOCK=reset_block,
        num_warps=4,
    )
    _atomic_reserve_kernel[(batch, triton.cdiv(width, block_n))](
        req_ids,
        block_table,
        tokens,
        out,
        scratch,
        block_table.stride(0),
        block_table.stride(1),
        tokens.stride(0),
        tokens.stride(1),
        out.stride(0),
        out.stride(1),
        WIDTH=width,
        BLOCK_TABLE_WIDTH=block_table.shape[1],
        BLOCK_SIZE=block_size,
        DCP_SIZE=dcp_size,
        DCP_RANK=dcp_rank,
        DCP_INTERLEAVE=dcp_interleave,
        BLOCK_N=block_n,
        num_warps=4,
    )
    counts.copy_(scratch)
