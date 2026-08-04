"""Allocation-free Triton implementations of the localization contract.

Two implementations, same contract, different dispatch envelopes:

``localize_rowwise``
    One program scans one whole selection row. The stable prefix falls out of a
    single row-wide ``cumsum`` -- no cross-program communication, no atomics,
    one kernel launch. Cheapest converter of the three measured
    (see ``docs/dispatch.md``), and the right default at moderate context.

``localize_hierarchical``
    Bounds each program's scan to one tile, composes deterministic tile-prefix
    offsets through caller-owned workspace, and performs a stable scatter. Four
    launches instead of one, but each program's working set is bounded, which
    is what keeps it flat at long context.

Neither launcher allocates device memory. Every buffer -- inputs, outputs, and
workspace -- is supplied by the caller, so the same addresses can be captured
in a CUDA graph and replayed indefinitely. Both validate aggressively and raise
:class:`~silkern.errors.LocalizationError` rather than degrade.

``torch`` is imported lazily inside the launchers so that importing ``silkern``
costs nothing on a CPU-only machine.
"""

from __future__ import annotations

from numbers import Integral

from silkern.contract import (
    DEFAULT_TILE_SIZE,
    MAX_ROW_WIDTH,
    _validate_compact_flag,
    _validate_dcp_config,
)
from silkern.errors import LocalizationError
from silkern.workspace import workspace_shapes

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # CPU-only environments; launchers fail closed.
    triton = None
    tl = None


# Kernel definitions are guarded so that ``import silkern`` succeeds without a GPU
# stack; the launchers below raise LocalizationError if called anyway.
if triton is not None:

    @triton.jit
    def _rowwise_kernel(
        req_ids_ptr,
        block_table_ptr,
        tokens_ptr,
        out_ptr,
        counts_ptr,
        bt_stride0: tl.constexpr,
        bt_stride1: tl.constexpr,
        token_stride0: tl.constexpr,
        token_stride1: tl.constexpr,
        out_stride0: tl.constexpr,
        out_stride1: tl.constexpr,
        WIDTH: tl.constexpr,
        PADDED_WIDTH: tl.constexpr,
        BLOCK_TABLE_WIDTH: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        DCP_SIZE: tl.constexpr,
        DCP_RANK: tl.constexpr,
        DCP_INTERLEAVE: tl.constexpr,
        COMPACT_TO_FRONT: tl.constexpr,
    ):
        row = tl.program_id(0)
        row_i64 = row.to(tl.int64)
        column = tl.arange(0, PADDED_WIDTH)
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
        request_i64 = request.to(tl.int64)
        physical_block = tl.load(
            block_table_ptr
            + request_i64 * bt_stride0
            + logical_block * bt_stride1,
            mask=in_table,
            other=-1,
        )
        mapped_valid = in_table
        physical = physical_block * BLOCK_SIZE + offset
        valid_i = mapped_valid.to(tl.int32)
        total = tl.sum(valid_i, axis=0)

        if COMPACT_TO_FRONT:
            position = tl.cumsum(valid_i, axis=0) - valid_i
            tl.store(
                out_ptr + row_i64 * out_stride0 + column * out_stride1,
                tl.full((PADDED_WIDTH,), -1, tl.int32),
                mask=in_width,
            )
            tl.debug_barrier()
            tl.store(
                out_ptr + row_i64 * out_stride0 + position * out_stride1,
                physical,
                mask=mapped_valid,
            )
        else:
            value = tl.where(mapped_valid, physical, -1)
            tl.store(
                out_ptr + row_i64 * out_stride0 + column * out_stride1,
                value,
                mask=in_width,
            )
        tl.store(counts_ptr + row, total)

    @triton.jit
    def _map_tiles_kernel(
        req_ids_ptr,
        block_table_ptr,
        tokens_ptr,
        out_ptr,
        mapped_ptr,
        local_positions_ptr,
        tile_counts_ptr,
        bt_stride0: tl.constexpr,
        bt_stride1: tl.constexpr,
        token_stride0: tl.constexpr,
        token_stride1: tl.constexpr,
        out_stride0: tl.constexpr,
        out_stride1: tl.constexpr,
        mapped_stride0: tl.constexpr,
        mapped_stride1: tl.constexpr,
        positions_stride0: tl.constexpr,
        positions_stride1: tl.constexpr,
        tile_counts_stride0: tl.constexpr,
        tile_counts_stride1: tl.constexpr,
        WIDTH: tl.constexpr,
        BLOCK_TABLE_WIDTH: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        DCP_SIZE: tl.constexpr,
        DCP_RANK: tl.constexpr,
        DCP_INTERLEAVE: tl.constexpr,
        TILE_SIZE: tl.constexpr,
        DIRECT_OUTPUT: tl.constexpr,
    ):
        row = tl.program_id(0)
        tile = tl.program_id(1)
        row_i64 = row.to(tl.int64)
        tile_i64 = tile.to(tl.int64)
        lane = tl.arange(0, TILE_SIZE)
        column = tile * TILE_SIZE + lane
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
        request_i64 = request.to(tl.int64)
        physical_block = tl.load(
            block_table_ptr
            + request_i64 * bt_stride0
            + logical_block * bt_stride1,
            mask=in_table,
            other=-1,
        )
        physical = physical_block * BLOCK_SIZE + offset
        valid_i = in_table.to(tl.int32)
        local_position = tl.cumsum(valid_i, axis=0) - valid_i

        tl.store(
            mapped_ptr + row_i64 * mapped_stride0 + column * mapped_stride1,
            physical,
            mask=in_table,
        )
        tl.store(
            local_positions_ptr
            + row_i64 * positions_stride0
            + column * positions_stride1,
            tl.where(in_table, local_position, -1),
            mask=in_width,
        )
        tl.store(
            tile_counts_ptr
            + row_i64 * tile_counts_stride0
            + tile_i64 * tile_counts_stride1,
            tl.sum(valid_i, axis=0),
        )
        if DIRECT_OUTPUT:
            value = tl.where(in_table, physical, -1)
            tl.store(
                out_ptr + row_i64 * out_stride0 + column * out_stride1,
                value,
                mask=in_width,
            )

    @triton.jit
    def _tile_prefix_kernel(
        tile_counts_ptr,
        tile_offsets_ptr,
        counts_ptr,
        tile_counts_stride0: tl.constexpr,
        tile_counts_stride1: tl.constexpr,
        tile_offsets_stride0: tl.constexpr,
        tile_offsets_stride1: tl.constexpr,
        NUM_TILES: tl.constexpr,
        PADDED_TILES: tl.constexpr,
    ):
        row = tl.program_id(0)
        row_i64 = row.to(tl.int64)
        tile = tl.arange(0, PADDED_TILES)
        in_tiles = tile < NUM_TILES
        tile_count = tl.load(
            tile_counts_ptr
            + row_i64 * tile_counts_stride0
            + tile * tile_counts_stride1,
            mask=in_tiles,
            other=0,
        )
        tile_offset = tl.cumsum(tile_count, axis=0) - tile_count
        tl.store(
            tile_offsets_ptr
            + row_i64 * tile_offsets_stride0
            + tile * tile_offsets_stride1,
            tile_offset,
            mask=in_tiles,
        )
        tl.store(counts_ptr + row, tl.sum(tile_count, axis=0))

    @triton.jit
    def _fill_output_kernel(
        out_ptr,
        elements,
        BLOCK: tl.constexpr,
    ):
        offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(out_ptr + offset, -1, mask=offset < elements)

    @triton.jit
    def _scatter_tiles_kernel(
        out_ptr,
        mapped_ptr,
        local_positions_ptr,
        tile_counts_ptr,
        tile_offsets_ptr,
        out_stride0: tl.constexpr,
        out_stride1: tl.constexpr,
        mapped_stride0: tl.constexpr,
        mapped_stride1: tl.constexpr,
        positions_stride0: tl.constexpr,
        positions_stride1: tl.constexpr,
        tile_counts_stride0: tl.constexpr,
        tile_counts_stride1: tl.constexpr,
        tile_offsets_stride0: tl.constexpr,
        tile_offsets_stride1: tl.constexpr,
        WIDTH: tl.constexpr,
        TILE_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        tile = tl.program_id(1)
        row_i64 = row.to(tl.int64)
        tile_i64 = tile.to(tl.int64)
        lane = tl.arange(0, TILE_SIZE)
        column = tile * TILE_SIZE + lane
        in_width = column < WIDTH
        tile_count = tl.load(
            tile_counts_ptr
            + row_i64 * tile_counts_stride0
            + tile_i64 * tile_counts_stride1
        )
        tile_offset = tl.load(
            tile_offsets_ptr
            + row_i64 * tile_offsets_stride0
            + tile_i64 * tile_offsets_stride1
        )
        local_position = tl.load(
            local_positions_ptr
            + row_i64 * positions_stride0
            + column * positions_stride1,
            mask=in_width,
            other=-1,
        )
        mapped = tl.load(
            mapped_ptr + row_i64 * mapped_stride0 + column * mapped_stride1,
            mask=in_width,
            other=-1,
        )
        # Valid lanes are not necessarily the first ``tile_count`` input lanes.
        valid = in_width & (local_position >= 0) & (local_position < tile_count)
        destination = tile_offset + local_position
        tl.store(
            out_ptr + row_i64 * out_stride0 + destination * out_stride1,
            mapped,
            mask=valid,
        )


def localize_rowwise(
    req_ids,
    block_table,
    tokens,
    out,
    counts,
    *,
    block_size: int,
    dcp_size: int,
    dcp_rank: int,
    dcp_interleave: int = 1,
    compact_valid_to_front: bool = True,
    num_warps: int = 8,
) -> None:
    """Row-wide stable localization: one program per selection row, one launch.

    Every buffer is a preallocated contiguous int32 CUDA tensor supplied by the
    caller; nothing is allocated here, so the launch is safe to capture in a
    CUDA graph and replay at fixed addresses. ``out`` must be shaped like
    ``tokens`` and ``counts`` must hold one element per row. Outputs must not
    share storage with inputs.

    The row width is capped at :data:`silkern.contract.MAX_ROW_WIDTH` because the
    row-wide scan is a single ``cumsum`` over a power-of-two-padded row; wider
    rows belong to :func:`localize_hierarchical`.

    Raises :class:`~silkern.errors.LocalizationError` on any unsupported geometry
    or buffer binding rather than falling back to a different code path.
    """
    if triton is None:
        raise LocalizationError("triton is required for stable localization")
    import torch

    tensors = (req_ids, block_table, tokens, out, counts)
    if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise LocalizationError("all localization buffers must be torch tensors")
    if any(not tensor.is_cuda for tensor in tensors):
        raise LocalizationError("all localization buffers must be CUDA tensors")
    if any(tensor.device != tokens.device for tensor in tensors):
        raise LocalizationError("all localization buffers must be on the same device")
    if any(tensor.dtype != torch.int32 for tensor in tensors):
        raise LocalizationError("all localization buffers must use int32")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise LocalizationError("stable localization requires contiguous tensors")
    if tokens.ndim != 2 or out.shape != tokens.shape:
        raise LocalizationError("tokens and out must have the same 2D shape")
    if block_table.ndim != 2:
        raise LocalizationError("block_table must be 2D")
    batch, width = tokens.shape
    if req_ids.shape != (batch,) or counts.shape != (batch,):
        raise LocalizationError("req_ids and counts must contain one element per row")
    if (
        batch < 1
        or width < 1
        or block_table.shape[0] < 1
        or block_table.shape[1] < 1
    ):
        raise LocalizationError(
            "batch, width, and both block-table dimensions must be positive"
        )
    if width > MAX_ROW_WIDTH:
        raise LocalizationError(
            f"exact row width exceeds the current {MAX_ROW_WIDTH}-element limit"
        )
    input_storage = {
        req_ids.untyped_storage().data_ptr(),
        block_table.untyped_storage().data_ptr(),
        tokens.untyped_storage().data_ptr(),
    }
    if out.untyped_storage().data_ptr() in input_storage:
        raise LocalizationError("out must not share storage with an input")
    if counts.untyped_storage().data_ptr() in input_storage | {
        out.untyped_storage().data_ptr()
    }:
        raise LocalizationError("counts must not share storage with another buffer")
    _validate_dcp_config(dcp_size, dcp_rank, dcp_interleave)
    _validate_compact_flag(compact_valid_to_front)
    if (
        not isinstance(block_size, Integral)
        or isinstance(block_size, bool)
        or block_size < 1
        or block_size % dcp_interleave
    ):
        raise LocalizationError(
            "block_size must be a positive integer divisible by dcp_interleave"
        )
    if (
        not isinstance(num_warps, Integral)
        or isinstance(num_warps, bool)
        or num_warps not in (4, 8)
    ):
        raise LocalizationError("num_warps must be 4 or 8")

    padded = triton.next_power_of_2(width)
    compact = compact_valid_to_front and dcp_size > 1
    _rowwise_kernel[(batch,)](
        req_ids,
        block_table,
        tokens,
        out,
        counts,
        block_table.stride(0),
        block_table.stride(1),
        tokens.stride(0),
        tokens.stride(1),
        out.stride(0),
        out.stride(1),
        WIDTH=width,
        PADDED_WIDTH=padded,
        BLOCK_TABLE_WIDTH=block_table.shape[1],
        BLOCK_SIZE=block_size,
        DCP_SIZE=dcp_size,
        DCP_RANK=dcp_rank,
        DCP_INTERLEAVE=dcp_interleave,
        COMPACT_TO_FRONT=compact,
        num_warps=num_warps,
    )

def localize_hierarchical(
    req_ids,
    block_table,
    tokens,
    out,
    counts,
    mapped_workspace,
    local_positions_workspace,
    tile_counts_workspace,
    tile_offsets_workspace,
    *,
    block_size: int,
    dcp_size: int,
    dcp_rank: int,
    dcp_interleave: int = 1,
    compact_valid_to_front: bool = True,
    tile_size: int = DEFAULT_TILE_SIZE,
    num_warps: int = 4,
) -> None:
    """Hierarchical stable localization: bounded tile scans plus a stable scatter.

    ``mapped_workspace`` and ``local_positions_workspace`` must match the input
    shape.  The two tile workspaces must have shape
    ``(batch, ceil(width / tile_size))``.  Every buffer is an int32 contiguous
    CUDA tensor on one device and must use distinct storage.

    The compacting path launches four deterministic stages: per-tile mapping and
    local prefix, per-row tile prefix, output initialization, and stable scatter.
    The non-compacting path writes columns directly in the mapping stage and
    launches only the tile-prefix stage to produce counts.
    """
    if triton is None:
        raise LocalizationError("triton is required for stable localization")
    import torch

    tensors = (
        req_ids,
        block_table,
        tokens,
        out,
        counts,
        mapped_workspace,
        local_positions_workspace,
        tile_counts_workspace,
        tile_offsets_workspace,
    )
    if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise LocalizationError(
            "all hierarchical localization buffers must be torch tensors"
        )
    if any(not tensor.is_cuda for tensor in tensors):
        raise LocalizationError(
            "all hierarchical localization buffers must be CUDA tensors"
        )
    if any(tensor.device != tokens.device for tensor in tensors):
        raise LocalizationError(
            "all hierarchical localization buffers must be on the same device"
        )
    if any(tensor.dtype != torch.int32 for tensor in tensors):
        raise LocalizationError(
            "all hierarchical localization buffers must use int32"
        )
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise LocalizationError(
            "hierarchical stable localization requires contiguous tensors"
        )
    if tokens.ndim != 2 or out.shape != tokens.shape:
        raise LocalizationError("tokens and out must have the same 2D shape")
    if block_table.ndim != 2:
        raise LocalizationError("block_table must be 2D")
    batch, width = tokens.shape
    if req_ids.shape != (batch,) or counts.shape != (batch,):
        raise LocalizationError("req_ids and counts must contain one element per row")
    if (
        batch < 1
        or width < 1
        or block_table.shape[0] < 1
        or block_table.shape[1] < 1
    ):
        raise LocalizationError(
            "batch, width, and both block-table dimensions must be positive"
        )

    shapes = workspace_shapes(
        batch,
        width,
        tile_size=tile_size,
    )
    observed_shapes = {
        "mapped": tuple(mapped_workspace.shape),
        "local_positions": tuple(local_positions_workspace.shape),
        "tile_counts": tuple(tile_counts_workspace.shape),
        "tile_offsets": tuple(tile_offsets_workspace.shape),
    }
    for name, expected in shapes.items():
        if observed_shapes[name] != expected:
            raise LocalizationError(
                f"{name} workspace shape mismatch: expected {expected}, "
                f"observed {observed_shapes[name]}"
            )

    storage_pointers = [tensor.untyped_storage().data_ptr() for tensor in tensors]
    if len(set(storage_pointers)) != len(storage_pointers):
        raise LocalizationError(
            "hierarchical localization buffers must use distinct storage"
        )
    _validate_dcp_config(dcp_size, dcp_rank, dcp_interleave)
    _validate_compact_flag(compact_valid_to_front)
    if (
        not isinstance(block_size, Integral)
        or isinstance(block_size, bool)
        or block_size < 1
        or block_size % dcp_interleave
    ):
        raise LocalizationError(
            "block_size must be a positive integer divisible by dcp_interleave"
        )
    if (
        not isinstance(num_warps, Integral)
        or isinstance(num_warps, bool)
        or num_warps not in (4, 8)
    ):
        raise LocalizationError("num_warps must be 4 or 8")

    num_tiles = shapes["tile_counts"][1]
    compact = compact_valid_to_front and dcp_size > 1
    _map_tiles_kernel[(batch, num_tiles)](
        req_ids,
        block_table,
        tokens,
        out,
        mapped_workspace,
        local_positions_workspace,
        tile_counts_workspace,
        block_table.stride(0),
        block_table.stride(1),
        tokens.stride(0),
        tokens.stride(1),
        out.stride(0),
        out.stride(1),
        mapped_workspace.stride(0),
        mapped_workspace.stride(1),
        local_positions_workspace.stride(0),
        local_positions_workspace.stride(1),
        tile_counts_workspace.stride(0),
        tile_counts_workspace.stride(1),
        WIDTH=width,
        BLOCK_TABLE_WIDTH=block_table.shape[1],
        BLOCK_SIZE=block_size,
        DCP_SIZE=dcp_size,
        DCP_RANK=dcp_rank,
        DCP_INTERLEAVE=dcp_interleave,
        TILE_SIZE=tile_size,
        DIRECT_OUTPUT=not compact,
        num_warps=num_warps,
    )
    padded_tiles = triton.next_power_of_2(num_tiles)
    _tile_prefix_kernel[(batch,)](
        tile_counts_workspace,
        tile_offsets_workspace,
        counts,
        tile_counts_workspace.stride(0),
        tile_counts_workspace.stride(1),
        tile_offsets_workspace.stride(0),
        tile_offsets_workspace.stride(1),
        NUM_TILES=num_tiles,
        PADDED_TILES=padded_tiles,
        num_warps=1,
    )
    if compact:
        elements = batch * width
        fill_block = 256
        _fill_output_kernel[(triton.cdiv(elements, fill_block),)](
            out,
            elements,
            BLOCK=fill_block,
            num_warps=4,
        )
        _scatter_tiles_kernel[(batch, num_tiles)](
            out,
            mapped_workspace,
            local_positions_workspace,
            tile_counts_workspace,
            tile_offsets_workspace,
            out.stride(0),
            out.stride(1),
            mapped_workspace.stride(0),
            mapped_workspace.stride(1),
            local_positions_workspace.stride(0),
            local_positions_workspace.stride(1),
            tile_counts_workspace.stride(0),
            tile_counts_workspace.stride(1),
            tile_offsets_workspace.stride(0),
            tile_offsets_workspace.stride(1),
            WIDTH=width,
            TILE_SIZE=tile_size,
            num_warps=num_warps,
        )
