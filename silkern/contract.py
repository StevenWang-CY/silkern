"""The localization contract: geometry validation and the reference oracle.

This module is pure Python. It has no GPU or Triton dependency, so it can be
imported and executed anywhere, including in CI without a device. It is the
normative definition of the contract; :mod:`silkern.kernels` must reproduce it
exactly, and :func:`silkern.conformance` is what proves that on your stack.

The contract, stated once
------------------------

Given a batch of selection rows holding *global logical* token positions, a
per-row request id, and a per-request page (block) table, produce for one
context-parallel rank the *rank-local physical* KV slots it must read, in the
order the selector returned them, plus an exact valid count.

Five things happen per element, in this order:

1. **Request routing** -- the row's request id selects its block-table row.
2. **Ownership** -- ``owner = (token // interleave) % dcp_size``; elements owned
   by another rank are dropped.
3. **Deinterleave** -- ``local = (token // (dcp_size * interleave)) * interleave
   + token % interleave``.
4. **Paged translation** -- ``physical = block_table[logical_block] * block_size
   + offset``, where ``logical_block, offset = divmod(local, block_size)``.
   Negative tokens, foreign tokens, and out-of-table blocks map to ``-1``.
5. **Stable front compaction** -- surviving values keep their relative input
   order and occupy a prefix; the tail is ``-1``.

Step 5 is the whole point. A converter that reserves output segments with
atomic additions satisfies steps 1-4 and produces the same *set* and the same
*count*, but the order of the prefix depends on the order in which tile
reservations happen to land -- which varies run to run. See ``docs/determinism.md``.

``dcp_size == 1`` bypasses compaction entirely, matching the production
converter it mirrors: mapped and invalid values stay in their input columns.
"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral

from silkern.errors import LocalizationError

MAX_ROW_WIDTH = 4096
DEFAULT_TILE_SIZE = 128
SUPPORTED_TILE_SIZES = (64, 128, 256)


def _validate_dcp_config(dcp_size: int, dcp_rank: int, dcp_interleave: int) -> None:
    if any(
        not isinstance(value, Integral) or isinstance(value, bool)
        for value in (dcp_size, dcp_rank, dcp_interleave)
    ):
        raise LocalizationError("DCP size, rank, and interleave must be integers")
    if dcp_size < 1 or not 0 <= dcp_rank < dcp_size:
        raise LocalizationError("invalid dcp configuration")
    if dcp_interleave < 1:
        raise LocalizationError("dcp_interleave must be positive")


def _validate_compact_flag(compact_valid_to_front: bool) -> None:
    if not isinstance(compact_valid_to_front, bool):
        raise LocalizationError("compact_valid_to_front must be a bool")


def localize_reference(
    req_ids: Sequence[int],
    block_table: Sequence[Sequence[int]],
    rows: Sequence[Sequence[int]],
    *,
    block_size: int,
    dcp_size: int,
    dcp_rank: int,
    dcp_interleave: int = 1,
    compact_valid_to_front: bool = True,
) -> tuple[list[list[int]], list[int]]:
    """Pure-Python oracle for the localization contract. The normative definition.

    This is the falsifier. It is deliberately the slowest and most obviously
    correct thing in the package: no vectorization, no cleverness, one
    ``for`` loop over one element at a time. When a GPU implementation and this
    function disagree, this function is right.

    For ``dcp_size > 1``, ownership and local-token conversion are:

    ``owner = (token // interleave) % dcp_size``

    ``local = (token // (dcp_size * interleave)) * interleave
             + token % interleave``

    The local logical block is then resolved through the request's block-table
    row.  Negative input tokens, tokens owned by another rank, and logical
    blocks beyond the table produce ``-1``.  A block-table value is otherwise
    used verbatim, matching the upstream precondition that selected tokens
    address populated entries.  With front compaction enabled, valid mapped values retain
    input order and occupy a prefix.

    The production converter this mirrors bypasses compaction entirely when
    ``dcp_size == 1``; so does this reference. Mapped and invalid values stay
    in their input columns even when ``compact_valid_to_front`` is true.
    """
    _validate_dcp_config(dcp_size, dcp_rank, dcp_interleave)
    _validate_compact_flag(compact_valid_to_front)
    if not isinstance(block_size, Integral) or isinstance(block_size, bool):
        raise LocalizationError("block_size must be an integer")
    if block_size < 1:
        raise LocalizationError("block_size must be positive")
    if block_size % dcp_interleave:
        raise LocalizationError("block_size must be divisible by dcp_interleave")
    if len(req_ids) != len(rows):
        raise LocalizationError("req_ids must contain one request id per row")
    if len(block_table) == 0:
        raise LocalizationError("block_table must contain at least one request row")
    if len(rows) == 0:
        raise LocalizationError("rows must contain at least one nonempty row")
    table_width = len(block_table[0])
    if table_width == 0 or any(len(row) != table_width for row in block_table):
        raise LocalizationError("block_table must be nonempty and rectangular")
    row_width = len(rows[0])
    if row_width == 0 or any(len(row) != row_width for row in rows):
        raise LocalizationError("rows must be nonempty and rectangular")

    out_rows: list[list[int]] = []
    counts: list[int] = []
    compact = compact_valid_to_front and dcp_size > 1
    for row_id, row in enumerate(rows):
        request = int(req_ids[row_id])
        if not 0 <= request < len(block_table):
            raise LocalizationError(f"request id {request} is outside block_table")
        table_row = block_table[request]
        mapped: list[int] = []
        valid_values: list[int] = []
        for raw_token in row:
            token = int(raw_token)
            physical = -1
            is_valid = False
            if token >= 0:
                owner = (token // dcp_interleave) % dcp_size
                if owner == dcp_rank:
                    local_token = (
                        token // (dcp_size * dcp_interleave)
                    ) * dcp_interleave + token % dcp_interleave
                    logical_block, offset = divmod(local_token, block_size)
                    if logical_block < len(table_row):
                        physical_block = int(table_row[logical_block])
                        physical = physical_block * block_size + offset
                        is_valid = True
            mapped.append(physical)
            if is_valid:
                valid_values.append(physical)

        counts.append(len(valid_values))
        if compact:
            out_rows.append(valid_values + [-1] * (len(row) - len(valid_values)))
        else:
            out_rows.append(mapped)
    return out_rows, counts
