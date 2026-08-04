"""Caller-owned workspace sizing for the hierarchical implementation.

Allocation is the enemy of CUDA-graph capture. The hierarchical kernel needs
scratch, so it asks the caller for it: you size the buffers with
:func:`workspace_shapes`, allocate once before capture, and pass the same
addresses on every replay. This module allocates nothing and imports no GPU
library on purpose -- you can plan a capture without a device present.
"""

from __future__ import annotations

from numbers import Integral

from silkern.contract import (
    DEFAULT_TILE_SIZE,
    MAX_ROW_WIDTH,
    SUPPORTED_TILE_SIZES,
)
from silkern.errors import LocalizationError


def workspace_shapes(
    batch: int,
    width: int,
    *,
    tile_size: int = DEFAULT_TILE_SIZE,
) -> dict[str, tuple[int, ...]]:
    """Return the exact caller-owned int32 workspace shapes.

    This helper performs no allocation and intentionally returns plain tuples so
    callers can provision buffers before CUDA-graph capture.
    """
    if any(
        not isinstance(value, Integral) or isinstance(value, bool)
        for value in (batch, width, tile_size)
    ):
        raise LocalizationError("batch, width, and tile_size must be integers")
    if batch < 1 or width < 1:
        raise LocalizationError("batch and width must be positive")
    if width > MAX_ROW_WIDTH:
        raise LocalizationError(
            f"exact row width exceeds the current {MAX_ROW_WIDTH}-element limit"
        )
    if tile_size not in SUPPORTED_TILE_SIZES:
        supported = ", ".join(str(value) for value in SUPPORTED_TILE_SIZES)
        raise LocalizationError(f"tile_size must be one of {supported}")
    tiles = (width + tile_size - 1) // tile_size
    return {
        "mapped": (batch, width),
        "local_positions": (batch, width),
        "tile_counts": (batch, tiles),
        "tile_offsets": (batch, tiles),
    }
