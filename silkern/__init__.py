"""silkern -- deterministic sparse-index localization for context-parallel decode.

Token-sparse decode selects *global logical* positions. Context-parallel
attention consumes *rank-owned physical* KV slots. Something has to convert
between them, and the usual converter reserves output segments with atomic
additions: it yields the right set and the right count, in an order that
changes from run to run. Downstream floating-point reduction is not
associative, so the same prompt produces different logits, then different
tokens.

``silkern`` is the converter that does not do that. Same contract, same set, same
count -- plus the selector's order, preserved exactly, with no device
allocation, so the whole thing captures and replays inside a CUDA graph.

    import silkern

    # The contract, in pure Python. Slow, obvious, and normative.
    out, counts = silkern.localize_reference(
        req_ids, block_table, rows,
        block_size=64, dcp_size=2, dcp_rank=0,
    )

    # The GPU implementations. Allocate every buffer once, before capture.
    silkern.localize_rowwise(req, table, tokens, out, counts,
                           block_size=64, dcp_size=2, dcp_rank=0)

    # Prove it on your own stack before you trust it.
    report = silkern.conformance()
    assert report.ok, report.summary()

See ``docs/contract.md`` for the contract, ``docs/determinism.md`` for why the
atomic converter is unstable, and ``docs/dispatch.md`` for choosing between the
two implementations.
"""

from __future__ import annotations

from silkern.contract import (
    DEFAULT_TILE_SIZE,
    MAX_ROW_WIDTH,
    SUPPORTED_TILE_SIZES,
    localize_reference,
)
from silkern.errors import LocalizationError
from silkern.kernels import localize_hierarchical, localize_rowwise
from silkern.verify import CellReport, ConformanceReport, conformance
from silkern.workspace import workspace_shapes

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_TILE_SIZE",
    "MAX_ROW_WIDTH",
    "SUPPORTED_TILE_SIZES",
    "CellReport",
    "ConformanceReport",
    "LocalizationError",
    "conformance",
    "localize_hierarchical",
    "localize_reference",
    "localize_rowwise",
    "workspace_shapes",
]
