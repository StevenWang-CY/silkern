"""Drop-in replacement for a production sparse-DCP index converter.

The upstream wrapper this targets calls
``triton_filter_and_convert_dcp_index`` with an allocation-owning API: it
returns freshly allocated tensors. That is fine in eager mode and fatal under
CUDA-graph capture, so this module supplies the same callable surface while
keeping every output and scratch buffer caller-owned and fixed-address.

The lifecycle is two-phase on purpose:

1. :meth:`WorkspaceAdapter.prepare` -- once, before warmup or capture. Allocates
   the output, counts, and (hierarchical only) workspace buffers and records the
   address/shape/stride/dtype/device signature of every input.
2. :meth:`WorkspaceAdapter.__call__` -- on every step, including inside a
   replayed graph. Allocates nothing. If the call geometry or any input binding
   differs from what was registered, it **raises** rather than reallocating.

That last sentence is the design: a silent fallback inside a captured graph
would write to a stale address. Failing closed is the only safe behavior.

Installing this does not modify any upstream file --
:func:`install_converter` swaps the module attribute for the duration of a
``with`` block and restores it afterward.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Literal

from silkern.contract import DEFAULT_TILE_SIZE
from silkern.errors import LocalizationError
from silkern.kernels import localize_hierarchical, localize_rowwise
from silkern.workspace import workspace_shapes

Arm = Literal["row_stable", "hierarchical_stable"]

#: Upstream revision this adapter's call signature and geometry were checked
#: against. Verify before installing against a different revision.
UPSTREAM_COMMIT = "67f9046e4a12d939ffad0abc917585a00ef5b42d"
UPSTREAM_SPARSE_UTILS_SHA256 = (
    "a15a5767a6afe1427dccc395d7125d3f8f2b701b859c31eec91918d04d5761b4"
)
UPSTREAM_WRAPPER_SHA256 = (
    "550079022e858952c37bfa1198dbb1c77884571af30eb4c2f1a8055cd6ca7b07"
)
UPSTREAM_INDEXER_SHA256 = (
    "e9bc6fbe3bd83f2116e7ba48f3a581e0699fbcce602a816724261a13ba64f1a5"
)

QUALIFIED_DCP_SIZES = (2, 4)
QUALIFIED_INTERLEAVES = (1,)
QUALIFIED_BLOCK_SIZES = (32, 64)
QUALIFIED_TOPK_WIDTHS = (2048,)
QUALIFIED_BLOCK_N = 128


class AdapterError(LocalizationError):
    """The call geometry or fixed-address binding is unsupported."""


def _positive_int(name: str, value: int) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
        raise AdapterError(f"{name} must be a positive integer")
    return int(value)


def validate_production_call(
    *,
    dcp_size: int,
    dcp_rank: int,
    cp_kv_cache_interleave_size: int,
    block_size: int,
    num_topk_tokens: int,
    block_n: int,
    return_valid_counts: bool,
    compact_valid_to_front: bool,
) -> None:
    """Reject anything outside the geometry this adapter has been qualified on.

    This is deliberately narrower than what :mod:`silkern.kernels` supports.
    The kernels handle grouped interleave and a wide range of geometries; the
    *production integration* has only been exercised at the surface below, and
    conflating the two would be the easiest way to ship a wrong index:

    * the upstream sparse indexer itself fails closed for DCP interleave > 1;
    * the SM100 sparse backend advertises block sizes 32 and 64;
    * the compatible model/backend contract fixes top-k to 2048;
    * ``BLOCK_N`` must match the wrapper being replaced.

    Widen :data:`QUALIFIED_DCP_SIZES` and friends only after running
    :func:`silkern.conformance` on the new geometry.
    """

    size = _positive_int("dcp_size", dcp_size)
    if (
        not isinstance(dcp_rank, Integral)
        or isinstance(dcp_rank, bool)
        or dcp_rank < 0
    ):
        raise AdapterError("dcp_rank must be a nonnegative integer")
    rank = int(dcp_rank)
    interleave = _positive_int(
        "cp_kv_cache_interleave_size", cp_kv_cache_interleave_size
    )
    block = _positive_int("block_size", block_size)
    width = _positive_int("num_topk_tokens", num_topk_tokens)
    tile = _positive_int("block_n", block_n)
    if size not in QUALIFIED_DCP_SIZES:
        raise AdapterError(
            f"dcp_size={size} is outside the integrated qualification "
            f"{QUALIFIED_DCP_SIZES}"
        )
    if not 0 <= rank < size:
        raise AdapterError(f"dcp_rank={rank} is outside [0, {size})")
    if interleave not in QUALIFIED_INTERLEAVES:
        raise AdapterError(
            "the upstream sparse indexer fails closed for DCP interleave > 1"
        )
    if block not in QUALIFIED_BLOCK_SIZES:
        raise AdapterError(
            f"block_size={block} is outside the qualified backend surface "
            f"{QUALIFIED_BLOCK_SIZES}"
        )
    if width not in QUALIFIED_TOPK_WIDTHS:
        raise AdapterError(
            f"top-k width={width} is outside the compatible sparse-MLA "
            f"surface {QUALIFIED_TOPK_WIDTHS}"
        )
    if tile != QUALIFIED_BLOCK_N:
        raise AdapterError(
            f"BLOCK_N={tile} differs from the qualified wrapper value "
            f"{QUALIFIED_BLOCK_N}"
        )
    if return_valid_counts is not True:
        raise AdapterError(
            "the production sparse-attention call requires valid counts"
        )
    if compact_valid_to_front is not True:
        raise AdapterError(
            "the production sparse-attention call requires a contiguous valid prefix"
        )


def _tensor_signature(tensor: Any) -> tuple[Any, ...]:
    """Fixed-address signature without importing torch in CPU-only processes."""

    return (
        int(tensor.data_ptr()),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype),
        str(tensor.device),
    )


@dataclass
class _Binding:
    call: tuple[int, int, int, int, int, int]
    req_ids: tuple[Any, ...]
    block_table: tuple[Any, ...]
    token_indices: tuple[Any, ...]
    out: Any
    counts: Any
    workspace: dict[str, Any]


class WorkspaceAdapter:
    """A pre-registered, allocation-free stand-in for the upstream converter."""

    def __init__(
        self,
        arm: Arm,
        *,
        hierarchical_tile_size: int = DEFAULT_TILE_SIZE,
    ) -> None:
        if arm not in ("row_stable", "hierarchical_stable"):
            raise AdapterError(f"unknown arm: {arm}")
        self.arm = arm
        self.hierarchical_tile_size = _positive_int(
            "hierarchical_tile_size", hierarchical_tile_size
        )
        self._binding: _Binding | None = None
        self.calls = 0

    @staticmethod
    def _call_key(
        *,
        dcp_size: int,
        dcp_rank: int,
        cp_kv_cache_interleave_size: int,
        block_size: int,
        num_topk_tokens: int,
        block_n: int,
    ) -> tuple[int, int, int, int, int, int]:
        return (
            int(dcp_size),
            int(dcp_rank),
            int(cp_kv_cache_interleave_size),
            int(block_size),
            int(num_topk_tokens),
            int(block_n),
        )

    def prepare(
        self,
        req_id: Any,
        block_table: Any,
        token_indices: Any,
        *,
        dcp_size: int,
        dcp_rank: int,
        cp_kv_cache_interleave_size: int = 1,
        block_size: int = 64,
        num_topk_tokens: int = 2048,
        block_n: int = QUALIFIED_BLOCK_N,
    ) -> None:
        """Allocate and bind fixed buffers before warmup or graph capture."""

        validate_production_call(
            dcp_size=dcp_size,
            dcp_rank=dcp_rank,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
            block_size=block_size,
            num_topk_tokens=num_topk_tokens,
            block_n=block_n,
            return_valid_counts=True,
            compact_valid_to_front=True,
        )
        try:
            import torch
        except ModuleNotFoundError as exc:  # pragma: no cover - GPU-only branch
            raise AdapterError("torch is required to prepare CUDA buffers") from exc
        if not all(isinstance(value, torch.Tensor) for value in (
            req_id,
            block_table,
            token_indices,
        )):
            raise AdapterError("all production inputs must be torch tensors")
        if token_indices.ndim != 2:
            raise AdapterError("token_indices must be two-dimensional")
        if token_indices.shape[1] != num_topk_tokens:
            raise AdapterError(
                "token_indices width does not match NUM_TOPK_TOKENS"
            )
        if req_id.shape != (token_indices.shape[0],):
            raise AdapterError("req_id must contain one id per token row")

        out = torch.empty_like(token_indices)
        counts = torch.empty(
            token_indices.shape[0],
            dtype=torch.int32,
            device=token_indices.device,
        )
        workspace: dict[str, Any] = {}
        if self.arm == "hierarchical_stable":
            workspace = {
                name: torch.empty(
                    shape,
                    dtype=torch.int32,
                    device=token_indices.device,
                )
                for name, shape in workspace_shapes(
                    token_indices.shape[0],
                    token_indices.shape[1],
                    tile_size=self.hierarchical_tile_size,
                ).items()
            }
        self._binding = _Binding(
            call=self._call_key(
                dcp_size=dcp_size,
                dcp_rank=dcp_rank,
                cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
                block_size=block_size,
                num_topk_tokens=num_topk_tokens,
                block_n=block_n,
            ),
            req_ids=_tensor_signature(req_id),
            block_table=_tensor_signature(block_table),
            token_indices=_tensor_signature(token_indices),
            out=out,
            counts=counts,
            workspace=workspace,
        )

    @property
    def prepared(self) -> bool:
        return self._binding is not None

    @property
    def output(self) -> Any:
        if self._binding is None:
            raise AdapterError("adapter has not been prepared")
        return self._binding.out

    @property
    def counts(self) -> Any:
        if self._binding is None:
            raise AdapterError("adapter has not been prepared")
        return self._binding.counts

    @property
    def fixed_buffer_signatures(self) -> dict[str, tuple[Any, ...]]:
        if self._binding is None:
            raise AdapterError("adapter has not been prepared")
        return {
            "out": _tensor_signature(self._binding.out),
            "counts": _tensor_signature(self._binding.counts),
            **{
                name: _tensor_signature(tensor)
                for name, tensor in self._binding.workspace.items()
            },
        }

    def __call__(
        self,
        req_id: Any,
        block_table: Any,
        token_indices: Any,
        dcp_size: int,
        dcp_rank: int,
        cp_kv_cache_interleave_size: int = 1,
        BLOCK_SIZE: int = 64,
        NUM_TOPK_TOKENS: int = 2048,
        BLOCK_N: int = QUALIFIED_BLOCK_N,
        return_valid_counts: bool = False,
        compact_valid_to_front: bool = True,
    ) -> tuple[Any, Any]:
        validate_production_call(
            dcp_size=dcp_size,
            dcp_rank=dcp_rank,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
            block_size=BLOCK_SIZE,
            num_topk_tokens=NUM_TOPK_TOKENS,
            block_n=BLOCK_N,
            return_valid_counts=return_valid_counts,
            compact_valid_to_front=compact_valid_to_front,
        )
        if self._binding is None:
            raise AdapterError(
                "graph bucket is unregistered; call prepare before wrapper execution"
            )
        observed_call = self._call_key(
            dcp_size=dcp_size,
            dcp_rank=dcp_rank,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
            block_size=BLOCK_SIZE,
            num_topk_tokens=NUM_TOPK_TOKENS,
            block_n=BLOCK_N,
        )
        if observed_call != self._binding.call:
            raise AdapterError(
                f"call geometry {observed_call} differs from registered "
                f"{self._binding.call}"
            )
        observed_inputs = (
            _tensor_signature(req_id),
            _tensor_signature(block_table),
            _tensor_signature(token_indices),
        )
        expected_inputs = (
            self._binding.req_ids,
            self._binding.block_table,
            self._binding.token_indices,
        )
        if observed_inputs != expected_inputs:
            raise AdapterError(
                "production input tensor address/shape/stride binding changed"
            )

        if self.arm == "row_stable":
            localize_rowwise(
                req_id,
                block_table,
                token_indices,
                self._binding.out,
                self._binding.counts,
                block_size=BLOCK_SIZE,
                dcp_size=dcp_size,
                dcp_rank=dcp_rank,
                dcp_interleave=cp_kv_cache_interleave_size,
                compact_valid_to_front=True,
            )
        else:
            workspace = self._binding.workspace
            localize_hierarchical(
                req_id,
                block_table,
                token_indices,
                self._binding.out,
                self._binding.counts,
                workspace["mapped"],
                workspace["local_positions"],
                workspace["tile_counts"],
                workspace["tile_offsets"],
                block_size=BLOCK_SIZE,
                dcp_size=dcp_size,
                dcp_rank=dcp_rank,
                dcp_interleave=cp_kv_cache_interleave_size,
                compact_valid_to_front=True,
                tile_size=self.hierarchical_tile_size,
            )
        self.calls += 1
        return self._binding.out, self._binding.counts


@contextmanager
def install_converter(module: Any, adapter: WorkspaceAdapter) -> Iterator[None]:
    """Swap the adapter in at the imported call site for the duration of a block.

    Restores the original callable on exit, including on exception. No upstream
    file is modified.
    """

    attribute = "triton_filter_and_convert_dcp_index"
    if not hasattr(module, attribute):
        raise AdapterError(
            f"module has no {attribute}; this is not the expected call site"
        )
    original = getattr(module, attribute)
    setattr(module, attribute, adapter)
    try:
        yield
    finally:
        setattr(module, attribute, original)
