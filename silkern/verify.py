"""One call that tries to prove the kernels wrong on *your* stack.

Everything in this package rests on one claim: the GPU implementations
reproduce :func:`silkern.contract.localize_reference` exactly, and do so at fixed
addresses without allocating. That claim is machine-checkable, so you should
not take it on trust from a README written on someone else's hardware.

    >>> import silkern
    >>> report = silkern.conformance()      # doctest: +SKIP
    >>> report.ok                         # doctest: +SKIP
    True

:func:`conformance` sweeps a randomized geometry matrix and runs five checks
per cell:

``oracle``
    Every localized array and count equals the pure-Python oracle, elementwise.
``order``
    The valid prefix equals the input row filtered by rank ownership, in input
    order. Set equality is not enough -- an atomic converter passes set
    equality and still scrambles the order. This is the check that matters.
``determinism``
    Repeated launches on identical input produce a bytewise-identical output
    buffer. A converter whose output depends on tile-reservation race order
    fails here.
``replay``
    The launch is captured in a CUDA graph and replayed; every buffer pointer
    is unchanged and ``torch.cuda.memory_allocated()`` does not grow.
``immutability``
    Inputs are unmodified, and guard bytes placed around every output buffer
    are still zero -- i.e. nothing was written out of bounds.

A failing cell is reported, not raised. The report tells you which geometry
failed and which check, so a narrowed geometry is an actionable result rather
than a stack trace.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field

from silkern.contract import DEFAULT_TILE_SIZE, localize_reference
from silkern.errors import LocalizationError
from silkern.kernels import localize_hierarchical, localize_rowwise
from silkern.workspace import workspace_shapes

CHECKS = ("oracle", "order", "determinism", "replay", "immutability")

#: Geometry matrix used when ``conformance()`` is called with no arguments.
#: Chosen to cross the interesting boundaries: width below/at/above a tile,
#: non-power-of-two widths, both qualified page sizes, DCP degrees 1/2/4/8, the
#: last rank of each degree (most likely to expose an off-by-one), and grouped
#: interleave.
DEFAULT_MATRIX: tuple[dict[str, int], ...] = (
    {"width": 1, "block_size": 64, "dcp_size": 1, "dcp_rank": 0, "dcp_interleave": 1},
    {"width": 63, "block_size": 32, "dcp_size": 2, "dcp_rank": 1, "dcp_interleave": 1},
    {"width": 128, "block_size": 64, "dcp_size": 2, "dcp_rank": 0, "dcp_interleave": 1},
    {"width": 129, "block_size": 64, "dcp_size": 4, "dcp_rank": 3, "dcp_interleave": 1},
    {"width": 513, "block_size": 32, "dcp_size": 4, "dcp_rank": 1, "dcp_interleave": 4},
    {"width": 1024, "block_size": 64, "dcp_size": 8, "dcp_rank": 7, "dcp_interleave": 1},
    {"width": 2048, "block_size": 64, "dcp_size": 2, "dcp_rank": 1, "dcp_interleave": 2},
    {"width": 2048, "block_size": 64, "dcp_size": 4, "dcp_rank": 2, "dcp_interleave": 1},
    {"width": 4096, "block_size": 64, "dcp_size": 2, "dcp_rank": 0, "dcp_interleave": 1},
)

GUARD_ELEMENTS = 64
REPLAY_COUNT = 32
DETERMINISM_REPEATS = 8


@dataclass
class CellReport:
    """Outcome of one (arm, geometry) cell."""

    arm: str
    geometry: dict[str, int]
    checks: dict[str, bool] = field(default_factory=dict)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and all(self.checks.values())

    def failures(self) -> list[str]:
        if self.error is not None:
            return [f"error: {self.error}"]
        return [name for name, passed in self.checks.items() if not passed]


@dataclass
class ConformanceReport:
    """Aggregate outcome. Truthy iff every cell passed every check."""

    cells: list[CellReport] = field(default_factory=list)
    device: str = "unknown"
    skipped: str | None = None

    @property
    def ok(self) -> bool:
        return self.skipped is None and bool(self.cells) and all(c.ok for c in self.cells)

    def __bool__(self) -> bool:
        return self.ok

    def summary(self) -> str:
        if self.skipped is not None:
            return f"conformance skipped: {self.skipped}"
        passed = sum(1 for c in self.cells if c.ok)
        head = (
            f"silkern conformance on {self.device}: "
            f"{passed}/{len(self.cells)} cells passed "
            f"({', '.join(CHECKS)})"
        )
        if passed == len(self.cells):
            return head
        lines = [head, ""]
        for cell in self.cells:
            if cell.ok:
                continue
            geometry = " ".join(f"{k}={v}" for k, v in cell.geometry.items())
            lines.append(f"  FAIL {cell.arm:22s} {geometry}")
            for failure in cell.failures():
                lines.append(f"       - {failure}")
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.summary()


def _random_case(
    *,
    width: int,
    batch: int,
    block_size: int,
    dcp_size: int,
    seed: int,
) -> tuple[list[int], list[list[int]], list[list[int]]]:
    """Build one adversarial case.

    Deliberately includes negative tokens, tokens past the end of the page
    table, a fragmented (non-identity, non-monotonic) page table, several
    requests sharing one batch, and the two extreme in-range positions.
    """
    rng = random.Random(seed)
    table_width = 17
    requests = 4
    req_ids = [(row * 3 + 1) % requests for row in range(batch)]
    block_table = [
        [rng.randrange(-2, 4096) for _ in range(table_width)] for _ in range(requests)
    ]
    global_limit = block_size * table_width * dcp_size
    rows: list[list[int]] = []
    for _row in range(batch):
        values: list[int] = []
        for column in range(width):
            draw = rng.random()
            if draw < 0.12:
                values.append(rng.choice((-9, -3, -1)))
            elif draw < 0.20:
                values.append(global_limit + rng.randrange(block_size * dcp_size + 1))
            elif column == 0:
                values.append(0)
            elif column == width - 1:
                values.append(max(global_limit - 1, 0))
            else:
                values.append(rng.randrange(max(global_limit, 1)))
        rows.append(values)
    return req_ids, block_table, rows


def _expected_order(
    row: Sequence[int],
    table_row: Sequence[int],
    *,
    block_size: int,
    dcp_size: int,
    dcp_rank: int,
    dcp_interleave: int,
) -> list[int]:
    """The prefix a *stable* converter must produce, derived independently.

    Computed as a filter over the input row rather than reusing the oracle's
    control flow, so agreement is not an artifact of shared code.
    """
    kept: list[int] = []
    for raw in row:
        token = int(raw)
        if token < 0:
            continue
        if (token // dcp_interleave) % dcp_size != dcp_rank:
            continue
        local = (token // (dcp_size * dcp_interleave)) * dcp_interleave + (
            token % dcp_interleave
        )
        logical_block, offset = divmod(local, block_size)
        if logical_block >= len(table_row):
            continue
        kept.append(int(table_row[logical_block]) * block_size + offset)
    return kept


def _guarded(torch, shape, device):
    """An int32 tensor with zeroed guard elements on both sides.

    Returns ``(view, storage)``. Out-of-bounds writes land in the guards, so a
    kernel that overruns its row is caught even when the in-bounds values
    happen to be correct.
    """
    total = 1
    for dim in shape:
        total *= dim
    storage = torch.zeros(total + 2 * GUARD_ELEMENTS, dtype=torch.int32, device=device)
    view = storage[GUARD_ELEMENTS : GUARD_ELEMENTS + total].view(*shape)
    return view, storage


def _guards_clean(storage) -> bool:
    return bool(
        (storage[:GUARD_ELEMENTS] == 0).all()
        and (storage[-GUARD_ELEMENTS:] == 0).all()
    )


def _run_cell(
    torch,
    arm: str,
    geometry: dict[str, int],
    *,
    batch: int,
    seed: int,
    tile_size: int,
    device: str,
) -> CellReport:
    report = CellReport(arm=arm, geometry=dict(geometry))
    width = geometry["width"]
    block_size = geometry["block_size"]
    dcp_size = geometry["dcp_size"]
    dcp_rank = geometry["dcp_rank"]
    dcp_interleave = geometry["dcp_interleave"]

    req_ids, block_table, rows = _random_case(
        width=width,
        batch=batch,
        block_size=block_size,
        dcp_size=dcp_size,
        seed=seed,
    )

    req = torch.tensor(req_ids, dtype=torch.int32, device=device)
    table = torch.tensor(block_table, dtype=torch.int32, device=device)
    tokens = torch.tensor(rows, dtype=torch.int32, device=device)
    tokens_before = tokens.clone()
    table_before = table.clone()

    out, out_storage = _guarded(torch, (batch, width), device)
    counts, counts_storage = _guarded(torch, (batch,), device)

    workspace: dict[str, object] = {}
    guards = [out_storage, counts_storage]
    if arm == "hierarchical_stable":
        for name, shape in workspace_shapes(batch, width, tile_size=tile_size).items():
            view, storage = _guarded(torch, shape, device)
            workspace[name] = view
            guards.append(storage)

    def launch() -> None:
        if arm == "row_stable":
            localize_rowwise(
                req,
                table,
                tokens,
                out,
                counts,
                block_size=block_size,
                dcp_size=dcp_size,
                dcp_rank=dcp_rank,
                dcp_interleave=dcp_interleave,
            )
        else:
            localize_hierarchical(
                req,
                table,
                tokens,
                out,
                counts,
                workspace["mapped"],
                workspace["local_positions"],
                workspace["tile_counts"],
                workspace["tile_offsets"],
                block_size=block_size,
                dcp_size=dcp_size,
                dcp_rank=dcp_rank,
                dcp_interleave=dcp_interleave,
                tile_size=tile_size,
            )

    try:
        launch()
        torch.cuda.synchronize()
        observed_out = out.cpu().tolist()
        observed_counts = counts.cpu().tolist()

        expected_out, expected_counts = localize_reference(
            req_ids,
            block_table,
            rows,
            block_size=block_size,
            dcp_size=dcp_size,
            dcp_rank=dcp_rank,
            dcp_interleave=dcp_interleave,
        )
        report.checks["oracle"] = (
            observed_out == expected_out and observed_counts == expected_counts
        )

        # dcp_size == 1 bypasses compaction by contract, so there is no prefix
        # to order-check; the oracle check above already covers it.
        if dcp_size == 1:
            report.checks["order"] = observed_out == expected_out
        else:
            order_ok = True
            for row_id, row in enumerate(rows):
                want = _expected_order(
                    row,
                    block_table[req_ids[row_id]],
                    block_size=block_size,
                    dcp_size=dcp_size,
                    dcp_rank=dcp_rank,
                    dcp_interleave=dcp_interleave,
                )
                got = observed_out[row_id][: observed_counts[row_id]]
                tail = observed_out[row_id][observed_counts[row_id] :]
                if got != want or any(value != -1 for value in tail):
                    order_ok = False
                    break
            report.checks["order"] = order_ok

        first = out.clone()
        stable = True
        for _ in range(DETERMINISM_REPEATS):
            launch()
            torch.cuda.synchronize()
            if not bool(torch.equal(out, first)):
                stable = False
                break
        report.checks["determinism"] = stable

        pointers = [t.data_ptr() for t in (req, table, tokens, out, counts)]
        pointers += [t.data_ptr() for t in workspace.values()]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                launch()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()
        allocated_before = torch.cuda.memory_allocated()
        for _ in range(REPLAY_COUNT):
            graph.replay()
        torch.cuda.synchronize()
        after_pointers = [t.data_ptr() for t in (req, table, tokens, out, counts)]
        after_pointers += [t.data_ptr() for t in workspace.values()]
        report.checks["replay"] = (
            torch.cuda.memory_allocated() == allocated_before
            and after_pointers == pointers
            and out.cpu().tolist() == expected_out
        )

        report.checks["immutability"] = (
            bool(torch.equal(tokens, tokens_before))
            and bool(torch.equal(table, table_before))
            and all(_guards_clean(storage) for storage in guards)
        )
    except LocalizationError as exc:
        report.error = f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # pragma: no cover - surfaced, not swallowed
        report.error = f"{type(exc).__name__}: {exc}"

    for name in CHECKS:
        report.checks.setdefault(name, False)
    return report


def conformance(
    matrix: Sequence[dict[str, int]] | None = None,
    *,
    arms: Sequence[str] = ("row_stable", "hierarchical_stable"),
    batch: int = 5,
    tile_size: int = DEFAULT_TILE_SIZE,
    seed: int = 20260803,
) -> ConformanceReport:
    """Run the conformance matrix on the local device.

    Args:
        matrix: Geometry dicts with ``width``, ``block_size``, ``dcp_size``,
            ``dcp_rank``, ``dcp_interleave``. Defaults to :data:`DEFAULT_MATRIX`.
            Pass your own deployment geometry before trusting the kernels on it.
        arms: Which implementations to check.
        batch: Selection rows per cell. Rows share a small pool of request ids,
            so request routing is exercised, not bypassed.
        tile_size: Tile size for the hierarchical arm.
        seed: Base seed; each cell derives its own, so runs are reproducible.

    Returns:
        A :class:`ConformanceReport`. It is falsy if anything failed, and
        ``report.summary()`` names the geometry and the check. Missing CUDA or
        Triton yields a skipped (falsy) report rather than an exception, so this
        is safe to call unconditionally in CI.
    """
    report = ConformanceReport()
    try:
        import torch
    except ModuleNotFoundError:
        report.skipped = "torch is not installed"
        return report
    try:
        import triton  # noqa: F401
    except ModuleNotFoundError:
        report.skipped = "triton is not installed"
        return report
    if not torch.cuda.is_available():
        report.skipped = "no CUDA device is available"
        return report

    report.device = torch.cuda.get_device_name(0)
    cells = tuple(DEFAULT_MATRIX if matrix is None else matrix)
    for arm in arms:
        if arm not in ("row_stable", "hierarchical_stable"):
            raise LocalizationError(f"unknown arm: {arm}")
        for index, geometry in enumerate(cells):
            report.cells.append(
                _run_cell(
                    torch,
                    arm,
                    geometry,
                    batch=batch,
                    seed=seed + index,
                    tile_size=tile_size,
                    device="cuda",
                )
            )
    return report


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI
    """``python -m silkern`` -- exits nonzero if any cell fails."""
    import argparse

    parser = argparse.ArgumentParser(description="Run the silkern conformance matrix.")
    parser.add_argument("--batch", type=int, default=5)
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument(
        "--arm",
        action="append",
        choices=["row_stable", "hierarchical_stable"],
        help="restrict to one arm; repeatable",
    )
    args = parser.parse_args(argv)

    report = conformance(
        arms=tuple(args.arm) if args.arm else ("row_stable", "hierarchical_stable"),
        batch=args.batch,
        tile_size=args.tile_size,
        seed=args.seed,
    )
    print(report.summary())
    if report.skipped is not None:
        return 0
    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
