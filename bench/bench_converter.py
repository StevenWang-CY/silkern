"""Time the three converters against each other on your own device.

This reproduces the shape of the converter-segment comparison in
``evidence/`` -- the deterministic row-wide converter costing *less* than the
nondeterministic atomic one -- on whatever hardware you have.

    python -m bench.bench_converter --width 2048 --batch 8 --dcp-size 2

What it measures
----------------

The converter in isolation, as a captured CUDA graph replayed many times, with
CUDA events around the replay block. Isolation is the point: it is the only way
to attribute a difference to the converter rather than to whatever the
downstream attention kernel happens to do with a differently ordered index.

``--with-atomic`` additionally times a reference atomic-reservation converter
included here purely as a baseline (``bench/atomic_baseline.py``). It reserves
one output segment per tile with ``atomic_add`` after a tile-local scan, which
is what makes its output order run-dependent. It is a baseline, not a
production kernel, and nothing in ``silkern`` depends on it.

Caveats worth stating before you quote a number
-----------------------------------------------

* One host, one device, one process, no statistical framing. This is a
  diagnostic, not an experiment. Real conclusions need fresh processes as
  inferential units and a prefixed decision rule.
* Converter time is a small fraction of a decode step. A converter that wins
  here can still lose end to end, and vice versa -- see ``docs/dispatch.md``.
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys

from silkern import (
    DEFAULT_TILE_SIZE,
    localize_hierarchical,
    localize_rowwise,
    workspace_shapes,
)


def _case(width: int, batch: int, block_size: int, dcp_size: int, seed: int):
    rng = random.Random(seed)
    table_width = max(4, (width * dcp_size) // block_size + 2)
    requests = 4
    req_ids = [(row * 3 + 1) % requests for row in range(batch)]
    # Fragmented, non-monotonic page table -- the realistic case.
    block_table = [
        rng.sample(range(4096), table_width) for _ in range(requests)
    ]
    limit = block_size * table_width * dcp_size
    rows = [
        sorted(rng.sample(range(limit), width)) for _ in range(batch)
    ]
    return req_ids, block_table, rows


def _time_graph(torch, launch, *, replays: int, blocks: int) -> list[float]:
    """Return per-replay microseconds for each timed block."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(5):
            launch()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()

    for _ in range(3):  # warm the replay path itself
        graph.replay()
    torch.cuda.synchronize()

    out: list[float] = []
    for _ in range(blocks):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        out.append(start.elapsed_time(end) * 1000.0 / replays)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--width", type=int, default=2048, help="top-k selection width")
    parser.add_argument("--batch", type=int, default=8, help="selection rows")
    parser.add_argument("--block-size", type=int, default=64, help="KV page size")
    parser.add_argument("--dcp-size", type=int, default=2)
    parser.add_argument("--dcp-rank", type=int, default=0)
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE)
    parser.add_argument("--replays", type=int, default=200)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--with-atomic",
        action="store_true",
        help="also time the atomic-reservation baseline",
    )
    args = parser.parse_args(argv)

    try:
        import torch
    except ModuleNotFoundError:
        print("torch is not installed; install silkern[gpu]", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("no CUDA device available", file=sys.stderr)
        return 2

    req_ids, block_table, rows = _case(
        args.width, args.batch, args.block_size, args.dcp_size, args.seed
    )
    dev = "cuda"
    req = torch.tensor(req_ids, dtype=torch.int32, device=dev)
    table = torch.tensor(block_table, dtype=torch.int32, device=dev)
    tokens = torch.tensor(rows, dtype=torch.int32, device=dev)
    out = torch.empty_like(tokens)
    counts = torch.empty(args.batch, dtype=torch.int32, device=dev)
    ws = {
        name: torch.empty(shape, dtype=torch.int32, device=dev)
        for name, shape in workspace_shapes(
            args.batch, args.width, tile_size=args.tile_size
        ).items()
    }

    common = dict(
        block_size=args.block_size,
        dcp_size=args.dcp_size,
        dcp_rank=args.dcp_rank,
    )

    arms: dict[str, object] = {
        "row_stable": lambda: localize_rowwise(
            req, table, tokens, out, counts, **common
        ),
        "hierarchical_stable": lambda: localize_hierarchical(
            req,
            table,
            tokens,
            out,
            counts,
            ws["mapped"],
            ws["local_positions"],
            ws["tile_counts"],
            ws["tile_offsets"],
            tile_size=args.tile_size,
            **common,
        ),
    }
    if args.with_atomic:
        from bench.atomic_baseline import launch_atomic_baseline

        scratch = torch.empty(args.batch, dtype=torch.int32, device=dev)
        arms["atomic_baseline"] = lambda: launch_atomic_baseline(
            req, table, tokens, out, counts, scratch, **common
        )

    print(f"device      : {torch.cuda.get_device_name(0)}")
    print(
        f"geometry    : width={args.width} batch={args.batch} "
        f"block_size={args.block_size} dcp={args.dcp_rank}/{args.dcp_size} "
        f"tile={args.tile_size}"
    )
    print(f"schedule    : {args.blocks} blocks x {args.replays} graph replays\n")

    results: dict[str, float] = {}
    for name, launch in arms.items():
        samples = _time_graph(
            torch, launch, replays=args.replays, blocks=args.blocks
        )
        median = statistics.median(samples)
        results[name] = median
        spread = max(samples) - min(samples)
        print(f"  {name:22s} {median:8.2f} us/replay   (block spread {spread:.2f} us)")

    if "atomic_baseline" in results:
        base = results["atomic_baseline"]
        print()
        for name, value in results.items():
            if name == "atomic_baseline":
                continue
            print(f"  {name:22s} {value / base:.3f} x atomic_baseline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
