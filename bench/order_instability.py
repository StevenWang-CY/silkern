"""Watch the atomic converter change its mind, then watch silkern not.

    python -m bench.order_instability

Replays each converter N times on *byte-identical* input and counts how many
distinct output orders come back. The set and the count are identical every
time for both. Only the order differs -- which is precisely the part nothing
upstream promised, and precisely the part that decides your logits.

Expected shape of the result: the atomic baseline returns nearly as many
distinct orders as you give it replays; both silkern implementations return
exactly one.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys

from silkern import (
    DEFAULT_TILE_SIZE,
    localize_hierarchical,
    localize_rowwise,
    workspace_shapes,
)


def _digest(tensor) -> str:
    return hashlib.sha256(tensor.cpu().numpy().tobytes()).hexdigest()[:16]


def _set_digest(tensor, counts) -> str:
    """Order-insensitive digest of the valid prefix, per row."""
    hasher = hashlib.sha256()
    rows = tensor.cpu().tolist()
    for row, count in zip(rows, counts.cpu().tolist(), strict=True):
        hasher.update(repr(sorted(row[:count])).encode())
    return hasher.hexdigest()[:16]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--dcp-size", type=int, default=2)
    parser.add_argument("--dcp-rank", type=int, default=0)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args(argv)

    try:
        import torch
    except ModuleNotFoundError:
        print("torch is not installed; install silkern[gpu]", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("no CUDA device available", file=sys.stderr)
        return 2

    from bench.atomic_baseline import launch_atomic_baseline

    rng = random.Random(args.seed)
    table_width = max(4, (args.width * args.dcp_size) // args.block_size + 2)
    requests = 4
    req_ids = [(row * 3 + 1) % requests for row in range(args.batch)]
    block_table = [rng.sample(range(4096), table_width) for _ in range(requests)]
    limit = args.block_size * table_width * args.dcp_size
    rows = [sorted(rng.sample(range(limit), args.width)) for _ in range(args.batch)]

    dev = "cuda"
    req = torch.tensor(req_ids, dtype=torch.int32, device=dev)
    table = torch.tensor(block_table, dtype=torch.int32, device=dev)
    tokens = torch.tensor(rows, dtype=torch.int32, device=dev)
    out = torch.empty_like(tokens)
    counts = torch.empty(args.batch, dtype=torch.int32, device=dev)
    scratch = torch.empty(args.batch, dtype=torch.int32, device=dev)
    ws = {
        name: torch.empty(shape, dtype=torch.int32, device=dev)
        for name, shape in workspace_shapes(args.batch, args.width).items()
    }
    common = dict(
        block_size=args.block_size, dcp_size=args.dcp_size, dcp_rank=args.dcp_rank
    )

    arms = {
        "atomic_baseline": lambda: launch_atomic_baseline(
            req, table, tokens, out, counts, scratch, **common
        ),
        "silkern row_stable": lambda: localize_rowwise(
            req, table, tokens, out, counts, **common
        ),
        "silkern hierarchical": lambda: localize_hierarchical(
            req,
            table,
            tokens,
            out,
            counts,
            ws["mapped"],
            ws["local_positions"],
            ws["tile_counts"],
            ws["tile_offsets"],
            tile_size=DEFAULT_TILE_SIZE,
            **common,
        ),
    }

    print(f"device   : {torch.cuda.get_device_name(0)}")
    print(
        f"geometry : width={args.width} batch={args.batch} "
        f"dcp={args.dcp_rank}/{args.dcp_size}, {args.replays} replays of identical input\n"
    )
    print(f"  {'converter':22s} {'distinct orders':>16s} {'distinct sets':>15s} {'counts':>8s}")
    print(f"  {'-' * 22} {'-' * 16:>16s} {'-' * 15:>15s} {'-' * 8:>8s}")

    for name, launch in arms.items():
        order_digests: set[str] = set()
        set_digests: set[str] = set()
        count_digests: set[str] = set()
        for _ in range(args.replays):
            launch()
            torch.cuda.synchronize()
            order_digests.add(_digest(out))
            set_digests.add(_set_digest(out, counts))
            count_digests.add(_digest(counts))
        print(
            f"  {name:22s} {len(order_digests):>16d} {len(set_digests):>15d} "
            f"{len(count_digests):>8d}"
        )

    print(
        "\nSame set, same count, different order. That is the whole problem:\n"
        "nothing was violated, and the answer changed anyway."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
