from __future__ import annotations

import random
from collections.abc import Callable

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from silkern import (  # noqa: E402
    LocalizationError,
    localize_hierarchical,
    localize_reference,
    localize_rowwise,
    workspace_shapes,
)

pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)


def _case(
    *,
    width: int,
    batch: int,
    block_size: int,
    dcp_size: int,
    seed: int,
) -> tuple[list[int], list[list[int]], list[list[int]]]:
    rng = random.Random(seed)
    table_width = 17
    requests = 4
    req_ids = [(row * 3 + 1) % requests for row in range(batch)]
    block_table = [
        [rng.randrange(-2, 4096) for _ in range(table_width)]
        for _ in range(requests)
    ]
    global_limit = block_size * table_width * dcp_size
    rows = []
    for _row in range(batch):
        values = []
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


def _cuda_int(values):
    return torch.tensor(values, dtype=torch.int32, device="cuda")


def _launch_both(
    req_ids: list[int],
    block_table: list[list[int]],
    rows: list[list[int]],
    *,
    block_size: int,
    dcp_size: int,
    dcp_rank: int,
    dcp_interleave: int,
    compact_valid_to_front: bool = True,
) -> tuple[tuple[list[list[int]], list[int]], tuple[list[list[int]], list[int]]]:
    req = _cuda_int(req_ids)
    table = _cuda_int(block_table)
    tokens = _cuda_int(rows)
    batch, width = tokens.shape

    row_out = torch.empty_like(tokens)
    row_counts = torch.empty(batch, dtype=torch.int32, device="cuda")
    localize_rowwise(
        req,
        table,
        tokens,
        row_out,
        row_counts,
        block_size=block_size,
        dcp_size=dcp_size,
        dcp_rank=dcp_rank,
        dcp_interleave=dcp_interleave,
        compact_valid_to_front=compact_valid_to_front,
    )

    hierarchical_out = torch.empty_like(tokens)
    hierarchical_counts = torch.empty(batch, dtype=torch.int32, device="cuda")
    shapes = workspace_shapes(batch, width)
    workspace = {
        name: torch.empty(shape, dtype=torch.int32, device="cuda")
        for name, shape in shapes.items()
    }
    localize_hierarchical(
        req,
        table,
        tokens,
        hierarchical_out,
        hierarchical_counts,
        workspace["mapped"],
        workspace["local_positions"],
        workspace["tile_counts"],
        workspace["tile_offsets"],
        block_size=block_size,
        dcp_size=dcp_size,
        dcp_rank=dcp_rank,
        dcp_interleave=dcp_interleave,
        compact_valid_to_front=compact_valid_to_front,
    )
    torch.cuda.synchronize()
    return (
        (row_out.cpu().tolist(), row_counts.cpu().tolist()),
        (hierarchical_out.cpu().tolist(), hierarchical_counts.cpu().tolist()),
    )


@pytest.mark.parametrize("width", [1, 63, 128, 129, 513, 1024, 2048, 4096])
@pytest.mark.parametrize(
    ("dcp_size", "dcp_rank", "interleave"),
    [(1, 0, 1), (2, 1, 2), (4, 3, 4), (8, 7, 1)],
)
def test_exact_gpu_variants_match_cpu_oracle(
    width: int,
    dcp_size: int,
    dcp_rank: int,
    interleave: int,
) -> None:
    case = _case(
        width=width,
        batch=3,
        block_size=64,
        dcp_size=dcp_size,
        seed=650000 + width * 31 + dcp_size * 7 + dcp_rank,
    )
    expected = localize_reference(
        *case,
        block_size=64,
        dcp_size=dcp_size,
        dcp_rank=dcp_rank,
        dcp_interleave=interleave,
    )
    row, hierarchical = _launch_both(
        *case,
        block_size=64,
        dcp_size=dcp_size,
        dcp_rank=dcp_rank,
        dcp_interleave=interleave,
    )
    assert row == expected
    assert hierarchical == expected


def test_exact_gpu_every_rank_and_grouped_interleave() -> None:
    for dcp_size in (1, 2, 4, 8):
        case = _case(
            width=257,
            batch=4,
            block_size=64,
            dcp_size=dcp_size,
            seed=651000 + dcp_size,
        )
        for interleave in (1, 2, 4):
            for rank in range(dcp_size):
                expected = localize_reference(
                    *case,
                    block_size=64,
                    dcp_size=dcp_size,
                    dcp_rank=rank,
                    dcp_interleave=interleave,
                )
                row, hierarchical = _launch_both(
                    *case,
                    block_size=64,
                    dcp_size=dcp_size,
                    dcp_rank=rank,
                    dcp_interleave=interleave,
                )
                assert row == expected
                assert hierarchical == expected


def test_exact_gpu_noncompacting_column_semantics() -> None:
    case = _case(width=513, batch=3, block_size=32, dcp_size=4, seed=652001)
    for dcp_size, rank, interleave in ((1, 0, 4), (4, 2, 2)):
        expected = localize_reference(
            *case,
            block_size=32,
            dcp_size=dcp_size,
            dcp_rank=rank,
            dcp_interleave=interleave,
            compact_valid_to_front=False,
        )
        row, hierarchical = _launch_both(
            *case,
            block_size=32,
            dcp_size=dcp_size,
            dcp_rank=rank,
            dcp_interleave=interleave,
            compact_valid_to_front=False,
        )
        assert row == expected
        assert hierarchical == expected


def _guarded_cuda(
    shape: tuple[int, ...],
    *,
    guard: int = 32,
    sentinel: int = 0x5A5A5A5A,
):
    elements = 1
    for extent in shape:
        elements *= extent
    backing = torch.full(
        (elements + 2 * guard,),
        sentinel,
        dtype=torch.int32,
        device="cuda",
    )
    return backing, backing[guard : guard + elements].view(shape)


def test_hierarchical_gpu_canaries_and_input_immutability() -> None:
    case = _case(width=513, batch=3, block_size=64, dcp_size=4, seed=653001)
    expected = localize_reference(
        *case,
        block_size=64,
        dcp_size=4,
        dcp_rank=3,
        dcp_interleave=4,
    )
    guard = 32
    sentinel = 0x5A5A5A5A
    guarded = {}

    def make(name: str, values) -> object:
        tensor = torch.tensor(values, dtype=torch.int32)
        backing, view = _guarded_cuda(tuple(tensor.shape), guard=guard, sentinel=sentinel)
        view.copy_(tensor)
        guarded[name] = (backing, view.clone())
        return view

    req = make("req", case[0])
    table = make("table", case[1])
    tokens = make("tokens", case[2])
    out_backing, out = _guarded_cuda(tuple(tokens.shape), guard=guard, sentinel=sentinel)
    counts_backing, counts = _guarded_cuda((tokens.shape[0],), guard=guard, sentinel=sentinel)
    output_guarded = [out_backing, counts_backing]
    shapes = workspace_shapes(*tokens.shape)
    workspace = {}
    for name, shape in shapes.items():
        backing, view = _guarded_cuda(shape, guard=guard, sentinel=sentinel)
        output_guarded.append(backing)
        workspace[name] = view

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
        block_size=64,
        dcp_size=4,
        dcp_rank=3,
        dcp_interleave=4,
    )
    torch.cuda.synchronize()
    assert out.cpu().tolist() == expected[0]
    assert counts.cpu().tolist() == expected[1]
    for backing, original in guarded.values():
        view_elements = original.numel()
        assert torch.all(backing[:guard] == sentinel)
        assert torch.all(backing[guard + view_elements :] == sentinel)
        assert torch.equal(
            backing[guard : guard + view_elements].view(original.shape),
            original,
        )
    for backing in output_guarded:
        view_elements = backing.numel() - 2 * guard
        assert torch.all(backing[:guard] == sentinel)
        assert torch.all(backing[guard + view_elements :] == sentinel)


def _capture(op: Callable[[], None]):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            op()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op()
    torch.cuda.synchronize()
    return graph


def test_exact_gpu_cuda_graph_replay_and_allocation_stability() -> None:
    case = _case(width=2048, batch=8, block_size=64, dcp_size=4, seed=654001)
    expected = localize_reference(
        *case,
        block_size=64,
        dcp_size=4,
        dcp_rank=2,
        dcp_interleave=2,
    )
    req, table, tokens = (_cuda_int(values) for values in case)
    out = torch.empty_like(tokens)
    counts = torch.empty(tokens.shape[0], dtype=torch.int32, device="cuda")
    shapes = workspace_shapes(*tokens.shape)
    workspace = {
        name: torch.empty(shape, dtype=torch.int32, device="cuda")
        for name, shape in shapes.items()
    }

    def op() -> None:
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
            block_size=64,
            dcp_size=4,
            dcp_rank=2,
            dcp_interleave=2,
        )

    pointers = [tensor.data_ptr() for tensor in (req, table, tokens, out, counts)]
    pointers.extend(tensor.data_ptr() for tensor in workspace.values())
    graph = _capture(op)
    allocated_before = torch.cuda.memory_allocated()
    for _ in range(100):
        graph.replay()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == allocated_before
    assert pointers == [tensor.data_ptr() for tensor in (req, table, tokens, out, counts)] + [
        tensor.data_ptr() for tensor in workspace.values()
    ]
    assert out.cpu().tolist() == expected[0]
    assert counts.cpu().tolist() == expected[1]


def test_hierarchical_gpu_rejects_workspace_shape_and_alias() -> None:
    req = _cuda_int([0])
    table = _cuda_int([[0, 1]])
    tokens = _cuda_int([[0] * 129])
    out = torch.empty_like(tokens)
    counts = torch.empty(1, dtype=torch.int32, device="cuda")
    shapes = workspace_shapes(1, 129)
    workspace = {
        name: torch.empty(shape, dtype=torch.int32, device="cuda")
        for name, shape in shapes.items()
    }
    with pytest.raises(LocalizationError, match="tile_counts workspace shape"):
        localize_hierarchical(
            req,
            table,
            tokens,
            out,
            counts,
            workspace["mapped"],
            workspace["local_positions"],
            torch.empty((1, 1), dtype=torch.int32, device="cuda"),
            workspace["tile_offsets"],
            block_size=64,
            dcp_size=2,
            dcp_rank=0,
        )
    with pytest.raises(LocalizationError, match="distinct storage"):
        localize_hierarchical(
            req,
            table,
            tokens,
            out,
            counts,
            out,
            workspace["local_positions"],
            workspace["tile_counts"],
            workspace["tile_offsets"],
            block_size=64,
            dcp_size=2,
            dcp_rank=0,
        )
