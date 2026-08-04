from __future__ import annotations

import random

import pytest

from silkern import (
    DEFAULT_TILE_SIZE,
    MAX_ROW_WIDTH,
    LocalizationError,
    localize_reference,
    workspace_shapes,
)


def test_reference_routes_requests_and_nonidentity_blocks() -> None:
    rows = [
        [0, 1, 2, 3, 8, 9, -1, 99],
        [4, 5, 6, 7, 12, 13, 14, 15],
    ]
    out, counts = localize_reference(
        [1, 0],
        [[10, 30], [20, 40]],
        rows,
        block_size=4,
        dcp_size=2,
        dcp_rank=0,
    )
    assert out == [
        [80, 81, 160, -1, -1, -1, -1, -1],
        [42, 43, 122, 123, -1, -1, -1, -1],
    ]
    assert counts == [3, 4]


@pytest.mark.parametrize(
    ("rank", "expected"),
    [
        (0, [8, 9, 10, 11, -1, -1, -1, -1]),
        (1, [8, 9, 10, 11, -1, -1, -1, -1]),
    ],
)
def test_reference_honors_interleave(rank: int, expected: list[int]) -> None:
    out, counts = localize_reference(
        [0],
        [[2, 3]],
        [list(range(8))],
        block_size=4,
        dcp_size=2,
        dcp_rank=rank,
        dcp_interleave=2,
    )
    assert out == [expected]
    assert counts == [4]


def test_reference_preserves_columns_for_dcp_size_one() -> None:
    out, counts = localize_reference(
        [0],
        [[7]],
        [[0, -1, 3, 4]],
        block_size=4,
        dcp_size=1,
        dcp_rank=0,
        compact_valid_to_front=True,
    )
    assert out == [[28, -1, 31, -1]]
    assert counts == [2]


def test_reference_can_disable_front_compaction() -> None:
    out, counts = localize_reference(
        [0],
        [[5]],
        [[1, 0, 3, 2]],
        block_size=4,
        dcp_size=2,
        dcp_rank=0,
        compact_valid_to_front=False,
    )
    assert out == [[-1, 20, -1, 21]]
    assert counts == [2]


def test_reference_masks_owned_out_of_range_blocks() -> None:
    out, counts = localize_reference(
        [0],
        [[0]],
        [[0, 2, 8]],
        block_size=4,
        dcp_size=2,
        dcp_rank=0,
    )
    assert out == [[0, 1, -1]]
    assert counts == [2]


def test_reference_uses_in_range_block_table_values_verbatim() -> None:
    out, counts = localize_reference(
        [0],
        [[-1]],
        [[0, 2]],
        block_size=4,
        dcp_size=2,
        dcp_rank=0,
    )
    assert out == [[-4, -3]]
    assert counts == [2]


def test_reference_randomized_contract_matrix() -> None:
    rng = random.Random(20260728)
    block_size = 8
    block_table = [
        [rng.randrange(0, 200) for _ in range(7)]
        for _ in range(3)
    ]
    req_ids = [2, 0, 1, 2]

    for dcp_size in (1, 2, 4):
        global_limit = block_size * len(block_table[0]) * dcp_size
        rows = [
            [
                rng.choice((-3, -1))
                if rng.random() < 0.15
                else rng.randrange(global_limit + block_size * dcp_size)
                for _ in range(37)
            ]
            for _ in req_ids
        ]
        for interleave in (1, 2, 4):
            for rank in range(dcp_size):
                actual_rows, actual_counts = localize_reference(
                    req_ids,
                    block_table,
                    rows,
                    block_size=block_size,
                    dcp_size=dcp_size,
                    dcp_rank=rank,
                    dcp_interleave=interleave,
                )

                expected_rows = []
                expected_counts = []
                for request, row in zip(req_ids, rows, strict=True):
                    in_place = []
                    valid = []
                    for token in row:
                        physical = -1
                        keep = False
                        if token >= 0:
                            group_index, lane_in_group = divmod(token, interleave)
                            local_group, owner = divmod(group_index, dcp_size)
                            if owner == rank:
                                local_token = local_group * interleave + lane_in_group
                                logical_block, offset = divmod(
                                    local_token, block_size
                                )
                                if logical_block < len(block_table[request]):
                                    physical = (
                                        block_table[request][logical_block] * block_size
                                        + offset
                                    )
                                    keep = True
                        in_place.append(physical)
                        if keep:
                            valid.append(physical)
                    expected_counts.append(len(valid))
                    expected_rows.append(
                        in_place
                        if dcp_size == 1
                        else valid + [-1] * (len(row) - len(valid))
                    )

                assert actual_rows == expected_rows
                assert actual_counts == expected_counts


@pytest.mark.parametrize(
    "kwargs",
    [
        {"block_size": 0, "dcp_size": 2, "dcp_rank": 0},
        {"block_size": True, "dcp_size": 2, "dcp_rank": 0},
        {"block_size": 4, "dcp_size": 0, "dcp_rank": 0},
        {"block_size": 4, "dcp_size": 2.0, "dcp_rank": 0},
        {"block_size": 4, "dcp_size": 2, "dcp_rank": 2},
        {
            "block_size": 4,
            "dcp_size": 2,
            "dcp_rank": 0,
            "dcp_interleave": 3,
        },
        {
            "block_size": 4,
            "dcp_size": 2,
            "dcp_rank": 0,
            "compact_valid_to_front": 1,
        },
    ],
)
def test_reference_rejects_invalid_configuration(kwargs: dict) -> None:
    with pytest.raises(LocalizationError):
        localize_reference([0], [[0]], [[0]], **kwargs)


def test_reference_rejects_bad_request_routing() -> None:
    with pytest.raises(LocalizationError, match="request id"):
        localize_reference(
            [1],
            [[0]],
            [[0]],
            block_size=4,
            dcp_size=2,
            dcp_rank=0,
        )


def test_reference_rejects_row_count_mismatch() -> None:
    with pytest.raises(LocalizationError, match="one request id per row"):
        localize_reference(
            [],
            [[0]],
            [[0]],
            block_size=4,
            dcp_size=2,
            dcp_rank=0,
        )


@pytest.mark.parametrize(
    ("block_table", "rows", "message"),
    [
        ([[]], [[0]], "block_table"),
        ([[0], [1, 2]], [[0]], "block_table"),
        ([[0]], [], "rows"),
        ([[0]], [[]], "rows"),
        ([[0]], [[0], [0, 1]], "rows"),
    ],
)
def test_reference_rejects_nonrectangular_inputs(
    block_table: list[list[int]],
    rows: list[list[int]],
    message: str,
) -> None:
    req_ids = [0] * len(rows)
    with pytest.raises(LocalizationError, match=message):
        localize_reference(
            req_ids,
            block_table,
            rows,
            block_size=4,
            dcp_size=2,
            dcp_rank=0,
        )


def test_workspace_shapes_cover_partial_tile() -> None:
    assert workspace_shapes(3, 129) == {
        "mapped": (3, 129),
        "local_positions": (3, 129),
        "tile_counts": (3, 2),
        "tile_offsets": (3, 2),
    }
    assert workspace_shapes(2, 257, tile_size=64) == {
        "mapped": (2, 257),
        "local_positions": (2, 257),
        "tile_counts": (2, 5),
        "tile_offsets": (2, 5),
    }
    assert DEFAULT_TILE_SIZE == 128


@pytest.mark.parametrize(
    ("batch", "width", "tile_size"),
    [
        (0, 1, 128),
        (1, 0, 128),
        (True, 1, 128),
        (1, 1.5, 128),
        (1, MAX_ROW_WIDTH + 1, 128),
        (1, 128, 32),
    ],
)
def test_workspace_shapes_reject_invalid_configuration(
    batch: int,
    width: int,
    tile_size: int,
) -> None:
    with pytest.raises(LocalizationError):
        workspace_shapes(batch, width, tile_size=tile_size)
