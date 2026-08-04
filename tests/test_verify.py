"""CPU-side tests for the conformance harness.

The harness itself needs a GPU, but its reporting contract, its independent
order derivation, and its skip behavior are all checkable without one -- and
they are the parts most likely to lie about a failure.
"""

from __future__ import annotations

import silkern
from silkern.verify import (
    CHECKS,
    DEFAULT_MATRIX,
    CellReport,
    ConformanceReport,
    _expected_order,
    _random_case,
)


def test_conformance_skips_cleanly_without_a_device() -> None:
    report = silkern.conformance()
    # In CI there is no CUDA device; a skip must be falsy but not an exception,
    # and must say why.
    if report.skipped is not None:
        assert not report.ok
        assert not report
        assert report.skipped in {
            "torch is not installed",
            "triton is not installed",
            "no CUDA device is available",
        }
        assert "skipped" in report.summary()


def test_empty_report_is_not_ok() -> None:
    assert not ConformanceReport().ok


def test_cell_report_ok_requires_every_check() -> None:
    cell = CellReport(arm="row_stable", geometry={"width": 8})
    cell.checks = dict.fromkeys(CHECKS, True)
    assert cell.ok
    assert cell.failures() == []

    cell.checks["order"] = False
    assert not cell.ok
    assert cell.failures() == ["order"]

    cell.error = "boom"
    assert cell.failures() == ["error: boom"]


def test_failing_cell_summary_names_geometry_and_check() -> None:
    cell = CellReport(arm="row_stable", geometry={"width": 513, "dcp_size": 4})
    cell.checks = dict.fromkeys(CHECKS, True)
    cell.checks["determinism"] = False
    report = ConformanceReport(cells=[cell], device="Test Device")
    summary = report.summary()
    assert "FAIL" in summary
    assert "row_stable" in summary
    assert "width=513" in summary
    assert "determinism" in summary


def test_default_matrix_is_well_formed_and_crosses_boundaries() -> None:
    required = {"width", "block_size", "dcp_size", "dcp_rank", "dcp_interleave"}
    for geometry in DEFAULT_MATRIX:
        assert set(geometry) == required
        assert 0 <= geometry["dcp_rank"] < geometry["dcp_size"]
        assert geometry["width"] <= silkern.MAX_ROW_WIDTH
        assert geometry["block_size"] % geometry["dcp_interleave"] == 0
    widths = {g["width"] for g in DEFAULT_MATRIX}
    assert {1, silkern.MAX_ROW_WIDTH} <= widths          # both extremes
    assert any(w % 2 for w in widths)                  # a non-power-of-two width
    assert {g["dcp_size"] for g in DEFAULT_MATRIX} >= {1, 2, 4, 8}
    assert any(g["dcp_interleave"] > 1 for g in DEFAULT_MATRIX)
    assert any(
        g["dcp_rank"] == g["dcp_size"] - 1 for g in DEFAULT_MATRIX if g["dcp_size"] > 1
    )


def test_independent_order_derivation_agrees_with_the_oracle() -> None:
    """The harness's ``order`` check must not be a restatement of the oracle.

    They are written separately on purpose; this test proves they agree, so a
    disagreement in the field is a real signal rather than a harness bug.
    """
    for seed in range(12):
        for dcp_size, dcp_rank, interleave in [
            (2, 0, 1),
            (2, 1, 2),
            (4, 3, 1),
            (4, 1, 4),
            (8, 7, 1),
        ]:
            req_ids, block_table, rows = _random_case(
                width=131,
                batch=3,
                block_size=64,
                dcp_size=dcp_size,
                seed=seed,
            )
            expected_out, expected_counts = silkern.localize_reference(
                req_ids,
                block_table,
                rows,
                block_size=64,
                dcp_size=dcp_size,
                dcp_rank=dcp_rank,
                dcp_interleave=interleave,
            )
            for row_id, row in enumerate(rows):
                want = _expected_order(
                    row,
                    block_table[req_ids[row_id]],
                    block_size=64,
                    dcp_size=dcp_size,
                    dcp_rank=dcp_rank,
                    dcp_interleave=interleave,
                )
                count = expected_counts[row_id]
                assert want == expected_out[row_id][:count]
                assert all(v == -1 for v in expected_out[row_id][count:])


def test_random_case_actually_exercises_the_hard_paths() -> None:
    req_ids, block_table, rows = _random_case(
        width=256, batch=4, block_size=64, dcp_size=2, seed=7
    )
    flat = [v for row in rows for v in row]
    assert any(v < 0 for v in flat), "no negative sentinels generated"
    limit = 64 * 17 * 2
    assert any(v >= limit for v in flat), "no out-of-table tokens generated"
    assert len(set(req_ids)) > 1, "request routing not exercised"
    table_flat = [v for row in block_table for v in row]
    assert table_flat != sorted(table_flat), "page table is not fragmented"
