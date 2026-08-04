from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from silkern import exact_stable_compact_reference  # noqa: E402
from silkern.integrations.vllm import (  # noqa: E402
    AdapterError,
    WorkspaceAdapter,
)

pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)


def _case():
    batch = 3
    width = 2048
    table_width = 32
    req_ids = [2, 0, 1]
    block_table = [
        [request * table_width + (column * 7) % table_width
         for column in range(table_width)]
        for request in range(3)
    ]
    rows = []
    for row in range(batch):
        values = [
            -1 if column % 19 == 0 else (column * 13 + row * 17) % 4096
            for column in range(width)
        ]
        rows.append(values)
    return req_ids, block_table, rows


@pytest.mark.parametrize("arm", ["row_stable", "hierarchical_stable"])
def test_adapter_matches_exact_oracle_and_replays_at_fixed_addresses(arm: str) -> None:
    case = _case()
    expected = exact_stable_compact_reference(
        *case,
        block_size=64,
        dcp_size=2,
        dcp_rank=1,
        dcp_interleave=1,
    )
    req_ids, block_table, tokens = (
        torch.tensor(values, dtype=torch.int32, device="cuda") for values in case
    )
    adapter = WorkspaceAdapter(arm)  # type: ignore[arg-type]
    adapter.prepare(
        req_ids,
        block_table,
        tokens,
        dcp_size=2,
        dcp_rank=1,
        block_size=64,
    )
    signatures = adapter.fixed_buffer_signatures

    def op() -> None:
        output, counts = adapter(
            req_ids,
            block_table,
            tokens,
            2,
            1,
            BLOCK_SIZE=64,
            NUM_TOPK_TOKENS=2048,
            BLOCK_N=128,
            return_valid_counts=True,
        )
        assert output.data_ptr() == adapter.output.data_ptr()
        assert counts.data_ptr() == adapter.counts.data_ptr()

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
    allocated = torch.cuda.memory_allocated()
    for _ in range(100):
        graph.replay()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == allocated
    assert adapter.fixed_buffer_signatures == signatures
    assert adapter.output.cpu().tolist() == expected[0]
    assert adapter.counts.cpu().tolist() == expected[1]


def test_adapter_rejects_a_different_fixed_address_binding() -> None:
    case = _case()
    req_ids, block_table, tokens = (
        torch.tensor(values, dtype=torch.int32, device="cuda") for values in case
    )
    adapter = WorkspaceAdapter("row_stable")
    adapter.prepare(
        req_ids,
        block_table,
        tokens,
        dcp_size=2,
        dcp_rank=0,
        block_size=64,
    )
    with pytest.raises(AdapterError, match="binding changed"):
        adapter(
            req_ids.clone(),
            block_table,
            tokens,
            2,
            0,
            BLOCK_SIZE=64,
            NUM_TOPK_TOKENS=2048,
            return_valid_counts=True,
        )
