from __future__ import annotations

from types import SimpleNamespace

import pytest

from silkern.integrations.vllm import (
    AdapterError,
    WorkspaceAdapter,
    install_converter,
    validate_production_call,
)


def _valid_call() -> dict:
    return {
        "dcp_size": 2,
        "dcp_rank": 0,
        "cp_kv_cache_interleave_size": 1,
        "block_size": 64,
        "num_topk_tokens": 2048,
        "block_n": 128,
        "return_valid_counts": True,
        "compact_valid_to_front": True,
    }


@pytest.mark.parametrize("dcp_size", [2, 4])
@pytest.mark.parametrize("block_size", [32, 64])
def test_integrated_surface_accepts_pinned_backend_geometry(
    dcp_size: int,
    block_size: int,
) -> None:
    values = _valid_call()
    values["dcp_size"] = dcp_size
    values["dcp_rank"] = dcp_size - 1
    values["block_size"] = block_size
    validate_production_call(**values)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dcp_size", 1),
        ("dcp_size", 8),
        ("dcp_rank", -1),
        ("dcp_rank", 2),
        ("dcp_rank", True),
        ("cp_kv_cache_interleave_size", 2),
        ("block_size", 16),
        ("block_size", 128),
        ("num_topk_tokens", 1024),
        ("block_n", 64),
        ("return_valid_counts", False),
        ("compact_valid_to_front", False),
    ],
)
def test_integrated_surface_fails_closed(field: str, value: object) -> None:
    values = _valid_call()
    values[field] = value
    with pytest.raises(AdapterError):
        validate_production_call(**values)


def test_adapter_requires_known_arm() -> None:
    with pytest.raises(AdapterError):
        WorkspaceAdapter("atomic")  # type: ignore[arg-type]


def test_call_site_installation_is_scoped_and_restored() -> None:
    native = object()
    module = SimpleNamespace(triton_filter_and_convert_dcp_index=native)
    adapter = WorkspaceAdapter("row_stable")
    with install_converter(module, adapter):
        assert module.triton_filter_and_convert_dcp_index is adapter
    assert module.triton_filter_and_convert_dcp_index is native


def test_call_site_installation_rejects_unknown_source_surface() -> None:
    with pytest.raises(AdapterError):
        with install_converter(SimpleNamespace(), WorkspaceAdapter(
            "row_stable"
        )):
            pass
