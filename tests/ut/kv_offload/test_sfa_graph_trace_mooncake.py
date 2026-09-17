"""Regression coverage using a real HostRegister mapping, not an HBM stand-in."""

import uuid
import sys

import pytest
import torch
import torch_npu

from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.mooncake_host_pool import (
    HostPoolTopology,
    allocate_mooncake_host_region,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (
    FSA_SELECTION_MEMBERSHIP_STORAGE_INT16_COUNT,
    FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT,
    FSA_SELECTION_MEMBERSHIP_REQUIRER_COLUMNS,
    SparseKVOffloadManager,
)


def test_real_mooncake_host_alias_survives_graph_replays():
    engine = sys.modules.get("mooncake.engine")
    if engine is not None and getattr(engine, "__file__", None) is None:
        pytest.skip("UT mock replaces mooncake.engine; run this file directly for the real mapping test")
    if not torch_npu.npu.is_available():
        pytest.skip("Ascend NPU is unavailable")
    torch_npu.npu.set_device(0)
    rows = 4
    columns = FSA_SELECTION_MEMBERSHIP_STORAGE_INT16_COUNT
    region = allocate_mooncake_host_region(
        size_bytes=rows * columns * torch.int16.itemsize,
        alignment=64,
        topology=HostPoolTopology(tp_rank=0, tp_size=1, device_id=0),
        name="sfa_graph_trace_test_" + uuid.uuid4().hex,
    )
    manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
    manager._build_cpp()
    mapped = region.tensor.view(torch.int16).view(rows, columns)
    host_alias = manager._restore_int16_tensor(region.host_data_ptr, [rows, columns])
    start = FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT - 1024
    end = FSA_SELECTION_MEMBERSHIP_REQUIRER_COLUMNS
    window = host_alias[:, start:end]
    mapped_window = mapped[:, start:end]
    staging = torch.full((rows, end - start), 17, dtype=torch.int16, device="npu")
    assert window.stride() == (columns, 1)
    assert not window.is_contiguous()
    print(f"Host VA={region.host_data_ptr:#x}, device VA={mapped.data_ptr():#x}", flush=True)
    graph = torch.npu.NPUGraph()
    try:
        mapped_window.copy_(staging, non_blocking=True)
        torch_npu.npu.synchronize()
        print("Host window after publish:", window[0, :8].tolist(), flush=True)
        print("NPU window after publish:", mapped_window.contiguous().cpu()[0, :8].tolist(), flush=True)
        assert bool((window == 17).all())
        staging.fill_(29)
        with torch.npu.graph(graph):
            mapped_window.copy_(staging, non_blocking=True)
            manager.sparse_kv_offload_cpp.enqueue_graph_trace_tensor(
                window, "real_mooncake_host_window", 4, 0, window.numel()
            )
        for _ in range(3):
            host_alias.fill_(43)
            graph.replay()
            torch_npu.npu.synchronize()
            assert bool((window == 29).all())
    finally:
        del graph
        torch_npu.npu.synchronize()
        region.release()


if __name__ == "__main__":
    # Run independently of tests/ut/conftest.py's mooncake.engine mock.
    test_real_mooncake_host_alias_survives_graph_replays()
    print("REAL_MOONCAKE_GRAPH_TRACE_OK", flush=True)
