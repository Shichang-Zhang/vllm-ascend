from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("torch_npu")
pytest.importorskip("memfabric_hybrid")

from vllm_ascend.distributed.kv_transfer.sparse_kv_offload import (  # noqa: E402
    sparse_kv_offload_manager as manager_module,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.mooncake_host_pool import (  # noqa: E402
    MooncakeHostPool,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (  # noqa: E402
    SparseKVOffloadManager,
    SFA_HOST_KV_VISIBILITY_STAGES,
)


def _set_mooncake_allocator(manager):
    allocator = object.__new__(MooncakeHostPool)
    allocator.topology = SimpleNamespace()
    manager._host_kv_allocator = allocator


def _make_manager(*, mooncake: bool):
    manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
    manager.tp_rank = 0
    manager.tp_size = 1
    manager.tp_group = MagicMock()
    manager.host_backend = "mooncake" if mooncake else "memfabric"
    manager.use_fused_overlap = True
    manager.layer_name_to_offload_id = {"layer.0": 0}
    manager.token_size_bytes_k = 2 * torch.bfloat16.itemsize
    manager.token_size_bytes_v = 1 * torch.bfloat16.itemsize
    manager.current_kv_by_layer = {}
    manager.index_copy_probe_slots_by_layer = {}
    manager.adj_probe_enabled = True
    manager.sparse_kv_offload_cpp = MagicMock()
    manager.graph_host_kv_visibility_npu = torch.zeros(len(SFA_HOST_KV_VISIBILITY_STAGES), dtype=torch.int64)
    host_k = torch.zeros((8, 2), dtype=torch.bfloat16)
    host_v = torch.zeros((8, 1), dtype=torch.bfloat16)
    manager.k_caches_cpu = [host_k]
    manager.v_caches_cpu = [host_v]
    if mooncake:
        _set_mooncake_allocator(manager)
    else:
        manager._host_kv_allocator = None
    return manager, host_k, host_v


def test_visibility_probe_noop_when_disabled():
    manager, _, _ = _make_manager(mooncake=True)
    manager.adj_probe_enabled = False
    manager.trace_graph_host_kv_visibility("layer.0", stage="pre_join")
    manager.sparse_kv_offload_cpp.enqueue_graph_trace_tensor.assert_not_called()
    manager.sparse_kv_offload_cpp.enqueue_host_kv_visibility_compare_cpu.assert_not_called()


def test_visibility_probe_device_path_on_mooncake():
    manager, host_k, host_v = _make_manager(mooncake=True)
    # Write the current K/V into the host view first: the probe must count
    # zero mismatches for landed rows.
    slots = torch.tensor([2, 5], dtype=torch.int64)
    current_k = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    current_v = torch.tensor([[7.0], [8.0]], dtype=torch.bfloat16)
    manager.current_kv_by_layer[0] = (current_k, current_v)
    manager.index_copy_probe_slots_by_layer[0] = slots
    host_k[2] = current_k[0]
    host_k[5] = current_k[1]
    host_v[2] = current_v[0]
    host_v[5] = current_v[1]

    with patch.object(manager_module.torch_npu.npu, "current_stream", MagicMock()):
        manager.trace_graph_host_kv_visibility("layer.0", stage="pre_join")

    manager.sparse_kv_offload_cpp.enqueue_host_kv_visibility_compare_cpu.assert_not_called()
    assert manager.sparse_kv_offload_cpp.enqueue_graph_trace_tensor.call_count == 1
    traced = manager.sparse_kv_offload_cpp.enqueue_graph_trace_tensor.call_args
    pre_join_index = SFA_HOST_KV_VISIBILITY_STAGES.index("pre_join")
    assert traced.args[0] is manager.graph_host_kv_visibility_npu[pre_join_index : pre_join_index + 1]
    assert traced.args[1] == "host_kv_visibility_pre_join"
    assert traced.args[5] is False
    assert int(manager.graph_host_kv_visibility_npu[pre_join_index]) == 0

    # An unwritten row must be counted as a mismatch.
    manager.graph_host_kv_visibility_npu.zero_()
    host_k[5] = 0
    host_v[5] = 0
    with patch.object(manager_module.torch_npu.npu, "current_stream", MagicMock()):
        manager.trace_graph_host_kv_visibility("layer.0", stage="pre_join")
    assert int(manager.graph_host_kv_visibility_npu[pre_join_index]) == 1


def test_frame_open_stage_zeroes_all_counters_and_marks_frame():
    manager, host_k, host_v = _make_manager(mooncake=True)
    slots = torch.tensor([2], dtype=torch.int64)
    current_k = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
    current_v = torch.tensor([[7.0]], dtype=torch.bfloat16)
    manager.current_kv_by_layer[0] = (current_k, current_v)
    manager.index_copy_probe_slots_by_layer[0] = slots
    host_k[2] = current_k[0]
    host_v[2] = current_v[0]
    manager.graph_host_kv_visibility_npu.fill_(9)

    with patch.object(manager_module.torch_npu.npu, "current_stream", MagicMock()):
        manager.trace_graph_host_kv_visibility("layer.0", stage="post_write")

    # Only the post_write slot may be nonzero, and the record opens a frame.
    open_index = SFA_HOST_KV_VISIBILITY_STAGES.index("post_write")
    assert int(manager.graph_host_kv_visibility_npu[open_index]) == 0
    assert int(manager.graph_host_kv_visibility_npu.sum()) == 0
    traced = manager.sparse_kv_offload_cpp.enqueue_graph_trace_tensor.call_args
    assert traced.args[1] == "host_kv_visibility_post_write"
    assert traced.args[5] is True


def test_visibility_probe_hostview_path_on_memfabric():
    manager, host_k, host_v = _make_manager(mooncake=False)
    slots = torch.tensor([2, -1], dtype=torch.int64)
    current_k = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    current_v = torch.tensor([[7.0], [8.0]], dtype=torch.bfloat16)
    manager.current_kv_by_layer[0] = (current_k, current_v)
    manager.index_copy_probe_slots_by_layer[0] = slots

    manager.trace_graph_host_kv_visibility("layer.0", stage="post_join")

    manager.sparse_kv_offload_cpp.enqueue_graph_trace_tensor.assert_not_called()
    assert manager.sparse_kv_offload_cpp.enqueue_host_kv_visibility_compare_cpu.call_count == 1
    call = manager.sparse_kv_offload_cpp.enqueue_host_kv_visibility_compare_cpu.call_args
    assert call.args[0] is host_k.reshape(-1, 2)
    assert call.args[1] is host_v.reshape(-1, 1)
    assert torch.equal(call.args[4], slots)
    assert call.args[5] == 0
    assert call.args[7] == "post_join"


def test_offload_new_kv_brackets_the_ready_broadcast():
    """Experiment 1: post_write / pre_bcast / post_bcast are emitted per layer."""
    manager, _, _ = _make_manager(mooncake=True)
    manager.tp_size = 8
    manager.mtp_layer_id = -1
    manager._offload_new_kv_on_current_stream = MagicMock()
    manager.current_kv_writeback_on_side_stream = False
    slot_mapping = torch.tensor([3], dtype=torch.int64)
    current_k = torch.ones((1, 2), dtype=torch.bfloat16)
    current_v = torch.ones((1, 1), dtype=torch.bfloat16)

    with patch.object(manager_module.torch_npu.npu, "current_stream", MagicMock()):
        manager.offload_new_kv(
            "layer.0",
            slot_mapping,
            manager.k_caches_cpu[0],
            manager.v_caches_cpu[0],
            None,
            None,
            current_k,
            current_v,
            has_prefill=False,
            capturing=True,
        )

    stages = [call.args[1] for call in manager.sparse_kv_offload_cpp.enqueue_graph_trace_tensor.call_args_list]
    assert stages == [
        "slot_mapping_l0",
        "host_kv_visibility_post_write",
        "host_kv_visibility_pre_bcast",
        "host_kv_visibility_post_bcast",
    ]
    manager.tp_group.broadcast.assert_called_once()
    # Layer 0 post_write opens the probe frame.
    frames = [call.args[5] for call in manager.sparse_kv_offload_cpp.enqueue_graph_trace_tensor.call_args_list]
    assert frames == [False, True, False, False]


def test_offload_new_kv_skips_broadcast_probes_on_single_rank():
    manager, _, _ = _make_manager(mooncake=True)
    manager.tp_size = 1
    manager.mtp_layer_id = -1
    manager._offload_new_kv_on_current_stream = MagicMock()
    manager.current_kv_writeback_on_side_stream = False
    slot_mapping = torch.tensor([3], dtype=torch.int64)
    current_k = torch.ones((1, 2), dtype=torch.bfloat16)
    current_v = torch.ones((1, 1), dtype=torch.bfloat16)

    with patch.object(manager_module.torch_npu.npu, "current_stream", MagicMock()):
        manager.offload_new_kv(
            "layer.0",
            slot_mapping,
            manager.k_caches_cpu[0],
            manager.v_caches_cpu[0],
            None,
            None,
            current_k,
            current_v,
            has_prefill=False,
            capturing=True,
        )

    stages = [call.args[1] for call in manager.sparse_kv_offload_cpp.enqueue_graph_trace_tensor.call_args_list]
    assert "host_kv_visibility_pre_bcast" not in stages
    assert "host_kv_visibility_post_bcast" not in stages
    manager.tp_group.broadcast.assert_not_called()


def test_slot_mapping_diff_trace_on_capture():
    manager, _, _ = _make_manager(mooncake=True)
    manager.mtp_layer_id = -1
    manager._offload_new_kv_on_current_stream = MagicMock()
    slot_mapping = torch.tensor([3, 4], dtype=torch.int64)
    current_k = torch.ones((2, 2), dtype=torch.bfloat16)
    current_v = torch.ones((2, 1), dtype=torch.bfloat16)

    with patch.object(manager_module.torch_npu.npu, "current_stream", MagicMock()):
        manager.offload_new_kv(
            "layer.0",
            slot_mapping,
            manager.k_caches_cpu[0],
            manager.v_caches_cpu[0],
            None,
            None,
            current_k,
            current_v,
            has_prefill=False,
            capturing=True,
        )

    # Layer 0 on a graph capture: the slot cache is filled and the diff trace
    # is emitted (experiment 2), alongside the writeback path itself.
    assert manager.index_copy_probe_slots_by_layer[0] is slot_mapping
    stages = [call.args[1] for call in manager.sparse_kv_offload_cpp.enqueue_graph_trace_tensor.call_args_list]
    assert "slot_mapping_l0" in stages


def test_planner_stats_trace_publishes_counter_block():
    """Experiment 0: the planner counters are traced right after the call."""
    manager, _, _ = _make_manager(mooncake=True)
    manager.adj_planner_stats_cpu = torch.zeros(4, dtype=torch.int32)

    manager._trace_planner_stats(0)

    assert manager.sparse_kv_offload_cpp.enqueue_graph_trace_tensor.call_count == 1
    call = manager.sparse_kv_offload_cpp.enqueue_graph_trace_tensor.call_args
    assert call.args[0] is manager.adj_planner_stats_cpu
    assert call.args[1] == "planner_stats"
    assert call.args[4] == 4
    assert call.args[5] is False


def test_planner_stats_trace_is_noop_without_buffer():
    manager, _, _ = _make_manager(mooncake=True)
    manager.adj_planner_stats_cpu = None

    manager._trace_planner_stats(0)

    manager.sparse_kv_offload_cpp.enqueue_graph_trace_tensor.assert_not_called()


def test_sparse_copy_landing_probe_counts_mismatches():
    manager, _, _ = _make_manager(mooncake=False)
    manager.token_size_bytes_k = 2 * torch.bfloat16.itemsize
    manager.token_size_bytes_v = 1 * torch.bfloat16.itemsize
    host_k = torch.zeros((8, 2), dtype=torch.bfloat16)
    host_v = torch.zeros((8, 1), dtype=torch.bfloat16)
    exp_k = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    exp_v = torch.tensor([[7.0], [8.0]], dtype=torch.bfloat16)
    slots_cpu = torch.tensor([2, 5], dtype=torch.int64)
    valid_cpu = torch.tensor([True, True])
    expected = (exp_k, exp_v, slots_cpu, valid_cpu)

    with patch.object(manager_module.torch_npu.npu, "synchronize", MagicMock()):
        counts_all_stale = manager._probe_sparse_copy_landing(expected, host_k, host_v, 0.001)
    assert counts_all_stale == (2, 2, 2, 2)

    # Rows not landed yet: both rows mismatch, reported for K and V.
    with patch.object(manager_module.torch_npu.npu, "synchronize", MagicMock()):
        host_k[2] = exp_k[0]
        host_v[2] = exp_v[0]
        counts_partial = manager._probe_sparse_copy_landing(expected, host_k, host_v, 0.001)
    assert counts_partial == (1, 1, 1, 1)
