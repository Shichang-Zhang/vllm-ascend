from contextlib import nullcontext
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
)


def _set_mooncake_allocator(manager):
    allocator = object.__new__(MooncakeHostPool)
    allocator.topology = SimpleNamespace()
    manager._host_kv_allocator = allocator


def _make_index_copy_probe_manager(*, mock_device_va=True):
    manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
    manager.index_copy_probe_enabled = True
    manager.index_copy_probe_steps = 0
    manager.tp_rank = 0
    manager.layer_name_to_offload_id = {"layer.0": 0}
    manager.token_size_bytes_k = 2 * torch.bfloat16.itemsize
    manager.token_size_bytes_v = torch.bfloat16.itemsize
    manager.current_kv_by_layer = {
        0: (
            torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16),
            torch.tensor([[5], [6]], dtype=torch.bfloat16),
        ),
    }
    manager.index_copy_probe_slots_by_layer = {
        0: torch.tensor([2, 4], dtype=torch.int64),
    }
    manager.k_caches_cpu = [torch.zeros((8, 2), dtype=torch.bfloat16)]
    manager.v_caches_cpu = [torch.zeros((8, 1), dtype=torch.bfloat16)]
    manager.k_caches_cpu[0][[2, 4]] = manager.current_kv_by_layer[0][0]
    manager.v_caches_cpu[0][[2, 4]] = manager.current_kv_by_layer[0][1]
    if mock_device_va:
        # These CPU-only unit tests validate probe comparison logic. Production
        # execution still unconditionally checks for Mooncake's NPU Device VA.
        manager._assert_mooncake_device_va = lambda *args, **kwargs: None
    return manager


def test_selection_debug_is_noop_when_disabled():
    manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
    manager.debug_mooncake_selection(
        "layer.0", block_table=None, req_ids=None,
        stable_prefix_lens=None, topk_indices=None, selection_kv_cache=None,
        selection_k_rope=None, skip_topk=False,
    )


def test_graph_trace_filters_to_layer_four_and_descriptor_owners():
    manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
    manager.index_copy_probe_enabled = True
    manager.mtp_layer_id = 79
    manager.tp_rank = 3
    manager.sparse_kv_offload_cpp = MagicMock()
    tensor = torch.tensor([1, 2], dtype=torch.int16)

    manager._enqueue_graph_trace_cpu("planner_output_cpu", 5, tensor, 2)
    manager.sparse_kv_offload_cpp.enqueue_graph_trace_tensor.assert_not_called()

    manager._enqueue_graph_trace_cpu("planner_output_cpu", 4, tensor, 2)
    manager._enqueue_graph_trace_cpu("index_descriptor_dst_cpu", 0, tensor, 2)
    manager._enqueue_graph_trace_cpu("index_descriptor_dst_cpu", 79, tensor, 2)

    calls = manager.sparse_kv_offload_cpp.enqueue_graph_trace_tensor.call_args_list
    assert [call.args[1:3] for call in calls] == [
        ("planner_output_cpu", 4),
        ("index_descriptor_dst_cpu", 0),
        ("index_descriptor_dst_cpu", 79),
    ]
    assert all(call.args[3:] == (3, 2) for call in calls)


@pytest.mark.parametrize("reused", [False, True])
def test_selection_debug_labels_current_history_and_plan_reuse(reused):
    manager = _make_index_copy_probe_manager()
    manager.index_copy_probe_steps = 1
    manager.block_size = 2
    manager.fused_overlap_plan_owner_layer_id = 1 if reused else 0
    manager.fused_plan_current_linear_slots_npu = torch.tensor([0, 1])
    manager.lru_miss_count_cpu_list = [torch.tensor([1, 0])]
    manager.lru_miss_tokens_cpu_list = [torch.tensor([[1, 0], [0, 0]])]
    if reused:
        # Stale per-layer miss arrays must never be inspected on plan reuse.
        manager.lru_miss_count_cpu_list = None
        manager.lru_miss_tokens_cpu_list = None
    current_k, current_v = manager.current_kv_by_layer[0]
    with patch.object(manager_module.logger, "warning") as warning:
        manager.debug_mooncake_selection(
            "layer.0",
            block_table=torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]]),
            req_ids=torch.tensor([7, 7]),
            stable_prefix_lens=torch.tensor([2, 2]),
            topk_indices=torch.tensor([[0, 1, 2], [1, 0, 3]]),
            selection_kv_cache=current_k.clone(),
            selection_k_rope=current_v.clone(),
            skip_topk=reused,
        )
    context = warning.call_args_list[0].args
    assert "[SFA_KV_CONTEXT]" in context[0]
    assert context[6] == reused
    assert context[11:13] == (True, True)
    row = warning.call_args_list[1].args
    assert "[SFA_KV_ROWS]" in row[0]
    assert row[8] == 2  # current token position
    assert row[10] == [0, 1]  # stable historical candidates only
    assert row[11] == [0, 1]  # physical host slots
    assert (row[13] is None) == reused
    second_row = warning.call_args_list[2].args
    assert second_row[5] == 7  # repeated request ID after MTP expansion
    assert second_row[6] == 1  # second token within the same request
    assert second_row[8] == 3  # stable prefix plus token ordinal
    assert second_row[11] == [1, 0]


def test_eager_current_kv_index_copy_filters_invalid_slots():
    manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
    manager.token_size_bytes_k = 2 * torch.bfloat16.itemsize
    manager.token_size_bytes_v = 1 * torch.bfloat16.itemsize
    manager.max_d2h_index_copy_tokens = 4

    host_k = torch.zeros((8, 2), dtype=torch.bfloat16)
    host_v = torch.zeros((8, 1), dtype=torch.bfloat16)
    current_k = torch.tensor(
        [[1, 2], [3, 4], [5, 6]], dtype=torch.bfloat16
    )
    current_v = torch.tensor([[7], [8], [9]], dtype=torch.bfloat16)

    manager._offload_new_kv_via_index_copy(
        slot_mapping=torch.tensor([3, -1, 5], dtype=torch.int64),
        k_cache_cpu=host_k,
        v_cache_cpu=host_v,
        k=current_k,
        v=current_v,
        capturing=False,
    )

    assert torch.equal(host_k[3], current_k[0])
    assert torch.equal(host_v[3], current_v[0])
    assert torch.equal(host_k[5], current_k[2])
    assert torch.equal(host_v[5], current_v[2])
    assert torch.count_nonzero(host_k[0]).item() == 0
    assert torch.count_nonzero(host_v[0]).item() == 0


def test_index_copy_probe_accepts_exact_shared_host_values():
    manager = _make_index_copy_probe_manager()

    manager._probe_mooncake_index_copy(
        "layer.0",
        stage="all_tp_post_plan",
    )

    assert manager.index_copy_probe_steps == 1


def test_index_copy_probe_rejects_cpu_shared_segment():
    manager = _make_index_copy_probe_manager(mock_device_va=False)

    with pytest.raises(RuntimeError, match="NPU Device VA, not as a CPU tensor"):
        manager._probe_mooncake_index_copy(
            "layer.0",
            stage="all_tp_post_plan",
        )


def test_index_copy_probe_reports_first_value_mismatch():
    manager = _make_index_copy_probe_manager()
    manager.k_caches_cpu[0][4, 1] = 99

    with pytest.raises(RuntimeError, match=r"tp_rank=0.*K:source_row=1,slot=4,element=1"):
        manager._probe_mooncake_index_copy(
            "layer.0",
            stage="all_tp_post_plan",
        )


def test_index_copy_probe_rejects_duplicate_destinations():
    manager = _make_index_copy_probe_manager()
    manager.index_copy_probe_slots_by_layer[0] = torch.tensor(
        [2, 2],
        dtype=torch.int64,
    )

    with pytest.raises(RuntimeError, match="duplicate Decode destinations"):
        manager._probe_mooncake_index_copy(
            "layer.0",
            stage="all_tp_post_plan",
        )


def test_eager_prefill_kv_index_copy_filters_invalid_slots():
    manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
    manager.token_size_bytes_k = 2 * torch.bfloat16.itemsize
    manager.token_size_bytes_v = torch.bfloat16.itemsize
    manager.max_num_tokens = 4

    host_k = torch.zeros((8, 2), dtype=torch.bfloat16)
    host_v = torch.zeros((8, 1), dtype=torch.bfloat16)
    device_k = torch.arange(16, dtype=torch.bfloat16).reshape(8, 2)
    device_v = torch.arange(8, dtype=torch.bfloat16).reshape(8, 1)

    manager._offload_prefill_kv_via_index_copy(
        slot_mapping=torch.tensor([3, -1, 5], dtype=torch.int64),
        k_cache_cpu=host_k,
        v_cache_cpu=host_v,
        k_cache_npu=device_k,
        v_cache_npu=device_v,
    )

    assert torch.equal(host_k[3], device_k[3])
    assert torch.equal(host_v[3], device_v[3])
    assert torch.equal(host_k[5], device_k[5])
    assert torch.equal(host_v[5], device_v[5])
    assert torch.count_nonzero(host_k[0]).item() == 0
    assert torch.count_nonzero(host_v[0]).item() == 0


def test_mooncake_prefill_dispatches_to_index_copy():
    manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
    manager.tp_rank = 0
    manager.use_fused_overlap = True
    manager.sparse_kv_offload_config = SimpleNamespace(
        keep_device_kv_cache=True,
    )
    manager._offload_prefill_kv_via_index_copy = MagicMock()
    _set_mooncake_allocator(manager)

    slot_mapping = torch.tensor([2], dtype=torch.int64)
    host_k = torch.zeros((4, 2), dtype=torch.bfloat16)
    host_v = torch.zeros((4, 1), dtype=torch.bfloat16)
    device_k = torch.ones((4, 2), dtype=torch.bfloat16)
    device_v = torch.ones((4, 1), dtype=torch.bfloat16)

    with patch.object(manager_module.offload, "sparse_copy") as sparse_copy:
        manager._offload_new_kv_on_current_stream(
            slot_mapping,
            host_k,
            host_v,
            device_k,
            device_v,
            None,
            None,
            has_prefill=True,
        )

    manager._offload_prefill_kv_via_index_copy.assert_called_once_with(
        slot_mapping=slot_mapping,
        k_cache_cpu=host_k,
        v_cache_cpu=host_v,
        k_cache_npu=device_k,
        v_cache_npu=device_v,
    )
    sparse_copy.assert_not_called()


def test_graph_mooncake_writeback_waits_for_save_stream():
    manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
    manager.tp_rank = 0
    manager.tp_size = 1
    manager.tp_group = MagicMock()
    manager.use_fused_overlap = True
    manager.mtp_layer_id = -1
    manager.layer_name_to_offload_id = {"layer.0": 0}
    manager.current_kv_by_layer = {}
    manager._offload_new_kv_on_current_stream = MagicMock()
    _set_mooncake_allocator(manager)
    manager.current_kv_save_stream = MagicMock()
    current_stream = MagicMock()
    current_kv_ready = object()
    current_stream.record_event.return_value = current_kv_ready

    slot_mapping = torch.tensor([2], dtype=torch.int64)
    host_k = torch.zeros((4, 2), dtype=torch.bfloat16)
    host_v = torch.zeros((4, 1), dtype=torch.bfloat16)
    current_k = torch.ones((1, 2), dtype=torch.bfloat16)
    current_v = torch.ones((1, 1), dtype=torch.bfloat16)

    with (
        patch.object(manager_module.torch_npu.npu, "current_stream", return_value=current_stream),
        patch.object(
            manager_module.torch_npu.npu,
            "stream",
            side_effect=lambda _: nullcontext(),
        ),
    ):
        manager.offload_new_kv(
            "layer.0",
            slot_mapping,
            host_k,
            host_v,
            None,
            None,
            current_k,
            current_v,
            capturing=True,
        )

    manager._offload_new_kv_on_current_stream.assert_called_once_with(
        slot_mapping,
        host_k,
        host_v,
        None,
        None,
        current_k,
        current_v,
        False,
        True,
        True,
    )
    assert manager.current_kv_by_layer[0] == (current_k, current_v)
    current_stream.record_event.assert_not_called()
    manager.current_kv_save_stream.wait_event.assert_not_called()
    current_stream.wait_stream.assert_called_once_with(manager.current_kv_save_stream)


def test_mooncake_decode_publishes_ready_after_tp0_write():
    manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
    manager.tp_rank = 0
    manager.tp_size = 2
    manager.tp_group = MagicMock()
    manager.use_fused_overlap = True
    manager.mtp_layer_id = -1
    manager.layer_name_to_offload_id = {"layer.0": 0}
    manager.current_kv_by_layer = {}
    manager.index_copy_probe_enabled = False
    manager._offload_new_kv_on_current_stream = MagicMock()
    _set_mooncake_allocator(manager)

    slot_mapping = torch.tensor([2], dtype=torch.int64)
    host_k = torch.zeros((4, 2), dtype=torch.bfloat16)
    host_v = torch.zeros((4, 1), dtype=torch.bfloat16)
    current_k = torch.ones((1, 2), dtype=torch.bfloat16)
    current_v = torch.ones((1, 1), dtype=torch.bfloat16)

    manager.offload_new_kv(
        "layer.0",
        slot_mapping,
        host_k,
        host_v,
        None,
        None,
        current_k,
        current_v,
        capturing=False,
    )

    manager._offload_new_kv_on_current_stream.assert_called_once()
    manager.tp_group.broadcast.assert_called_once()
    ready_tensor = manager.tp_group.broadcast.call_args.args[0]
    assert ready_tensor.dtype == torch.int8
    assert manager.tp_group.broadcast.call_args.kwargs == {"src": 0}


def test_graph_mooncake_index_copy_runs_on_save_stream():
    manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
    manager.token_size_bytes_k = 2 * torch.bfloat16.itemsize
    manager.token_size_bytes_v = torch.bfloat16.itemsize
    manager.max_d2h_index_copy_tokens = 4
    manager.d2h_slot_mapping_cpu = torch.zeros(4, dtype=torch.int64)
    manager.d2h_src_idx_cpu = torch.zeros(4, dtype=torch.int64)
    manager.d2h_dst_idx_cpu = torch.zeros(4, dtype=torch.int64)
    manager.d2h_index_count_cpu = torch.zeros(1, dtype=torch.int32)
    manager.d2h_src_idx_npu = torch.zeros(4, dtype=torch.int64)
    manager.d2h_dst_idx_npu = torch.zeros(4, dtype=torch.int64)
    manager.d2h_index_count_npu = torch.zeros(1, dtype=torch.int32)
    manager.current_kv_save_stream = MagicMock()
    current_stream = MagicMock()
    descriptors_ready = object()
    current_stream.record_event.return_value = descriptors_ready

    def enqueue_descriptors(
        slot_mapping,
        num_actual_tokens,
        max_num_tokens,
        num_host_slots,
        src_idx,
        dst_idx,
        count,
    ):
        assert slot_mapping[0].item() == 2
        assert num_actual_tokens == 1
        assert max_num_tokens == 4
        assert num_host_slots == 4
        src_idx.zero_()
        dst_idx.fill_(2)
        count.fill_(1)

    sparse_kv_ops = SimpleNamespace(
        enqueue_current_kv_index_copy_descriptors=MagicMock(
            side_effect=enqueue_descriptors,
        ),
    )
    manager.sparse_kv_offload_cpp = sparse_kv_ops
    host_k = torch.zeros((4, 2), dtype=torch.bfloat16)
    host_v = torch.zeros((4, 1), dtype=torch.bfloat16)
    current_k = torch.tensor([[1, 2]], dtype=torch.bfloat16)
    current_v = torch.tensor([[3]], dtype=torch.bfloat16)

    with (
        patch.object(manager_module.torch_npu.npu, "current_stream", return_value=current_stream),
        patch.object(
            manager_module.torch_npu.npu,
            "stream",
            side_effect=lambda _: nullcontext(),
        ),
    ):
        manager._offload_new_kv_via_index_copy(
            slot_mapping=torch.tensor([2], dtype=torch.int64),
            k_cache_cpu=host_k,
            v_cache_cpu=host_v,
            k=current_k,
            v=current_v,
            capturing=True,
            prepare_descriptors=True,
        )

    assert torch.equal(host_k[2], current_k[0])
    assert torch.equal(host_v[2], current_v[0])
    current_stream.record_event.assert_called_once_with()
    manager.current_kv_save_stream.wait_event.assert_called_once_with(
        descriptors_ready
    )
    sparse_kv_ops.enqueue_current_kv_index_copy_descriptors.assert_called_once()


def test_graph_mooncake_prepares_descriptors_on_first_layer_only():
    manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
    manager.tp_rank = 0
    manager.tp_size = 1
    manager.tp_group = MagicMock()
    manager.use_fused_overlap = True
    manager.mtp_layer_id = -1
    manager.layer_name_to_offload_id = {
        "layer.0": 0,
        "layer.1": 1,
    }
    manager.current_kv_by_layer = {}
    manager.current_kv_save_stream = MagicMock()
    manager._offload_new_kv_on_current_stream = MagicMock()
    _set_mooncake_allocator(manager)

    slot_mapping = torch.tensor([2], dtype=torch.int64)
    host_k = torch.zeros((4, 2), dtype=torch.bfloat16)
    host_v = torch.zeros((4, 1), dtype=torch.bfloat16)
    current_k = torch.ones((1, 2), dtype=torch.bfloat16)
    current_v = torch.ones((1, 1), dtype=torch.bfloat16)

    with patch.object(
        manager_module.torch_npu.npu,
        "current_stream",
        return_value=MagicMock(),
    ):
        for layer_name in ("layer.0", "layer.1"):
            manager.offload_new_kv(
                layer_name,
                slot_mapping,
                host_k,
                host_v,
                None,
                None,
                current_k,
                current_v,
                capturing=True,
            )

        assert manager._offload_new_kv_on_current_stream.call_count == 2
        first_call, second_call = (
            manager._offload_new_kv_on_current_stream.call_args_list
        )
        assert first_call.args[-3:] == (
            False,
            True,
            True,
        )
        assert second_call.args[-3:] == (
            False,
            True,
            False,
        )
        assert [call.args[-1] for call in (first_call, second_call)] == [True, False]


def test_eager_selection_probe_skips_actual_acl_capture():
    manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
    manager.index_copy_probe_enabled = True
    block_table = MagicMock()
    with patch.object(manager_module.torch_npu.npu, "is_current_stream_capturing", return_value=True):
        manager.debug_mooncake_selection(
            "layer.4", block_table=block_table, req_ids=None,
            stable_prefix_lens=None, topk_indices=None, selection_kv_cache=None,
            selection_k_rope=None, skip_topk=False,
        )
    block_table.detach.assert_not_called()


def test_graph_mooncake_prepares_descriptors_again_for_mtp_layer():
    manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
    manager.tp_rank = 0
    manager.tp_size = 1
    manager.tp_group = MagicMock()
    manager.use_fused_overlap = True
    manager.layer_name_to_offload_id = {
        "layer.0": 0,
        "layer.1": 1,
        "mtp": 2,
    }
    manager.mtp_layer_id = 2
    manager.current_kv_by_layer = {}
    manager.current_kv_save_stream = MagicMock()
    manager._offload_new_kv_on_current_stream = MagicMock()
    _set_mooncake_allocator(manager)

    target_slot_mapping = torch.tensor([2], dtype=torch.int64)
    mtp_slot_mapping = torch.tensor([3], dtype=torch.int64)
    host_k = torch.zeros((4, 2), dtype=torch.bfloat16)
    host_v = torch.zeros((4, 1), dtype=torch.bfloat16)
    current_k = torch.ones((1, 2), dtype=torch.bfloat16)
    current_v = torch.ones((1, 1), dtype=torch.bfloat16)

    with patch.object(
        manager_module.torch_npu.npu,
        "current_stream",
        return_value=MagicMock(),
    ):
        for layer_name, slot_mapping in (
            ("layer.0", target_slot_mapping),
            ("layer.1", target_slot_mapping),
            ("mtp", mtp_slot_mapping),
        ):
            manager.offload_new_kv(
                layer_name,
                slot_mapping,
                host_k,
                host_v,
                None,
                None,
                current_k,
                current_v,
                capturing=True,
            )

        calls = manager._offload_new_kv_on_current_stream.call_args_list
        assert len(calls) == 3
        assert [call.args[-1] for call in calls] == [True, False, True]
        assert calls[2].args[-1] is True
        assert torch.equal(calls[0].args[0], target_slot_mapping)
