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


@pytest.mark.parametrize(
    ("tp_size", "tp_rank", "capturing", "expected_events"),
    [
        (1, 0, True, ["writeback", "wait"]),
        (2, 0, True, ["writeback", "wait", "broadcast"]),
        (2, 1, True, ["writeback", "broadcast"]),
        (1, 0, False, ["writeback"]),
        (2, 0, False, ["writeback", "broadcast"]),
    ],
)
def test_mooncake_decode_writeback_waits_before_publish(tp_size, tp_rank, capturing, expected_events):
    manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
    manager.tp_rank = tp_rank
    manager.tp_size = tp_size
    manager.use_fused_overlap = True
    manager.layer_name_to_offload_id = {"layer.0": 0}
    manager.current_kv_by_layer = {}
    events = []
    manager._offload_new_kv_on_current_stream = MagicMock(side_effect=lambda *args: events.append("writeback"))
    manager.tp_group = SimpleNamespace(
        broadcast=MagicMock(side_effect=lambda *args, **kwargs: events.append("broadcast"))
    )
    _set_mooncake_allocator(manager)
    manager.current_kv_save_stream = MagicMock()
    current_stream = MagicMock()
    current_stream.wait_stream.side_effect = lambda stream: events.append("wait")

    slot_mapping = torch.tensor([2], dtype=torch.int64)
    host_k = torch.zeros((4, 2), dtype=torch.bfloat16)
    host_v = torch.zeros((4, 1), dtype=torch.bfloat16)
    current_k = torch.ones((1, 2), dtype=torch.bfloat16)
    current_v = torch.ones((1, 1), dtype=torch.bfloat16)

    with patch.object(manager_module.torch_npu.npu, "current_stream", return_value=current_stream):
        manager.offload_new_kv(
            "layer.0",
            slot_mapping,
            host_k,
            host_v,
            None,
            None,
            current_k,
            current_v,
            capturing=capturing,
        )
        # A consumer can run as soon as offload_new_kv returns. A later wait
        # after attention cannot protect that consumer's Host KV reads.
        assert events == expected_events

    manager._offload_new_kv_on_current_stream.assert_called_once_with(
        slot_mapping, host_k, host_v, None, None, current_k, current_v, False, capturing, True
    )
    assert manager.current_kv_by_layer[0] == (current_k, current_v)
    if capturing and tp_rank == 0:
        current_stream.wait_stream.assert_called_once_with(manager.current_kv_save_stream)
    else:
        current_stream.wait_stream.assert_not_called()


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
    manager.mtp_layer_id = 2
    manager.current_kv_save_stream = MagicMock()
    manager.use_fused_overlap = True
    manager.layer_name_to_offload_id = {
        "layer.0": 0,
        "layer.1": 1,
    }
    manager.current_kv_by_layer = {}
    manager._offload_new_kv_on_current_stream = MagicMock()
    _set_mooncake_allocator(manager)

    slot_mapping = torch.tensor([2], dtype=torch.int64)
    host_k = torch.zeros((4, 2), dtype=torch.bfloat16)
    host_v = torch.zeros((4, 1), dtype=torch.bfloat16)
    current_k = torch.ones((1, 2), dtype=torch.bfloat16)
    current_v = torch.ones((1, 1), dtype=torch.bfloat16)

    with patch.object(manager_module.torch_npu.npu, "current_stream"):
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


def test_graph_mooncake_mtp_writeback_refreshes_slots_each_iteration():
    manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
    manager.tp_rank = 0
    manager.tp_size = 1
    manager.use_fused_overlap = True
    manager.mtp_layer_id = 1
    manager.layer_name_to_offload_id = {"layer.0": 0, "mtp": 1}
    manager.current_kv_by_layer = {}
    manager.current_kv_save_stream = MagicMock()
    manager.token_size_bytes_k = 2 * torch.bfloat16.itemsize
    manager.token_size_bytes_v = torch.bfloat16.itemsize
    manager.max_d2h_index_copy_tokens = 4
    _set_mooncake_allocator(manager)
    for name in ("slot_mapping_cpu", "src_idx_cpu", "dst_idx_cpu", "src_idx_npu", "dst_idx_npu"):
        setattr(manager, "d2h_" + name, torch.zeros(4, dtype=torch.int64))
    manager.d2h_index_count_cpu = torch.zeros(1, dtype=torch.int32)
    manager.d2h_index_count_npu = torch.zeros(1, dtype=torch.int32)

    def enqueue_descriptors(slots, token_count, capacity, num_slots, src_idx, dst_idx, count):
        assert token_count == 1
        assert 0 <= slots[0] < num_slots
        src_idx.zero_()
        dst_idx.fill_(slots[0].item())
        count.fill_(token_count)

    manager.sparse_kv_offload_cpp = SimpleNamespace(
        enqueue_current_kv_index_copy_descriptors=MagicMock(side_effect=enqueue_descriptors)
    )
    target_k = torch.zeros((8, 2), dtype=torch.bfloat16)
    target_v = torch.zeros((8, 1), dtype=torch.bfloat16)
    mtp_k = torch.zeros_like(target_k)
    mtp_v = torch.zeros_like(target_v)

    with (
        patch.object(manager_module.torch_npu.npu, "current_stream"),
        patch.object(manager_module.torch_npu.npu, "stream", side_effect=lambda _: nullcontext()),
    ):
        for layer_name, slot, value, host_k, host_v in (
            ("layer.0", 2, 11, target_k, target_v),
            ("mtp", 5, 31, mtp_k, mtp_v),
            ("mtp", 6, 41, mtp_k, mtp_v),
        ):
            manager.offload_new_kv(
                layer_name,
                torch.tensor([slot], dtype=torch.int64),
                host_k,
                host_v,
                None,
                None,
                torch.tensor([[value, value + 1]], dtype=torch.bfloat16),
                torch.tensor([[value + 2]], dtype=torch.bfloat16),
                capturing=True,
            )

    assert torch.equal(target_k[2], torch.tensor([11, 12], dtype=torch.bfloat16))
    assert torch.equal(target_v[2], torch.tensor([13], dtype=torch.bfloat16))
    for slot, value in ((5, 31), (6, 41)):
        assert torch.equal(mtp_k[slot], torch.tensor([value, value + 1], dtype=torch.bfloat16))
        assert torch.equal(mtp_v[slot], torch.tensor([value + 2], dtype=torch.bfloat16))
    assert torch.count_nonzero(mtp_k[2]).item() == 0
    assert torch.count_nonzero(mtp_v[2]).item() == 0
