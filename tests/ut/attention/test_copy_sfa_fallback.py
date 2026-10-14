# SPDX-License-Identifier: Apache-2.0
"""Colocate fallback uses a packed arena without overwriting Copy-SFA rows."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from tests.ut.attention.test_copy_sfa_metadata import common, make_builder, populate
from vllm_ascend.attention.sfa_kv_offload import AscendAttentionState, AscendSFAKVOffloadImpl
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload import sparse_kv_offload_manager as manager_module
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.copy_sfa_topk_slots import (
    COPY_SFA_REQUEST_STATE_DENSE_RESTORE,
    copy_sfa_pool_capacity,
)

MODULE = "vllm_ascend.attention.sfa_kv_offload"
HOT = 8192
STRIDE = HOT + 256


def make_manager(keep_device=True):
    cfg = SimpleNamespace(use_fused_copy_sfa=True, topk_buffer_size=HOT, keep_device_kv_cache=keep_device)
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(num_speculative_tokens=3),
        scheduler_config=SimpleNamespace(max_num_seqs=2, max_num_batched_tokens=16),
        cache_config=SimpleNamespace(block_size=128),
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(kv_lora_rank=3, qk_rope_head_dim=1)),
    )
    empty = torch.empty

    def cpu_empty(*args, **kwargs):
        kwargs["device"] = "cpu"
        return empty(*args, **kwargs)

    with patch.object(manager_module.torch, "empty", side_effect=cpu_empty):
        k, v = manager_module.allocate_kv_offload_topk_buffer_pair(config, cfg)
    result = manager_module.SparseKVOffloadManager.__new__(manager_module.SparseKVOffloadManager)
    result.use_fused_copy_sfa = True
    result.sparse_kv_offload_config = cfg
    result.max_num_reqs = 2
    result.max_num_topk_rows = 8
    result.topk_buffer_size = HOT
    result.block_size = 128
    result.topk_buffers_k, result.topk_buffers_v = [k], [v]
    result.layer_name_to_offload_id = {"layer.0": 0}
    result.current_slots_npu = torch.zeros((8, 4), dtype=torch.int32)
    result.resident_block_table_npu = torch.arange(8 * HOT // 128, dtype=torch.int32).view(8, -1)
    result.resident_query_lens_npu = torch.arange(1, 9, dtype=torch.int32)
    result.resident_seq_lens_npu = torch.full((8,), HOT, dtype=torch.int32)
    result._bind_resident_fallback_buffers()
    return result


def test_profile_allocation_includes_a_private_packed_fallback_arena():
    manager = make_manager()
    pool_rows = 2 * copy_sfa_pool_capacity(manager.max_num_reqs)
    assert manager.topk_buffers_k[0].shape == (pool_rows + 8, STRIDE, 1, 3)
    k, v, _, table, _, _ = AscendSFAKVOffloadImpl._resident_views(manager, "layer.0", 2)
    assert k.shape == (2 * HOT // 128, 128, 1, 3)
    assert v.shape == (2 * HOT // 128, 128, 1, 1)
    assert manager.resident_buffers_k[0].stride(0) == HOT * 3
    manager.resident_buffers_k[0].fill_(5)
    manager.resident_buffers_v[0].fill_(6)
    assert not manager.topk_buffers_k[0][:pool_rows].count_nonzero()
    assert not manager.topk_buffers_v[0][:pool_rows].count_nonzero()
    assert k[table[1, 0]].flatten()[0] == 5


def test_disaggregated_allocation_does_not_reserve_fallback_bytes():
    manager = make_manager(keep_device=False)
    assert manager.topk_buffers_k[0].shape[0] == 2 * copy_sfa_pool_capacity(manager.max_num_reqs)
    assert manager.resident_buffers_k is manager.topk_buffers_k


def test_mixed_attention_fallback_then_pure_copy_sfa_decode_keeps_arenas_separate():
    manager = make_manager()
    manager.topk_buffers_k[0][:8].fill_(-10)
    saved = manager.topk_buffers_k[0][:8].clone()
    manager.onload_topk_kv = MagicMock(side_effect=lambda *args, **kwargs: manager.resident_buffers_k[0].fill_(3))
    impl = AscendSFAKVOffloadImpl.__new__(AscendSFAKVOffloadImpl)
    impl.layer_name = "layer.0"
    impl.use_fused_copy_sfa = True
    impl.use_fused_overlap = False
    impl.skip_topk = False
    impl._in_graph_runtime = lambda: False
    builder = make_builder()
    cm = common([1, 3], [10, 12], pools=(0, 1), generations=(7, 8))
    populate(builder, cm)
    with patch(MODULE + ".split_decodes_and_prefills", return_value=(1, 1, 1, 2)):
        md = builder._populate_offload_metadata(SimpleNamespace(), cm)
    md.req_ids_tensor = torch.tensor([7, 8])
    md.token_to_req = torch.tensor([0, 1, 1])
    md.block_table = torch.tensor([[0], [1]], dtype=torch.int32)
    q, qr = torch.ones(3, 2, 3), torch.ones(3, 2, 1)
    topk = torch.arange(4).reshape(1, 1, 4).expand(3, 1, 4)

    def resident_attention(impl, query, rope, caches, *args, **kwargs):
        assert caches[0].data_ptr() == manager.resident_buffers_k[0].data_ptr()
        assert kwargs["block_table"].tolist() == [list(range(HOT // 128))]
        return query.new_full(query.shape, 3)

    with (
        patch(MODULE + ".get_sparse_kv_offload_manager", return_value=manager),
        patch(
            MODULE + ".get_ascend_config",
            return_value=SimpleNamespace(sparse_kv_offload_config=manager.sparse_kv_offload_config),
        ),
        patch(MODULE + ".DeviceOperator.execute_sparse_flash_attention_process", side_effect=resident_attention),
        patch(MODULE + ".AscendSFAImpl._execute_sparse_flash_attention_process", return_value=q[1:] * 7),
    ):
        out = impl._execute_sparse_flash_attention_process(
            q, qr, (), topk, md, torch.tensor([1, 3]), torch.tensor([10, 12])
        )
        assert torch.all(out[0] == 3)
        assert torch.all(out[1:] == 7)
        torch.testing.assert_close(manager.topk_buffers_k[0][:8], saved)
        pure = populate(builder, common([1], [11], pools=(0,), generations=(7,)))
        assert pure.lim_request_state.tolist() == [COPY_SFA_REQUEST_STATE_DENSE_RESTORE]
        with patch.object(impl, "_copy_sfa_attention", return_value=q[:1] * 9) as copy_sfa:
            out = impl._execute_sparse_flash_attention_process(
                q[:1], qr[:1], (), topk[:1], pure, torch.tensor([1]), torch.tensor([11])
            )
        assert torch.all(out == 9)
        copy_sfa.assert_called_once()


def test_mixed_decode_refreshes_the_full_tail_before_returning_to_copy_sfa():
    manager = make_manager()
    impl = AscendSFAKVOffloadImpl.__new__(AscendSFAKVOffloadImpl)
    impl.use_fused_copy_sfa = True
    impl.copy_sfa_hot_tokens = HOT
    k = torch.arange(66 * 128 * 3).reshape(66, 128, 1, 3).bfloat16()
    v = torch.arange(66 * 128).reshape(66, 128, 1, 1).bfloat16()
    md = SimpleNamespace(
        num_decodes=1,
        num_prefills=1,
        cum_query_lens=torch.tensor([1, 3]),
        seq_lens=torch.tensor([HOT + 3, 5]),
        block_table=torch.arange(66).reshape(1, 66).expand(2, 66),
        copy_sfa_prefill_pool_slots=torch.tensor([0, 1]),
    )
    with patch(
        MODULE + ".get_ascend_config",
        return_value=SimpleNamespace(sparse_kv_offload_config=manager.sparse_kv_offload_config),
    ):
        impl._copy_prefill_kv_to_copy_sfa_row((k, v), md, manager, "layer.0")
    torch.testing.assert_close(
        manager.topk_buffers_k[0][0, HOT : HOT + 3].float(), k.reshape(-1, 1, 3)[HOT : HOT + 3].bfloat16().float()
    )


@pytest.mark.parametrize("keep_device", [False, True])
def test_pure_decode_keeps_paged_tail_current_for_later_mixed_recovery(keep_device):
    manager = make_manager(keep_device=keep_device)
    manager.tp_rank = 1
    manager.offload_new_kv = MagicMock()
    impl = AscendSFAKVOffloadImpl.__new__(AscendSFAKVOffloadImpl)
    impl.layer_name = "layer.0"
    impl.use_fused_copy_sfa = True
    impl.num_kv_heads = 1
    impl.kv_lora_rank = 3
    impl.qk_rope_head_dim = 1
    impl.copy_sfa_hot_tokens = HOT
    impl._in_graph_runtime = lambda: False
    main_k = torch.zeros((66, 128, 1, 3), dtype=torch.bfloat16)
    main_v = torch.zeros((66, 128, 1, 1), dtype=torch.bfloat16)
    main_k.view(-1, 3)[HOT].fill_(1)
    main_v.view(-1, 1)[HOT].fill_(11)
    manager.topk_buffers_k[0][0, HOT].fill_(1)
    manager.topk_buffers_v[0][0, HOT].fill_(11)
    current_k = torch.full((1, 1, 1, 3), 2, dtype=torch.bfloat16)
    current_v = torch.full((1, 1, 1, 1), 12, dtype=torch.bfloat16)
    impl._compute_kv_only = MagicMock(return_value=(current_k, current_v))
    md = SimpleNamespace(
        attn_state=AscendAttentionState.DecodeOnly,
        num_decodes=1,
        num_prefills=0,
        fused_copy_sfa_enabled=True,
        copy_sfa_device_slots=torch.tensor([HOT + 1]),
    )

    def scatter(key, value, key_cache, value_cache, slots):
        key_cache.view(-1, 3).index_copy_(0, slots, key.reshape(-1, 3))
        value_cache.view(-1, 1).index_copy_(0, slots, value.reshape(-1, 1))

    with (
        patch(MODULE + ".get_sparse_kv_offload_manager", return_value=manager),
        patch(MODULE + ".DeviceOperator.reshape_and_cache", side_effect=scatter) as paged_write,
    ):
        impl.exec_kv(None, None, None, (main_k, main_v) if keep_device else (None, None), torch.tensor([HOT + 1]), md)
    assert paged_write.call_count == int(keep_device)
    assert torch.all(manager.topk_buffers_k[0][0, HOT + 1] == 2)
    if not keep_device:
        return
    # The mixed batch adds a third token through the paged prefill path.
    main_k.view(-1, 3)[HOT + 2].fill_(3)
    main_v.view(-1, 1)[HOT + 2].fill_(13)
    mixed = SimpleNamespace(
        num_decodes=1,
        num_prefills=1,
        cum_query_lens=torch.tensor([1, 3]),
        seq_lens=torch.tensor([HOT + 3, 5]),
        block_table=torch.arange(66).reshape(1, 66).expand(2, 66),
        copy_sfa_prefill_pool_slots=torch.tensor([0, 1]),
    )
    with patch(
        MODULE + ".get_ascend_config",
        return_value=SimpleNamespace(sparse_kv_offload_config=manager.sparse_kv_offload_config),
    ):
        impl._copy_prefill_kv_to_copy_sfa_row((main_k, main_v), mixed, manager, "layer.0")
    torch.testing.assert_close(
        manager.topk_buffers_k[0][0, HOT : HOT + 3, 0, 0], torch.tensor([1, 2, 3], dtype=torch.bfloat16)
    )
    torch.testing.assert_close(
        manager.topk_buffers_v[0][0, HOT : HOT + 3, 0, 0], torch.tensor([11, 12, 13], dtype=torch.bfloat16)
    )
    builder = make_builder()
    cm = common([1, 3], [HOT + 3, 5])
    with patch(MODULE + ".split_decodes_and_prefills", return_value=(1, 1, 1, 2)):
        builder._populate_offload_metadata(SimpleNamespace(), cm)
    resumed = populate(builder, common([1], [HOT + 4]))
    assert resumed.lim_request_state.tolist() == [-2]
    assert resumed.copy_sfa_logical_lens.tolist() == [HOT + 4]
