# SPDX-License-Identifier: Apache-2.0
"""Native LIM/Copy-SFA recovery across MTP and colocate layout transitions."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch_npu  # noqa: F401

from tests.e2e.nightly.single_node.ops.singlecard_ops.test_copy_sfa_offload_metadata import (
    common,
    make_builder,
    populate,
)
from vllm_ascend.attention.sfa_kv_offload import AscendSFAKVOffloadImpl
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.copy_sfa_topk_slots import (
    COPY_SFA_REQUEST_STATE_DENSE_RESTORE,
)
from vllm_ascend.utils import enable_custom_op

HOT = 8192
STRIDE = HOT + 256
MODULE = "vllm_ascend.attention.sfa_kv_offload"


def make_serving_state(quantized=False):
    device = "npu:0"
    builder = make_builder(num_mtp_layers=1)
    impl = AscendSFAKVOffloadImpl.__new__(AscendSFAKVOffloadImpl)
    impl.has_indexer = True
    impl.skip_topk = False
    impl.lim_indexer_owner = impl
    impl.kv_lora_rank = 512
    impl.qk_rope_head_dim = 64
    impl.scale = 1.0
    impl.lim_key_scale = None
    impl.lim_query_scale = None
    impl.lim_reuse_request_count = 0
    impl.enable_sparse_li_c8 = quantized
    impl.n_head = 32
    impl.lim_slot_map = torch.full((8, 16384), -(1 << 31), dtype=torch.int32, device=device)
    impl.lim_topk_src = torch.zeros((16, 1, 2048), dtype=torch.int32, device=device)
    impl.lim_topk_dst = torch.zeros_like(impl.lim_topk_src)
    impl.lim_topk_misses = torch.zeros(16, dtype=torch.int32, device=device)
    impl.lim_miss_src = torch.zeros((4, 32768), dtype=torch.int32, device=device)
    impl.lim_miss_dst = torch.zeros_like(impl.lim_miss_src)
    impl.lim_misses = torch.zeros(4, dtype=torch.int32, device=device)
    impl.copy_sfa_reuse_logical_lens = torch.zeros(4, dtype=torch.int32, device=device)
    impl.copy_sfa_reuse_cache_tokens = torch.zeros_like(impl.copy_sfa_reuse_logical_lens)
    manager = SimpleNamespace(
        topk_buffers_k=[torch.zeros((8, STRIDE, 1, 512), dtype=torch.bfloat16, device=device)],
        topk_buffers_v=[torch.zeros((8, STRIDE, 1, 64), dtype=torch.bfloat16, device=device)],
        k_caches_cpu=[torch.randn((128, 128, 1, 512), dtype=torch.bfloat16, device=device)],
        v_caches_cpu=[torch.randn((128, 128, 1, 64), dtype=torch.bfloat16, device=device)],
        _get_offload_layer_id=lambda _: 0,
        ensure_copy_sfa_host_visibility=lambda **kwargs: None,
    )
    index_cache = torch.zeros((128, 128, 1, 128), dtype=torch.bfloat16, device=device)
    index_cache.view(-1, 128)[:, 0] = torch.arange(16384, device=device) / HOT
    if quantized:
        index_cache, scale = torch_npu.npu_dynamic_quant(index_cache.view(-1, 128))
        index_cache = index_cache.reshape(128, 128, 1, 128)
        impl.lim_key_scale = scale.to(torch.float16).reshape(128, 128, 1)
    return builder, impl, manager, index_cache


def execute(impl, manager, index_cache, md, *, current_delta=0):
    device = manager.topk_buffers_k[0].device
    count = md.num_decode_tokens
    ends = md.copy_sfa_query_ends
    widths = torch.diff(torch.cat((ends.new_zeros(1), ends)))
    positions = torch.arange(count, device=device)
    rows = torch.searchsorted(ends, positions, right=True).long()
    starts = torch.cat((ends.new_zeros(1), ends[:-1]))
    source_tokens = md.copy_sfa_seq_lens[rows] - widths[rows] + positions - starts[rows]
    for destination, host in zip(
        (manager.topk_buffers_k[0], manager.topk_buffers_v[0]), (manager.k_caches_cpu[0], manager.v_caches_cpu[0])
    ):
        destination.view(-1, destination.shape[-1]).index_copy_(
            0, md.copy_sfa_device_slots, host.view(-1, host.shape[-1])[source_tokens.long()] + current_delta
        )
    index_query = torch.zeros((count, 32, 128), dtype=torch.bfloat16, device=device)
    index_query[:, :, 0] = 1
    weights = torch.ones((count, 32), dtype=torch.bfloat16, device=device)
    impl._copy_sfa_metadata = md
    if impl.enable_sparse_li_c8:
        index_query, scale = torch_npu.npu_dynamic_quant(index_query.reshape(-1, 128))
        topk = impl._lim_select(
            index_query,
            weights,
            index_cache,
            md.copy_sfa_source_block_table,
            query_scale=scale.to(torch.float16),
            key_scale=impl.lim_key_scale,
        )
    else:
        topk = impl._lim_select(index_query, weights, index_cache, md.copy_sfa_source_block_table)
    query = torch.zeros((count, 8, 512), dtype=torch.bfloat16, device=device)
    rope = torch.zeros((count, 8, 64), dtype=torch.bfloat16, device=device)
    return impl._copy_sfa_attention(query, rope, topk, md, manager, "mtp.attn")


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("after_fallback", [False, True])
@pytest.mark.parametrize("quantized", [False, True], ids=["bf16", "int8"])
def test_sparse_mtp_bank_returns_to_dense_without_target_prefix_rollback(graph, after_fallback, quantized):
    enable_custom_op()
    torch.manual_seed(131)
    builder, impl, manager, index_cache = make_serving_state(quantized)
    previous = common([1], [8193], pools=(0,), generations=(7,))
    previous.max_query_len = 1
    previous.copy_sfa_restore_tails = False
    sparse = populate(builder, previous, draft_index=2, reuse_topk=False)
    assert sparse.lim_request_state.cpu().tolist() == [-2]
    execute(impl, manager, index_cache, sparse)
    torch.npu.synchronize()
    assert impl.lim_slot_map[0, 0].item() != 0

    if after_fallback:
        # The fallback planner owns private storage. Prefill can write the
        # paged main cache while this bank retains its sparse v2 row.
        with patch(MODULE + ".split_decodes_and_prefills", return_value=(1, 1, 1, 3)):
            builder._populate_offload_metadata(SimpleNamespace(), previous, draft_index=0)
    current = common([4], [8195], pools=(0,), generations=(7,))
    current.copy_sfa_restore_tails = False
    md = populate(builder, current, draft_index=0, reuse_topk=False)
    assert md.lim_request_state.cpu().tolist() == [COPY_SFA_REQUEST_STATE_DENSE_RESTORE]
    assert md.lim_cache_tokens.cpu().tolist() == [8191]
    manager.k_caches_cpu[0].view(-1, 512)[8191].fill_(5)
    # Current query KV must survive prefix recovery even before Host writeback.
    expected_k = manager.k_caches_cpu[0].view(-1, 1, 512)[:8195].clone()
    expected_v = manager.v_caches_cpu[0].view(-1, 1, 64)[:8195].clone()
    expected_k[8191:].add_(7)
    expected_v[8191:].add_(7)

    if graph:
        execute(impl, manager, index_cache, md, current_delta=7)
        captured = torch.npu.NPUGraph()
        with torch.npu.graph(captured):
            output = execute(impl, manager, index_cache, md, current_delta=7)
        # Replay must repair a permuted prefix using only device inputs.
        manager.topk_buffers_k[0][0, :8191].fill_(-7)
        captured.replay()
    else:
        output = execute(impl, manager, index_cache, md, current_delta=7)
    torch.npu.synchronize()
    torch.testing.assert_close(manager.topk_buffers_k[0][0, :8195], expected_k, rtol=0, atol=0)
    torch.testing.assert_close(manager.topk_buffers_v[0][0, :8195], expected_v, rtol=0, atol=0)
    for row in range(4):
        visible = 8192 + row
        expected = expected_k[:visible, 0].float().mean(dim=0)
        torch.testing.assert_close(output[row].float(), expected.expand(8, -1), rtol=0.02, atol=0.003)
    assert impl.lim_misses[0].item() == 8191
    assert impl.copy_sfa_reuse_cache_tokens[0].item() == 0
    if graph:
        md.lim_request_state.fill_(-3)
        md.lim_cache_tokens.zero_()
        captured.replay()
        torch.npu.synchronize()
        assert impl.lim_misses[0].item() == 0
