# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Completed Host frontiers survive V1 target/draft metadata rebuilding."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from vllm.config import CUDAGraphMode

from tests.ut.attention.test_copy_sfa_fallback import make_manager
from tests.ut.attention.test_copy_sfa_metadata import common, make_builder, populate
from vllm_ascend.attention.sfa_cache_mapping import map_sfa_indexer_slots
from vllm_ascend.attention.sfa_kv_offload import AscendAttentionState, AscendSFAKVOffloadImpl
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer

MODULE = "vllm_ascend.attention.sfa_kv_offload"
PROPOSER_MODULE = "vllm_ascend.spec_decode.llm_base_proposer"


def delayed_common(ends, lengths, prefixes, *, pools=(1, 0), generations=(11, 12)):
    metadata = common(ends, lengths, pools, generations)
    metadata.req_topk_buffer_stable_prefixes = torch.tensor(prefixes, dtype=torch.int32)
    metadata.copy_sfa_restore_tails = False
    return metadata


@pytest.mark.parametrize("draft_index", [None, 0, 1, 2])
def test_geometry_uses_completed_prefix_even_after_computed_boundary(draft_index):
    builder = make_builder()
    builder.delayed_d2h = True
    cm = delayed_common([4, 8], [8200, 10372], [8064, 10240])
    md = populate(builder, cm, draft_index=draft_index)
    # Computed 8196 crossed hot, but the last dense block is still being
    # copied. Target and draft must retain dense addressing until retire.
    assert md.copy_sfa_prefix_lens.tolist() == [8064, 10240]
    assert md.copy_sfa_cache_tokens.tolist() == [0, 8192]
    assert md.copy_sfa_logical_lens.tolist() == [8200, 8324]
    assert md.copy_sfa_device_slots[:4].tolist() == [8448 + pos for pos in range(8196, 8200)]
    assert md.copy_sfa_device_slots[4:].tolist() == [8192 + pos % 256 for pos in range(10368, 10372)]
    # No batch-local rounding may publish the in-flight block.
    assert md.copy_sfa_hbm_block_table[0, :].tolist() == list(range(66, 132))


def test_only_retire_snapshot_advances_prefix_and_switches_dense_to_ring():
    builder = make_builder()
    builder.delayed_d2h = True
    cm = delayed_common([4], [8200], [8064], pools=(1,), generations=(11,))
    md = populate(builder, cm)
    prefix_address = md.copy_sfa_prefix_lens.data_ptr()
    cm.seq_lens.add_(4)
    assert populate(builder, cm).copy_sfa_prefix_lens.tolist() == [8064]
    cm.req_topk_buffer_stable_prefixes[0] = 8192
    md = populate(builder, cm)
    assert md.copy_sfa_prefix_lens.data_ptr() == prefix_address
    assert md.copy_sfa_prefix_lens.tolist() == [8192]
    assert md.lim_request_state.tolist() == [-2]
    assert md.copy_sfa_device_slots.tolist() == [8448 + 8192 + pos % 256 for pos in range(8200, 8204)]


def test_target_and_all_draft_steps_keep_one_host_frontier_and_reuse_lengths():
    builder = make_builder()
    builder.delayed_d2h = True
    cm = delayed_common([4], [10492], [10240], pools=(1,), generations=(11,))
    target = populate(builder, cm)
    target_prefix_address = target.copy_sfa_prefix_lens.data_ptr()
    draft0 = populate(builder, cm, draft_index=0)
    saved = draft0.copy_sfa_reuse_logical_lens.clone()
    saved_address = draft0.copy_sfa_reuse_logical_lens.data_ptr()
    assert saved.tolist() == [8444]
    for step in (1, 2):
        cm = delayed_common([1], [10492 + step], [10240], pools=(1,), generations=(11,))
        later = populate(builder, cm, draft_index=step, reuse_topk=True)
        assert later.copy_sfa_prefix_lens.tolist() == [10240]
        assert later.lim_request_state.tolist() == [-3]
        assert later.copy_sfa_reuse_logical_lens is None
    assert target.copy_sfa_prefix_lens.data_ptr() == target_prefix_address
    assert target.copy_sfa_prefix_lens.tolist() == [10240]
    assert draft0.copy_sfa_reuse_logical_lens.data_ptr() == saved_address
    torch.testing.assert_close(draft0.copy_sfa_reuse_logical_lens, saved)


def test_reordering_moves_prefix_with_pool_slot_and_generation():
    builder = make_builder()
    builder.delayed_d2h = True
    populate(builder, delayed_common([4, 8], [8200, 10372], [8064, 10240]))
    reordered = delayed_common([4, 8], [10372, 8200], [10240, 8064], pools=(0, 1), generations=(12, 11))
    md = populate(builder, reordered)
    assert md.copy_sfa_prefix_lens.tolist() == [10240, 8064]
    assert md.lim_request_state.tolist() == [-1, -3]


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        torch.tensor([-1]),
        torch.tensor([129]),
        torch.tensor([1 << 32]),
        torch.tensor([128.0]),
        torch.tensor([[128]]),
        torch.tensor([], dtype=torch.int32),
        torch.empty(1, dtype=torch.int32, device="meta"),
    ],
)
def test_active_rows_require_valid_host_prefixes(invalid):
    builder = make_builder()
    builder.delayed_d2h = True
    cm = delayed_common([4], [8200], [8064], pools=(1,), generations=(11,))
    cm.req_topk_buffer_stable_prefixes = invalid
    with pytest.raises((ValueError, RuntimeError), match="prefix"):
        populate(builder, cm)


def test_dummy_capture_uses_private_rows_and_zero_prefix_without_live_owners():
    builder = make_builder()
    builder.delayed_d2h = True
    cm = delayed_common([4, 8], [999999, 999999], [-1, -1], generations=(-1, -1))
    cm.offload_dummy = True
    cm.req_topk_buffer_stable_prefixes = None
    md = populate(builder, cm)
    assert md.copy_sfa_prefix_lens.tolist() == [0, 0]
    assert md.copy_sfa_logical_lens.tolist() == [0, 0]
    assert md.copy_sfa_device_slots.min() >= builder.copy_sfa_pool_capacity * 8448


def make_real_common():
    lengths = torch.tensor([8200, 10372], dtype=torch.int32)
    main_table = torch.arange(256, dtype=torch.int32).view(2, -1)
    return AscendCommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 4, 8], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 4, 8], dtype=torch.int32),
        seq_lens=lengths,
        seq_lens_cpu=lengths.clone(),
        _seq_lens_cpu=lengths.clone(),
        num_reqs=2,
        num_actual_tokens=8,
        num_input_tokens=8,
        max_query_len=4,
        max_seq_len=10372,
        block_table_tensor=main_table,
        sfa_cache_group_tables=(main_table + 512, main_table, main_table + 1024),
        sfa_host_group_id=1,
        slot_mapping=torch.arange(8),
        positions=torch.arange(8),
        req_topk_buffer_slots=torch.tensor([1, 0], dtype=torch.int32),
        req_topk_buffer_generations=torch.tensor([11, 12]),
        req_topk_buffer_stable_prefixes=torch.tensor([8064, 10240], dtype=torch.int32),
        token_to_req=torch.tensor([0] * 4 + [1] * 4),
    )


def test_unpadded_slices_frontiers_with_request_ownership():
    cm = make_real_common()
    sliced = cm.unpadded(num_actual_tokens=4, num_actual_reqs=1)
    assert sliced.req_topk_buffer_stable_prefixes.tolist() == [8064]
    assert sliced.req_topk_buffer_slots.tolist() == [1]
    assert sliced.req_topk_buffer_generations.tolist() == [11]
    assert cm.req_topk_buffer_stable_prefixes.tolist() == [8064, 10240]
    assert len(sliced.sfa_cache_group_tables) == 3
    assert all(table.shape == (1, 128) for table in sliced.sfa_cache_group_tables)
    assert sliced.sfa_cache_group_tables[0].data_ptr() == cm.sfa_cache_group_tables[0].data_ptr()
    assert sliced.sfa_host_group_id == 1


@pytest.mark.parametrize("padded", [False, True])
def test_proposer_rebuild_preserves_frontier_while_rejecting_tokens(padded):
    cm = make_real_common()
    proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
    proposer.runner = SimpleNamespace(
        actual_seq_lengths_q=[4, 8], attn_state=AscendAttentionState.SpecDecoding, decode_token_per_req=4
    )
    proposer.token_arange_np = np.arange(16)
    proposer.arange = torch.arange(16)
    if padded:
        with patch(PROPOSER_MODULE + ".HAS_TRITON", False):
            rebuilt, _, sampled, rejected = proposer.prepare_inputs_padded(
                cm, SimpleNamespace(cu_num_draft_tokens=torch.tensor([3, 6])), torch.tensor([1, 2])
            )
        assert rejected.tolist() == [3, 2]
        assert sampled.tolist() == [0, 5]
    else:
        rebuilt, tokens = proposer.prepare_inputs(cm, [[1], [2, 3]], [3, 3])
        assert tokens.tolist() == [0, 4, 5]
        assert rebuilt.query_start_loc_cpu.tolist() == [0, 1, 3]
    assert rebuilt.req_topk_buffer_stable_prefixes is cm.req_topk_buffer_stable_prefixes
    assert rebuilt.req_topk_buffer_slots is cm.req_topk_buffer_slots
    assert rebuilt.sfa_cache_group_tables is cm.sfa_cache_group_tables
    assert rebuilt.sfa_host_group_id == 1
    for step in range(3):
        draft = proposer.shallow_copy_metadata(rebuilt)
        draft.copy_sfa_draft_index = step
        assert draft.req_topk_buffer_stable_prefixes.tolist() == [8064, 10240]


@pytest.mark.parametrize("graph", [CUDAGraphMode.NONE, CUDAGraphMode.FULL])
@pytest.mark.parametrize("reuse_topk", [False, True])
@pytest.mark.parametrize("delayed", [False, True])
def test_real_proposer_max_length_mask_uses_private_rows_without_negative_lengths(graph, reuse_topk, delayed):
    builder = make_builder()
    builder.delayed_d2h = delayed
    builder.offload_shared_indexer_only = True
    builder.lim_reuse_topk = reuse_topk
    cm = make_real_common()
    cm.seq_lens[:] = torch.tensor([10496, 10490])
    cm._seq_lens_cpu = cm.seq_lens.clone()
    cm.seq_lens_cpu = cm.seq_lens.clone()
    cm.req_topk_buffer_stable_prefixes[:] = 10368
    proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
    proposer.runner = SimpleNamespace(
        sparse_kv_offload_enabled=True, sparse_kv_offload_config=SimpleNamespace(use_fused_copy_sfa=True)
    )
    proposer.arange = torch.arange(17, dtype=torch.int32)
    proposer.token_arange_np = np.arange(17)
    proposer.uses_mrope, proposer.has_gdn, proposer.use_compress = False, False, False
    proposer.method, proposer.max_model_len, proposer.block_size, proposer.draft_window_size = "mtp", 10496, 128, None
    proposer.slot_mapping_group = [torch.zeros(8, dtype=torch.int32) for _ in range(3)]
    proposer.seq_lens_group = [torch.zeros(8, dtype=torch.int32) for _ in range(3)]
    proposer.query_start_loc_group = [torch.zeros(9, dtype=torch.int32) for _ in range(3)]

    class Group:
        def get_metadata_builder(self):
            return SimpleNamespace(
                build_for_drafting=lambda cm, index: populate(builder, cm, index, reuse_topk=reuse_topk)
            )

    updated, md = proposer.attn_update_stack_num_spec_norm(
        1, cm, 2, 8, torch.tensor([10495, 10489]), graph, attn_group=Group()
    )
    assert updated.seq_lens.tolist() == [1, 10491]
    assert updated.slot_mapping[:2].tolist() == [-1, 10490 + 16384]
    assert md.copy_sfa_prefix_lens.tolist() == [0, 10368]
    assert md.copy_sfa_logical_lens.tolist() == [0, 8315]
    assert md.copy_sfa_pool_entries.tolist() == [builder.copy_sfa_pool_capacity, 0]
    assert md.copy_sfa_device_slots[0] >= builder.copy_sfa_pool_capacity * 8448
    assert md.copy_sfa_device_slots[1] < builder.copy_sfa_pool_capacity * 8448
    assert torch.all(md.copy_sfa_device_slots[2:] >= builder.copy_sfa_pool_capacity * 8448)
    assert md.copy_sfa_hbm_block_table[0].min() >= builder.copy_sfa_pool_capacity * 66
    assert md.sfa_cache_group_tables is cm.sfa_cache_group_tables
    indexer_slots = map_sfa_indexer_slots(
        md.sfa_cache_group_tables[0], updated.query_start_loc[1:], updated.seq_lens, updated.slot_mapping, 128
    )
    assert indexer_slots.tolist() == [-1, 10490 + 16384 + 512 * 128, *([-1] * 6)]
    if not reuse_topk:
        # A later valid pass must not inherit a false "ready" history from
        # the ignored draft, including when the generation is unchanged.
        assert builder.lim_last_cache[1, 1] == -1
        restored = delayed_common([1], [10491], [10368], pools=(1,), generations=(11,))
        assert populate(builder, restored, draft_index=2).lim_request_state.tolist() == [-2]


def test_full_reuse_ignored_draft_masks_call_lengths_without_mutating_saved_selection():
    builder = make_builder()
    builder.delayed_d2h = True
    cm = delayed_common([1, 2], [1, 10491], [10368, 10368])
    cm.slot_mapping[0] = -1
    md = populate(builder, cm, draft_index=1, reuse_topk=True)
    manager = make_manager(keep_device=False)
    manager.ensure_copy_sfa_host_visibility = lambda **_kwargs: None
    manager.k_caches_cpu = [torch.zeros((128, 128, 1, 3), dtype=torch.bfloat16)]
    manager.v_caches_cpu = [torch.zeros((128, 128, 1, 1), dtype=torch.bfloat16)]
    impl = AscendSFAKVOffloadImpl.__new__(AscendSFAKVOffloadImpl)
    impl.delayed_d2h, impl.skip_topk, impl.has_indexer = True, True, True
    impl.lim_indexer_owner = impl
    impl.kv_lora_rank, impl.qk_rope_head_dim, impl.scale = 3, 1, 1.0
    impl._in_graph_runtime = lambda: False
    impl.copy_sfa_reuse_logical_lens = torch.tensor([8319, 8310], dtype=torch.int32)
    impl.copy_sfa_reuse_cache_tokens = torch.tensor([8192, 8192], dtype=torch.int32)
    for name in (
        "lim_reuse_topk_misses",
        "lim_reuse_misses",
        "lim_topk_dst",
        "lim_topk_src",
        "lim_miss_src",
        "lim_miss_dst",
    ):
        setattr(impl, name, torch.zeros(2, dtype=torch.int32))
    query, rope = torch.zeros(2, 1, 3), torch.zeros(2, 1, 1)
    with (
        patch(MODULE + ".prepare_copy_sfa_queries", return_value=(query, rope)),
        patch(MODULE + ".torch.ops._C_ascend.npu_fused_scatter_copy_sparse_flash_attention", create=True) as attention,
    ):
        impl._copy_sfa_attention(query, rope, None, md, manager, "layer.0")
    assert attention.call_args.args[3].tolist() == [0, 8310]
    assert impl.copy_sfa_reuse_logical_lens.tolist() == [8319, 8310]


@pytest.mark.parametrize("delayed,fused", [(True, True), (False, True), (True, False), (False, False)])
@pytest.mark.parametrize("draft_index", [None, 0, 1, 2])
def test_only_delayed_fused_target_and_draft_skip_per_layer_host_write(delayed, fused, draft_index):
    manager = make_manager(keep_device=False)
    manager.tp_rank = 0
    manager.k_caches_cpu = [torch.empty(1)]
    manager.v_caches_cpu = [torch.empty(1)]
    manager.offload_new_kv = MagicMock()
    impl = AscendSFAKVOffloadImpl.__new__(AscendSFAKVOffloadImpl)
    impl.layer_name = "layer.0"
    impl.use_fused_copy_sfa = True
    impl.delayed_d2h = delayed
    impl._in_graph_runtime = lambda: False
    k = torch.full((1, 1, 1, 3), 7, dtype=torch.bfloat16)
    v = torch.full((1, 1, 1, 1), 8, dtype=torch.bfloat16)
    impl._compute_kv_only = lambda *_args: (k, v)
    slot = 8192 + 17 if draft_index is not None else 17
    md = SimpleNamespace(
        attn_state=AscendAttentionState.DecodeOnly,
        num_decodes=1,
        num_prefills=0,
        fused_copy_sfa_enabled=fused,
        copy_sfa_device_slots=torch.tensor([slot]),
        copy_sfa_draft_index=draft_index,
    )
    with patch(MODULE + ".get_sparse_kv_offload_manager", return_value=manager):
        actual_v, actual_k = impl.exec_kv(None, None, None, (None, None), torch.tensor([999]), md)
    assert actual_k is k and actual_v is v
    assert manager.offload_new_kv.call_count == int(not (delayed and fused))
    if fused:
        assert torch.all(manager.topk_buffers_k[0][0, slot] == 7)
        assert torch.all(manager.topk_buffers_v[0][0, slot] == 8)
