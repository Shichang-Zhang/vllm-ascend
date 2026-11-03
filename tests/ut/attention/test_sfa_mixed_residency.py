# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config import CUDAGraphMode

from tests.ut.attention.test_copy_sfa_fallback import make_manager
from tests.ut.attention.test_copy_sfa_metadata import common, make_builder, populate
from tests.ut.kv_offload.test_sfa_mixed_cache_layout import DRAFT, INDEXER, MEMBER, OWNER, fixture_layout
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.sfa_cache_mapping import map_sfa_indexer_slots
from vllm_ascend.attention.sfa_kv_offload import AscendSFAKVOffloadImpl, AscendSFAOffloadMetadata
from vllm_ascend.attention.sfa_v1 import AscendSFAImpl, PreprocessType
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sfa_layer_residency import (
    bind_sfa_cache_groups,
    build_sfa_cache_layout,
)
from vllm_ascend.spec_decode.eagle_proposer import AscendEagleProposer
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

MODULE = "vllm_ascend.attention.sfa_kv_offload"
BASE = "vllm_ascend.attention.sfa_v1"


def metadata():
    main_table = torch.tensor([[1, 4], [2, 3]], dtype=torch.int32)
    indexer_table = torch.tensor([[7, 8], [9, 10]], dtype=torch.int32)
    host_table = torch.tensor([[11, 12], [13, 14]], dtype=torch.int32)
    return AscendSFAOffloadMetadata(
        num_actual_tokens=3,
        num_input_tokens=4,
        num_decodes=2,
        num_decode_tokens=3,
        slot_mapping=torch.tensor([255, 512, 260, -1]),
        seq_lens=torch.tensor([129, 5], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([999, 999]),  # Deliberately stale.
        cum_query_lens=torch.tensor([2, 3], dtype=torch.int32),
        block_table=main_table,
        block_size=128,
        sin=torch.ones(4, 1),
        cos=torch.ones(4, 1),
        attn_state=AscendAttentionState.SpecDecoding,
        fused_copy_sfa_enabled=True,
        sfa_cache_group_tables=(indexer_table, main_table, host_table),
        copy_sfa_source_block_table=host_table,
    )


def impl():
    layer = AscendSFAKVOffloadImpl.__new__(AscendSFAKVOffloadImpl)
    layer.offload_shared_indexer_only = True
    layer.sfa_indexer_group_id = 0
    layer.block_size = 128
    layer.lim_full_resident = True
    layer.lim_share_members = (object(),)
    return layer


def test_indexer_slots_use_distinct_physical_ids_and_current_device_lengths_without_sync():
    layer, md = impl(), metadata()
    with patch.object(torch.Tensor, "item", side_effect=AssertionError("device scalar read")):
        idx = layer._get_sfa_indexer_metadata(md)
    assert idx.slot_mapping.tolist() == [1023, 1024, 1156, -1]
    assert idx.block_table.data_ptr() == md.sfa_cache_group_tables[0].data_ptr()
    assert md.slot_mapping.tolist() == [255, 512, 260, -1]
    assert idx.copy_sfa_source_block_table is md.copy_sfa_source_block_table
    # Reusing the same metadata/tensor identities must read updated data.
    md.sfa_cache_group_tables[0][0, 1] = 15
    md.seq_lens[0] += 1
    assert layer._get_sfa_indexer_metadata(md).slot_mapping.tolist() == [1920, 1921, 1156, -1]


def test_ignored_draft_and_token_padding_cannot_overwrite_indexer_blocks():
    md = metadata()
    md.seq_lens[0] = 1
    md.slot_mapping[:2] = -1
    assert impl()._get_sfa_indexer_metadata(md).slot_mapping.tolist() == [-1, -1, 1156, -1]
    md.cum_query_lens[:] = 0
    assert impl()._get_sfa_indexer_metadata(md).slot_mapping.tolist() == [-1] * 4


@pytest.mark.parametrize("fault", ["missing_tables", "missing_binding", "short_rows", "empty_columns"])
def test_incomplete_indexer_geometry_fails_before_kernel(fault):
    layer, md = impl(), metadata()
    if fault == "missing_tables":
        md.sfa_cache_group_tables = None
    elif fault == "missing_binding":
        layer.sfa_indexer_group_id = None
    else:
        table = torch.zeros((1, 2) if fault == "short_rows" else (2, 0), dtype=torch.int32)
        md.sfa_cache_group_tables = (table,)
    with pytest.raises((ValueError, RuntimeError)):
        layer._get_sfa_indexer_metadata(md)


def test_cache_group_binding_uses_real_indexer_identity_and_validates_before_mutation():
    plan, _, config = fixture_layout()
    layout = build_sfa_cache_layout(config, plan)
    implementations = {
        OWNER: SimpleNamespace(has_indexer=True, indexer=SimpleNamespace(k_cache=SimpleNamespace(prefix=INDEXER))),
        MEMBER: SimpleNamespace(has_indexer=False),
        DRAFT: SimpleNamespace(has_indexer=True, indexer=SimpleNamespace(k_cache=SimpleNamespace(prefix="unknown"))),
    }
    with pytest.raises(ValueError, match="missing"):
        bind_sfa_cache_groups(implementations, layout)
    assert not hasattr(implementations[OWNER], "sfa_main_group_id")
    implementations[DRAFT].indexer.k_cache.prefix = INDEXER
    bind_sfa_cache_groups(implementations, layout)
    assert implementations[OWNER].sfa_main_group_id == 1
    assert implementations[OWNER].sfa_indexer_group_id == 0
    assert implementations[DRAFT].sfa_main_group_id == 2
    assert implementations[MEMBER].sfa_indexer_group_id is None


@pytest.mark.parametrize("shared_members", [False, True])
def test_resident_owner_keeps_lim_only_when_members_need_the_plan(shared_members):
    layer, md = impl(), metadata()
    layer.lim_share_members = (object(),) if shared_members else ()
    layer.enable_sparse_li_c8, layer.enable_sparse_sfa_c8 = True, False
    layer._lim_select = MagicMock(return_value="lim")
    caches = (object(), object(), object(), object())
    idx = layer._get_sfa_indexer_metadata(md)
    with patch.object(AscendSFAImpl, "_select_indexer_topk", return_value="li") as ordinary:
        result = layer._select_indexer_topk("query", "scale", "shape", "weights", caches, idx, None, None)
    assert result == "li"
    ordinary.assert_called_once()
    if shared_members:
        assert layer._lim_select.call_args.args[3].data_ptr() == md.sfa_cache_group_tables[0].data_ptr()
        assert layer._lim_select.call_args.kwargs == {"query_scale": "scale", "key_scale": caches[3]}
    else:
        layer._lim_select.assert_not_called()


@pytest.mark.parametrize("prefix", [4, 2], ids=["inline", "delayed"])
def test_resident_attention_includes_tail_and_current_token_while_member_lim_plan_is_preserved(prefix):
    layer, md = impl(), metadata()
    layer.enable_sparse_li_c8 = layer.enable_sparse_sfa_c8 = False
    layer.scale = 1.0
    md.seq_lens = torch.tensor([6])
    md.cum_query_lens = torch.tensor([1])
    md.block_table = torch.tensor([[2]], dtype=torch.int32)
    md.copy_sfa_prefix_lens = torch.tensor([prefix], dtype=torch.int32)
    cold_selection = torch.arange(prefix - 2, prefix, dtype=torch.int32).view(1, 1, 2)
    layer.lim_topk_src = cold_selection.clone()
    layer._lim_select = MagicMock(return_value=layer.lim_topk_src)
    # Logical tokens 4 and 5 (including the current token) are beyond both
    # Host frontiers. An accidental prefix-only selection returns zero output.
    keys = torch.zeros(4, 128, 1, 3, dtype=torch.bfloat16)
    keys[2, 4, 0, 0] = 5
    keys[2, 5, 0, 0] = 10
    values = keys[2, :6, 0].float()
    cache = (keys, torch.zeros(4, 128, 1, 1), torch.zeros(1))
    query = torch.tensor([[[1.0, 0.0, 0.0]]])

    def full_context_li(*args):
        context = int(args[-1][0])
        scores = values[:context] @ query[0, 0]
        return scores.topk(2).indices.to(torch.int32).view(1, 1, 2)

    def cpu_sparse_attention(**kwargs):
        indices = kwargs["sparse_indices"][0, 0].long()
        blocks = kwargs["block_table"][0, indices // 128].long()
        selected = kwargs["key"][blocks, indices % 128, 0].float()
        weights = (selected @ kwargs["query"][0, 0]).softmax(0)
        return (weights[:, None] * selected).sum(0).view(1, 1, 3)

    with (
        patch.object(AscendSFAImpl, "_select_indexer_topk", side_effect=full_context_li),
        patch(
            MODULE + ".torch.ops._C_ascend.npu_sparse_flash_attention", create=True, side_effect=cpu_sparse_attention
        ),
        patch(MODULE + ".get_sparse_kv_offload_manager", side_effect=AssertionError("resident Host access")),
    ):
        selected = layer._select_indexer_topk(query, None, None, None, cache, md, md.cum_query_lens, md.seq_lens)
        actual = layer._execute_sparse_flash_attention_process(
            query, torch.zeros(1, 1, 1), cache, selected, md, md.cum_query_lens, md.seq_lens
        )
    expected = (torch.tensor([5.0, 10.0]).softmax(0)[:, None] * values[4:6]).sum(0).view(1, 1, 3)
    torch.testing.assert_close(actual, expected)
    assert actual[0, 0, 0] > 9
    assert selected.tolist() == [[[5, 4]]]
    assert torch.equal(layer.lim_topk_src, cold_selection)
    layer._lim_select.assert_called_once()


def test_resident_write_and_attention_use_private_paged_cache_without_offload_manager():
    layer, md = impl(), metadata()
    q, qr = torch.ones(4, 2, 4), torch.ones(4, 2, 2)
    cache = (torch.zeros(8, 128, 1, 4), torch.zeros(8, 128, 1, 2))
    topk = torch.tensor([[127, 128], [0, 1], [2, 3]], dtype=torch.int32)
    with (
        patch.object(AscendSFAImpl, "exec_kv", return_value="paged_write") as write,
        patch.object(
            AscendSFAImpl, "_execute_sparse_flash_attention_process", return_value="paged_attention"
        ) as attention,
        patch(
            MODULE + ".get_sparse_kv_offload_manager", side_effect=AssertionError("resident layer queried Host manager")
        ),
    ):
        assert layer.exec_kv(q, md.cos, md.sin, cache, md.slot_mapping, md) == "paged_write"
        result = layer._execute_sparse_flash_attention_process(q, qr, cache, topk, md, md.cum_query_lens, md.seq_lens)
    assert result == "paged_attention"
    assert write.call_args.args[3] is cache
    args = attention.call_args.args
    assert args[2] is cache and args[4] is md
    assert args[3].shape == (4, 1, 2)
    assert args[3][-1].tolist() == [[0, 0]]
    assert args[4].block_table is md.sfa_cache_group_tables[1]


@pytest.mark.parametrize("c8", [False, True])
def test_real_native_forward_writes_indexer_keys_and_scales_with_indexer_slots(c8):
    layer, md = impl(), metadata()
    layer.use_fused_copy_sfa = True
    layer.layer_name = OWNER
    layer.enable_dsa_cp = layer.enable_dsa_cp_with_o_proj_tp = layer.enable_sp = False
    layer.has_indexer = True
    layer.enable_sparse_li_c8, layer.enable_sparse_sfa_c8 = c8, False
    layer.preprocess_type = PreprocessType.NATIVE
    layer.skip_topk, layer.use_index_cache = False, False
    layer.q_lora_rank, layer.kv_lora_rank, layer.qk_rope_head_dim = 1, 4, 2
    key = torch.arange(4 * 128).reshape(4, 128).to(torch.int8 if c8 else torch.bfloat16)
    scale = torch.arange(4).reshape(4, 1).to(torch.float16) if c8 else None
    indexer_cache = (torch.zeros(16, 128, 1, 128, dtype=key.dtype),)
    if c8:
        indexer_cache += (torch.zeros(16, 128, 1, 1, dtype=torch.float16),)
    layer.indexer = SimpleNamespace(k_cache=SimpleNamespace(kv_cache=indexer_cache))
    layer.indexer_select_pre_process = lambda **kwargs: (key, scale)
    layer.fused_qkv_a_proj = lambda hidden: (torch.ones(4, 7),)
    layer.q_a_layernorm = lambda x: x
    layer.exec_kv = lambda *args: (torch.ones(4, 1, 2), torch.ones(4, 1, 4))
    layer._maybe_gather_kv_for_dsacp = lambda pe, nope, qs, li, ls, full: (li, ls, None, [])
    layer._q_proj_and_k_up_proj = lambda q: (torch.ones(4, 2, 4), torch.ones(4, 2, 2))
    layer.rope_single = lambda q, *args: q
    layer._maybe_store_kvcache_for_c8_n_dsacp = lambda pe, nope, qs, li, *args: (pe, nope, li, None, [])
    layer._get_full_kv = lambda value, md: value
    layer.indexer_select_post_process = MagicMock(return_value=torch.zeros(4, 1, 2, dtype=torch.int32))
    layer._execute_sparse_flash_attention_process = MagicMock(return_value=torch.ones(4, 4))
    layer._v_up_proj = lambda x: x
    layer.o_proj = lambda x: (x,)
    cache = (torch.zeros(8, 128, 1, 4), torch.zeros(8, 128, 1, 2))

    def scatter(dst, slots, values):
        active = slots[:, 0] >= 0
        dst.index_copy_(0, slots[:, 0][active], values[active])

    with (
        patch(BASE + ".torch_npu.npu_scatter_nd_update_", side_effect=scatter),
        patch(BASE + ".wait_for_kv_layer_from_connector"),
        patch(BASE + ".notify_kv_cache_written"),
        patch(BASE + ".record_attention_compute_start"),
        patch(BASE + ".maybe_save_kv_layer_to_connector"),
    ):
        layer.forward(OWNER, torch.ones(4, 4), cache, md, output=torch.empty(4, 4))
    slots = torch.tensor([1023, 1024, 1156])
    torch.testing.assert_close(indexer_cache[0].view(-1, 128)[slots], key[:3])
    assert torch.count_nonzero(indexer_cache[0].view(-1, 128)[md.slot_mapping[:3]]) == 0
    if c8:
        torch.testing.assert_close(indexer_cache[1].view(-1, 1)[slots], scale[:3])
    index_md = layer.indexer_select_post_process.call_args.kwargs["attn_metadata"]
    assert index_md.block_table.data_ptr() == md.sfa_cache_group_tables[0].data_ptr()
    assert layer._execute_sparse_flash_attention_process.call_args.args[4] is md


def test_legacy_mode_preserves_original_metadata_and_indexer_fast_path():
    layer, md = impl(), metadata()
    layer.offload_shared_indexer_only = False
    assert layer._get_sfa_indexer_metadata(md) is md
    with patch.object(AscendSFAImpl, "_use_li_c8_reshape_optim", return_value=True):
        assert layer._use_li_c8_reshape_optim()
        layer.offload_shared_indexer_only = True
        assert not layer._use_li_c8_reshape_optim()


def test_mapping_masks_out_of_range_and_negative_blocks_without_unsafe_indexing():
    result = map_sfa_indexer_slots(
        torch.tensor([[-1], [8]]), torch.tensor([1, 2]), torch.tensor([1, 256]), torch.tensor([0, 0]), 128
    )
    assert result.tolist() == [-1, -1]


def test_owner_metadata_builds_tail_restore_and_member_copy_from_explicit_host_group():
    builder = make_builder()
    builder.offload_shared_indexer_only = True
    cm = common([1], [9000], pools=(0,), generations=(1,))
    cm.sfa_cache_group_tables = (cm.block_table_tensor + 300, cm.block_table_tensor + 200, cm.block_table_tensor + 100)
    cm.sfa_host_group_id = 2
    cm.block_table_tensor = cm.sfa_cache_group_tables[1]  # Resident owner table.
    md = populate(builder, cm)
    assert md.copy_sfa_source_block_table.data_ptr() == cm.sfa_cache_group_tables[2].data_ptr()
    assert md.copy_sfa_tail_src.tolist() == [[170 * 128, 171 * 128]]
    assert md.copy_sfa_tail_lengths.tolist() == [[39, 0]]

    manager = make_manager(keep_device=False)
    manager.ensure_copy_sfa_host_visibility = lambda **kwargs: None
    manager.k_caches_cpu = [torch.zeros(256, 128, 1, 3, dtype=torch.bfloat16)]
    manager.v_caches_cpu = [torch.zeros(256, 128, 1, 1, dtype=torch.bfloat16)]
    manager.k_caches_cpu[0][101, 1] = 17
    manager.k_caches_cpu[0][201, 1] = -33  # Poison at the owner's physical ID.
    member = impl()
    member.lim_full_resident, member.skip_topk, member.has_indexer = False, True, False
    owner = SimpleNamespace(
        lim_topk_dst=torch.zeros(1, 1, 2048, dtype=torch.int32),
        lim_topk_src=torch.full((1, 1, 2048), 129, dtype=torch.int32),
        lim_topk_misses=torch.ones(1, dtype=torch.int32),
        lim_miss_src=torch.ones(1, 32768, dtype=torch.int32),
        lim_miss_dst=torch.zeros(1, 32768, dtype=torch.int32),
        lim_misses=torch.ones(1, dtype=torch.int32),
    )
    member.lim_indexer_owner = owner
    member._in_graph_runtime = lambda: False
    member.kv_lora_rank, member.qk_rope_head_dim, member.scale = 3, 1, 1.0

    def consume(*args):
        logical = int(args[6][0, 0, 0])
        physical = int(args[12][0, logical // 128])
        args[-1].copy_(args[16][physical, logical % 128])

    with patch(
        MODULE + ".torch.ops._C_ascend.npu_fused_scatter_copy_sparse_flash_attention", create=True, side_effect=consume
    ):
        out = member._copy_sfa_attention(torch.ones(1, 8, 3), torch.ones(1, 8, 1), None, md, manager, "layer.0")
    assert torch.all(out == 17)


@pytest.mark.parametrize("host_group", [None, -1, 3])
def test_copy_sfa_rejects_missing_or_invalid_host_group_in_mixed_metadata(host_group):
    cm = common([1], [9000], pools=(0,), generations=(1,))
    cm.sfa_cache_group_tables = (cm.block_table_tensor,)
    cm.sfa_host_group_id = host_group
    with pytest.raises(ValueError, match="Host group"):
        populate(make_builder(), cm)


def test_real_runner_metadata_routes_main_groups_and_passes_all_tables_to_mtp():
    _, _, cache_config = fixture_layout()
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.kv_cache_config = cache_config
    runner.sparse_kv_offload_enabled = True
    runner.sparse_kv_offload_config = SimpleNamespace(use_fused_copy_sfa=True, offload_shared_indexer_only=True)
    runner.sparse_kv_offload_manager = SimpleNamespace(
        d2h_host_group_id=2, sfa_cache_layout=SimpleNamespace(host_group_id=2)
    )
    runner._delayed_d2h_runtime = MagicMock()
    runner._offload_pool_slots = SimpleNamespace(cpu=torch.tensor([0, 1, 2]), np=None)
    runner._offload_pool_generations = SimpleNamespace(cpu=torch.tensor([1, 2, -1]), np=None)
    runner._offload_stable_prefixes = SimpleNamespace(cpu=torch.tensor([0, 128, 0]), np=None)
    runner._offload_req_ids_tensor = runner._offload_token_to_req = None
    runner.query_start_loc = SimpleNamespace(gpu=torch.tensor([0, 2, 3, 3]), cpu=torch.tensor([0, 2, 3, 3]))
    runner.seq_lens = torch.tensor([129, 5, 0])
    runner.optimistic_seq_lens_cpu = runner.seq_lens.clone()
    tables = [torch.arange(6, dtype=torch.int32).reshape(3, 2) + offset for offset in (100, 200, 300)]
    block_tables = [
        SimpleNamespace(
            get_device_tensor=lambda table=table: table,
            get_numpy_array=lambda table=table: table.numpy(),
            slot_mapping=SimpleNamespace(gpu=torch.tensor([1, 2, 3, 999])),
        )
        for table in tables
    ]
    runner.input_batch = SimpleNamespace(
        req_ids=["a", "b"],
        block_table=block_tables,
        num_computed_tokens_cpu_tensor=torch.tensor([127, 4, 0]),
        num_prompt_tokens_cpu_tensor=torch.tensor([100, 2, 0]),
    )
    runner.num_spec_tokens = 3
    runner.device_metadata_executor = runner.device_metadata_providers = None
    runner.dcp_size = 1
    runner.use_async_spec_decode = runner.use_compress = runner._has_gdn = runner.is_mm_prefix_lm = False
    runner.model_config = SimpleNamespace(enable_return_routed_experts=False)
    runner.device = torch.device("cpu")
    runner.positions = torch.tensor([127, 128, 4, 0])
    runner.attn_state = AscendAttentionState.SpecDecoding
    runner.actual_seq_lengths_q, runner.decode_token_per_req = [2, 3], 2
    runner.group_len = runner.group_key_idx = runner.group_key_cache_idx = SimpleNamespace(gpu=torch.zeros(3))
    runner.compilation_config = SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE)
    runner.vllm_config = SimpleNamespace(compilation_config=runner.compilation_config)
    runner.speculative_config = SimpleNamespace()
    runner.drafter = AscendEagleProposer.__new__(AscendEagleProposer)
    runner.drafter.attn_layer_names = [DRAFT]
    runner._get_encoder_seq_lens = lambda *args: (None, None)
    runner.attn_groups = [
        [
            SimpleNamespace(
                layer_names=group.layer_names,
                get_metadata_builder=lambda *args: SimpleNamespace(
                    build=lambda **kwargs: kwargs["common_attn_metadata"]
                ),
            )
        ]
        for group in cache_config.kv_cache_groups
    ]
    with patch("vllm_ascend.worker.model_runner_v1.update_sparse_kv_offload_metadata"):
        by_layer, draft_common = runner._build_attention_metadata(
            3,
            2,
            2,
            num_tokens_padded=4,
            num_reqs_padded=3,
            num_scheduled_tokens_np=torch.tensor([2, 1]).numpy(),
        )
    assert by_layer[OWNER].block_table_tensor.data_ptr() == tables[1].data_ptr()
    assert by_layer[MEMBER].block_table_tensor.data_ptr() == tables[2].data_ptr()
    assert draft_common.block_table_tensor.data_ptr() == tables[2].data_ptr()
    assert draft_common.sfa_host_group_id == 2
    assert [table.shape for table in draft_common.sfa_cache_group_tables] == [(2, 2)] * 3
    assert [table.data_ptr() for table in draft_common.sfa_cache_group_tables] == [table.data_ptr() for table in tables]
    assert all(block.slot_mapping.gpu[-1] == -1 for block in block_tables)
    assert all(torch.count_nonzero(table[2]) == 0 for table in tables)
