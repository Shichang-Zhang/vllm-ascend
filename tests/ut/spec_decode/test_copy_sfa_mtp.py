# SPDX-License-Identifier: Apache-2.0
"""MTP compaction keeps logical TopK and device residency outputs aligned."""

import copy
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch
from torch import nn
from vllm.config import CUDAGraphMode

from vllm_ascend.attention.sfa_kv_offload import AscendSFAKVOffloadImpl
from vllm_ascend.ops.mla import AscendMultiHeadLatentAttention
from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer
from vllm_ascend.spec_decode.utils import compact_mtp_topk_indices


def make_impl(offset):
    impl = AscendSFAKVOffloadImpl.__new__(AscendSFAKVOffloadImpl)
    impl.lim_indexer_owner = impl
    impl.lim_reuse_request_count = 2
    impl.lim_topk_src = torch.arange(32).reshape(8, 1, 4).int() + offset
    impl.lim_topk_dst = impl.lim_topk_src + 100
    return impl


def test_shared_logical_topk_is_compacted_once_and_each_lim_owner_is_compacted():
    logical = torch.arange(32).reshape(8, 1, 4).int()
    saved_logical = logical.clone()
    impls = [make_impl(0), make_impl(1000)]
    saved_src = [impl.lim_topk_src.clone() for impl in impls]
    saved_dst = [impl.lim_topk_dst.clone() for impl in impls]
    modules = [
        SimpleNamespace(topk_indices_buffer=logical, compact_lim_topk_metadata=impl.compact_lim_topk_metadata)
        for impl in impls
    ]
    slots = torch.tensor([6, 1], dtype=torch.int32)
    draft = MagicMock()
    compact_mtp_topk_indices(draft, slots, 8, lim_topk_compactors=modules)
    torch.testing.assert_close(logical[:2], saved_logical[slots])
    for impl, src, dst in zip(impls, saved_src, saved_dst):
        torch.testing.assert_close(impl.lim_topk_src[:2], src[slots])
        torch.testing.assert_close(impl.lim_topk_dst[:2], dst[slots])
    draft.compact_topk_indices.assert_not_called()


def test_shared_attention_follower_does_not_compact_its_owners_outputs_twice():
    follower = AscendSFAKVOffloadImpl.__new__(AscendSFAKVOffloadImpl)
    follower.lim_indexer_owner = make_impl(0)
    before = follower.lim_indexer_owner.lim_topk_src.clone()
    follower.compact_lim_topk_metadata(torch.tensor([6, 1]))
    torch.testing.assert_close(follower.lim_indexer_owner.lim_topk_src, before)


def test_mla_wrapper_forwards_topk_buffer_and_skip_flag_to_current_vllm_layer():
    wrapper = AscendMultiHeadLatentAttention.__new__(AscendMultiHeadLatentAttention)
    nn.Module.__init__(wrapper)
    impl = SimpleNamespace(skip_topk=False, topk_indices_buffer=None, use_fused_copy_sfa=True)
    wrapper.mla_attn = SimpleNamespace(impl=impl)
    shared = torch.empty((8, 1, 4), dtype=torch.int32)
    wrapper.topk_indices_buffer = shared
    wrapper.skip_topk = True
    assert wrapper.mla_attn.topk_indices_buffer is shared
    assert impl.topk_indices_buffer is shared
    assert impl.skip_topk
    assert wrapper.mla_attn.skip_topk
    assert wrapper.uses_lim_topk_metadata


def test_full_graph_draft_uses_request_rows_without_expanding_to_verify_token_padding():
    builder = MagicMock()
    builder.build_for_drafting.return_value = None
    group = SimpleNamespace(get_metadata_builder=lambda: builder)
    proposer = SimpleNamespace(
        runner=SimpleNamespace(
            sparse_kv_offload_enabled=True, sparse_kv_offload_config=SimpleNamespace(use_fused_copy_sfa=True)
        ),
        shallow_copy_metadata=copy.copy,
        arange=torch.arange(17, dtype=torch.int32),
        token_arange_np=np.arange(17),
        uses_mrope=False,
        method="mtp",
        max_model_len=32768,
        has_gdn=False,
        block_size=128,
        draft_window_size=None,
        use_compress=False,
        slot_mapping_group=[torch.zeros(8, dtype=torch.int32) for _ in range(3)],
        seq_lens_group=[torch.zeros(8, dtype=torch.int32) for _ in range(3)],
        query_start_loc_group=[torch.zeros(9, dtype=torch.int32) for _ in range(3)],
    )
    cm = SimpleNamespace(
        num_reqs=2,
        block_table_tensor=torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32),
        seq_lens=torch.tensor([100, 200]),
        seq_lens_cpu=None,
        _seq_lens_cpu=torch.tensor([100, 200]),
        num_computed_tokens_cpu=None,
        positions=torch.zeros(8, dtype=torch.int64),
        query_start_loc=torch.tensor([0, 4, 8]),
        query_start_loc_cpu=torch.tensor([0, 4, 8]),
        req_topk_buffer_slots=torch.tensor([1, 0]),
        req_topk_buffer_generations=torch.tensor([11, 12]),
        copy_sfa_restore_tails=True,
    )
    result, _ = AscendSpecDecodeBaseProposer.attn_update_stack_num_spec_norm(
        proposer,
        1,
        cm,
        2,
        8,
        torch.tensor([99, 199]),
        CUDAGraphMode.FULL,
        attn_group=group,
    )
    assert result.num_reqs == 2
    assert result.num_input_tokens == 8
    assert result.query_start_loc_cpu.tolist() == [0, 1, 2]
    assert result.seq_lens.tolist() == [101, 201]
    assert result.slot_mapping[2:].tolist() == [-1] * 6
    assert result.req_topk_buffer_slots is cm.req_topk_buffer_slots
    assert not result.copy_sfa_restore_tails
