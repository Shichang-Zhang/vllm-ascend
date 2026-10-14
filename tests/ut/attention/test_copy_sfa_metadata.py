# SPDX-License-Identifier: Apache-2.0
"""CPU regression coverage of Copy-SFA device tensor geometry and LIM lifecycle."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from vllm_ascend.attention.sfa_kv_offload import AscendSFAKVOffloadImpl, AscendSFAKVOffloadMetadataBuilder
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.copy_sfa_topk_slots import (
    COPY_SFA_REQUEST_STATE_DENSE_RESTORE,
    prepare_copy_sfa_dummy_slots,
    prepare_copy_sfa_request_slots,
)

MODULE = "vllm_ascend.attention.sfa_kv_offload"


def make_builder(num_mtp_layers=1):
    builder = AscendSFAKVOffloadMetadataBuilder.__new__(AscendSFAKVOffloadMetadataBuilder)
    builder.use_fused_copy_sfa = True
    builder.decode_threshold = 4
    builder.is_pd_decode_consumer = True
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=2, max_num_batched_tokens=16),
        speculative_config=SimpleNamespace(
            num_speculative_tokens=3,
            draft_model_config=SimpleNamespace(
                hf_config=SimpleNamespace(num_nextn_predict_layers=num_mtp_layers, index_share_for_mtp_iteration=True)
            ),
        ),
        model_config=SimpleNamespace(
            max_model_len=16384, hf_text_config=SimpleNamespace(kv_lora_rank=512, qk_rope_head_dim=64)
        ),
    )
    with patch(
        MODULE + ".get_ascend_config",
        return_value=SimpleNamespace(sparse_kv_offload_config=SimpleNamespace(topk_buffer_size=8192)),
    ):
        builder._init_copy_sfa_metadata_buffers(config, torch.device("cpu"))
    return builder


def common(ends, lengths, pools=(1, 0), generations=(11, 12)):
    count = len(lengths)
    return SimpleNamespace(
        query_start_loc=torch.tensor([0, *ends], dtype=torch.int32, device="cpu"),
        query_start_loc_cpu=torch.tensor([0, *ends], dtype=torch.int32),
        # Deliberately stale mirrors must never affect fused_copy_sfa geometry.
        seq_lens=torch.tensor(lengths, dtype=torch.int32, device="cpu"),
        seq_lens_cpu=torch.full((count,), 777777, dtype=torch.int32),
        _seq_lens_cpu=torch.full((count,), 777777, dtype=torch.int32),
        copy_sfa_draft_index=None,
        # Most geometry tests exercise the optional restore descriptors too.
        copy_sfa_restore_tails=True,
        req_topk_buffer_slots=torch.tensor(pools, dtype=torch.int32),
        req_topk_buffer_generations=torch.tensor(generations, dtype=torch.int64),
        block_table_tensor=torch.arange(count * 128, dtype=torch.int32, device="cpu").reshape(count, 128),
        slot_mapping=torch.arange(16, dtype=torch.int64, device="cpu") + 128,
        req_ids_tensor=None,
        token_to_req=None,
        offload_dummy=False,
        max_query_len=4,
        num_reqs=count,
        num_input_tokens=ends[-1],
    )


def populate(builder, cm, draft_index=None, *, reuse_topk=False):
    cm.copy_sfa_draft_index = draft_index
    builder.lim_reuse_topk = reuse_topk
    metadata = SimpleNamespace(slot_mapping=cm.slot_mapping[: cm.num_input_tokens])
    with patch(MODULE + ".split_decodes_and_prefills", return_value=(cm.num_reqs, 0, cm.num_input_tokens, 0)):
        if draft_index in (None, 0):
            # The proposer uses ordinary build() for draft step 0 too.
            with patch(MODULE + ".AscendSFAMetadataBuilder.build", return_value=metadata):
                metadata = builder.build(0, cm)
        else:
            with patch(MODULE + ".AscendSFAMetadataBuilder.build_for_drafting", return_value=metadata):
                metadata = builder.build_for_drafting(cm, draft_index=draft_index)
    return metadata


def test_prefill_batches_carry_pool_slots():
    builder = make_builder()
    # Prefill batch (num_prefills > 0): fused_copy_sfa_enabled is False, but the row
    # slots must still ride on the metadata for the exec_kv prefill-end D2D.
    cm = common([4], [10371])
    metadata = SimpleNamespace()
    with patch(MODULE + ".split_decodes_and_prefills", return_value=(0, cm.num_reqs, cm.num_input_tokens, 0)):
        builder._populate_offload_metadata(metadata, cm)
    assert metadata.fused_copy_sfa_enabled is False
    assert metadata.copy_sfa_prefill_pool_slots.cpu().tolist() == [1]

    # Decode batch: the per-step tail restore is always skipped now (initial
    # KV arrives via the PD D2D or the colocate prefill-end D2D).
    colocate_builder = make_builder()
    colocate_builder.is_pd_decode_consumer = False
    metadata = populate(colocate_builder, common([4], [10371]))
    assert metadata.fused_copy_sfa_enabled is True
    assert metadata.copy_sfa_prefill_pool_slots.cpu().tolist() == [1]


def test_device_lengths_tail_geometry_and_rejection():
    builder = make_builder()
    cm = common([4, 5], [10371, 8321])
    metadata = populate(builder, cm)
    # [S-Q] are 10367 and 8320. Prefix rounds down to complete 128-token blocks.
    assert metadata.copy_sfa_prefix_lens.cpu().tolist() == [10240, 8320]
    assert metadata.copy_sfa_cache_tokens.cpu().tolist() == [8192, 8192]
    assert metadata.copy_sfa_logical_lens.cpu().tolist() == [8323, 8193]
    # Current query KV is scattered locally: descriptors still describe prior KV
    # so prefix rollback can eager-restore. PD decode skips the graph H2D.
    assert metadata.copy_sfa_tail_lengths.cpu().tolist() == [[127, 0], [0, 0]]
    assert metadata.copy_sfa_copy_count.item() == 8
    assert metadata.copy_sfa_copy_lengths.cpu().tolist() == [127 * 1024, 0, 0, 0, 127 * 128, 0, 0, 0]
    assert metadata.copy_sfa_tail_src.cpu().tolist() == [[10240, 10368], [24704, 24832]]
    assert metadata.copy_sfa_hbm_block_table[:, 64:66].cpu().tolist() == [[130, 131], [65, 64]]
    stride = 8192 + 256
    assert metadata.copy_sfa_device_slots.cpu().tolist() == [
        stride + 8192 + pos % 256 for pos in range(10367, 10371)
    ] + [8192 + 8320 % 256]
    address = metadata.copy_sfa_prefix_lens.data_ptr()
    cm.seq_lens.copy_(torch.tensor([10243, 8321], dtype=torch.int32, device="cpu"))
    revised = populate(builder, cm)
    assert revised.copy_sfa_prefix_lens.data_ptr() == address
    assert revised.copy_sfa_prefix_lens.cpu().tolist() == [10112, 8320]


def test_generation_compaction_and_prefix_rollback_reset():
    builder = make_builder()
    cm = common([4, 8], [10371, 8324])
    metadata = populate(builder, cm)
    assert metadata.lim_request_state.cpu().tolist() == [-2, -2]
    assert populate(builder, cm).lim_request_state.cpu().tolist() == [-1, -1]
    # Swap batch order, keeping request-owned pool and generation together.
    cm = common([4, 8], [8324, 10371], pools=(0, 1), generations=(12, 11))
    metadata = populate(builder, cm)
    assert metadata.lim_request_state.cpu().tolist() == [-1, -1]
    # New generation and rollback independently force cold fill.
    cm.req_topk_buffer_generations[0] = 13
    cm.seq_lens[1] = 10243
    assert populate(builder, cm).lim_request_state.cpu().tolist() == [-2, -2]


def test_builder_histories_are_independent():
    target_builder = make_builder()
    draft_builder = make_builder()
    cm = common([4], [10371], pools=(1,), generations=(11,))
    first = populate(target_builder, cm)
    draft = populate(draft_builder, cm, draft_index=0)
    assert first.lim_request_state.cpu().tolist() == [-2]
    assert draft.lim_request_state.cpu().tolist() == [-2]
    with patch(MODULE + ".split_decodes_and_prefills", return_value=(0, 1, 0, 4)):
        target_builder._populate_offload_metadata(SimpleNamespace(), common([4], [10371]))
    assert populate(target_builder, cm).lim_request_state.cpu().tolist() == [-2]
    assert populate(draft_builder, cm, draft_index=0).lim_request_state.cpu().tolist() == [-1]


def test_target_and_draft_history_are_separate_and_reuse_does_not_advance_it():
    builder = make_builder()
    target = populate(builder, common([4], [10371], pools=(1,), generations=(11,)))
    draft = populate(builder, common([4], [10371], pools=(1,), generations=(11,)), draft_index=0)
    assert target.lim_request_state.cpu().tolist() == [-2]
    assert draft.lim_request_state.cpu().tolist() == [-2]
    assert target.lim_request_state.data_ptr() != draft.lim_request_state.data_ptr()
    # Later Q1 steps cross a block boundary, but skip LIM entirely.
    populate(builder, common([1], [10380], pools=(1,), generations=(11,)), draft_index=1, reuse_topk=True)
    populate(builder, common([1], [10381], pools=(1,), generations=(11,)), draft_index=2, reuse_topk=True)
    assert builder.lim_last_prefix[1, 1].item() == 10240
    assert draft.lim_request_state.cpu().tolist() == [-2]
    next_draft = populate(builder, common([4], [10371], pools=(1,), generations=(11,)), draft_index=0)
    assert next_draft.lim_request_state.cpu().tolist() == [-1]


def test_fallback_resets_only_the_affected_model_history():
    builder = make_builder()
    cm = common([4], [10371], pools=(1,), generations=(11,))
    populate(builder, cm)
    populate(builder, cm, draft_index=0)
    cm.copy_sfa_draft_index = None
    with patch(MODULE + ".split_decodes_and_prefills", return_value=(0, 1, 0, 4)):
        builder._populate_offload_metadata(SimpleNamespace(), cm)
    assert populate(builder, cm).lim_request_state.cpu().tolist() == [-2]
    assert populate(builder, cm, draft_index=0).lim_request_state.cpu().tolist() == [-1]


def test_physical_mtp_layers_have_independent_history():
    builder = make_builder(num_mtp_layers=2)
    for step, expected in ((0, -2), (1, -2), (2, -1)):
        cm = common([1], [10371], pools=(1,), generations=(11,))
        md = populate(builder, cm, draft_index=step, reuse_topk=False)
        assert md.lim_request_state.cpu().tolist() == [expected]


@pytest.mark.parametrize("draft_index", [None, 0, 1, 2], ids=["target", "draft0", "draft1", "draft2"])
@pytest.mark.parametrize("restore_dense", [False, True])
def test_lim_consumes_shared_state_without_modifying_it(draft_index, restore_dense):
    builder = make_builder()
    cm = common([4, 8, 12], [10371, 500, 0], pools=(1, 0, 0), generations=(11, 12, -1))
    populate(builder, cm, draft_index=draft_index)
    populate(builder, cm, draft_index=draft_index)
    if restore_dense:
        populate(
            builder,
            common([4, 8, 12], [10371, 9000, 0], pools=(1, 0, 0), generations=(11, 12, -1)),
            draft_index=draft_index,
        )
    metadata = populate(builder, cm, draft_index=draft_index)
    expected_states = [-1, COPY_SFA_REQUEST_STATE_DENSE_RESTORE if restore_dense else -3, -3]
    assert metadata.lim_request_state.cpu().tolist() == expected_states
    impl = AscendSFAKVOffloadImpl.__new__(AscendSFAKVOffloadImpl)
    impl._copy_sfa_metadata = metadata
    impl.lim_slot_map = torch.empty((8, 16384), dtype=torch.int32, device="cpu")
    for name, shape in (
        ("lim_topk_src", (12, 1, 2048)),
        ("lim_topk_dst", (12, 1, 2048)),
        ("lim_topk_misses", (12,)),
        ("lim_miss_src", (3, 32768)),
        ("lim_miss_dst", (3, 32768)),
        ("lim_misses", (3,)),
        ("copy_sfa_reuse_logical_lens", (3,)),
        ("copy_sfa_reuse_cache_tokens", (3,)),
    ):
        setattr(impl, name, torch.full(shape, -999, dtype=torch.int32, device="cpu"))
    impl.lim_reuse_request_count = 0
    impl.lim_key_scale = None
    indexer = SimpleNamespace(
        k_cache=SimpleNamespace(kv_cache=[torch.empty((1, 128, 1, 128), dtype=torch.bfloat16, device="cpu")])
    )
    query = torch.empty((12, 32, 128), dtype=torch.bfloat16, device="cpu")
    weights = torch.empty((12, 32), dtype=torch.bfloat16, device="cpu")
    with patch(MODULE + ".torch.ops._C_ascend.npu_fused_lightning_indexer_manage", create=True) as lim:
        impl._lim_select(query, weights, indexer.k_cache.kv_cache[0], cm.block_table_tensor)
        assert lim.call_args.args[10].data_ptr() == metadata.lim_request_state.data_ptr()
        assert lim.call_args.args[9].data_ptr() == metadata.lim_cache_tokens.data_ptr()
        assert metadata.lim_request_state.cpu().tolist() == expected_states
        if restore_dense:
            assert lim.call_args.args[9][1] == 496
    bank = 0 if draft_index is None else 1
    assert builder.lim_last_generation[bank, 1].item() == 11
    if draft_index == 0:
        assert impl.lim_reuse_request_count == 3
        assert impl.copy_sfa_reuse_logical_lens.cpu().tolist() == [8192, 500, 0]
        assert impl.copy_sfa_reuse_cache_tokens.cpu().tolist() == [8192, 0, 2048]
    else:
        assert impl.lim_reuse_request_count == 0
        assert impl.copy_sfa_reuse_logical_lens.cpu().tolist() == [-999] * 3
        assert impl.copy_sfa_reuse_cache_tokens.cpu().tolist() == [-999] * 3


def test_reuse_extent_is_prepared_once_per_draft_round():
    builder = make_builder()
    storage = builder.copy_sfa_vectors["reuse_logical_lens"]
    storage.fill_(-999)
    cm = common([4, 8, 12], [500, 10371, 0], pools=(1, 0, 0), generations=(11, 12, -1))
    target = populate(builder, cm)
    assert target.copy_sfa_reuse_logical_lens is None
    assert (storage == -999).all().item()

    first = populate(builder, cm, draft_index=0)
    assert first.copy_sfa_reuse_logical_lens.cpu().tolist() == [500, 8192, 0]
    address = first.copy_sfa_reuse_logical_lens.data_ptr()
    saved = storage.clone()
    for step in (1, 2):
        later = populate(
            builder,
            common([1, 2, 3], [500 + step, 10371 + step, 0], pools=(1, 0, 0), generations=(11, 12, -1)),
            draft_index=step,
            reuse_topk=True,
        )
        assert later.copy_sfa_reuse_logical_lens is None
        torch.testing.assert_close(storage, saved)

    # A new round refreshes step 0 in place, including changed activity.
    next_cm = common([4, 8, 12], [504, 10375, 0], pools=(1, 0, 0), generations=(11, 12, -1))
    assert populate(builder, next_cm).copy_sfa_reuse_logical_lens is None
    torch.testing.assert_close(storage, saved)
    next_cm.req_topk_buffer_generations[1] = -1
    next_first = populate(builder, next_cm, draft_index=0)
    assert next_first.copy_sfa_reuse_logical_lens.data_ptr() == address
    assert next_first.copy_sfa_reuse_logical_lens.cpu().tolist() == [504, 0, 0]
    assert (storage[0] == -999).all().item()
    assert (storage[2:] == -999).all().item()


def test_short_row_dense_geometry_in_mixed_batch():
    builder = make_builder()
    # Row 0: short (aligned prefix 4992 < hot 8192), row 1: long.
    cm = common([4, 8], [5000, 10371])
    metadata = populate(builder, cm)
    assert metadata.copy_sfa_prefix_lens.cpu().tolist() == [4992, 10240]
    # Short rows run copy-SFA's dense mode (C == 0) and see the whole
    # sequence; long rows keep the full hot budget.
    assert metadata.copy_sfa_cache_tokens.cpu().tolist() == [0, 8192]
    assert metadata.copy_sfa_logical_lens.cpu().tolist() == [5000, 8323]
    assert metadata.copy_sfa_reuse_logical_lens is None
    # Short row: identity block table for every stride block, zero tail, and
    # front-to-back device slots (token p -> row slot p).
    stride_blocks = 8192 // 128 + 2
    assert metadata.copy_sfa_hbm_block_table[0].cpu().tolist() == [1 * stride_blocks + b for b in range(stride_blocks)]
    assert metadata.copy_sfa_tail_lengths[0].cpu().tolist() == [0, 0]
    assert metadata.copy_sfa_device_slots[:4].cpu().tolist() == [1 * (8192 + 256) + 4996 + p for p in range(4)]
    # Long row keeps the ring-tail geometry.
    long_ring = metadata.copy_sfa_hbm_block_table[1, 64:66].cpu().tolist()
    assert long_ring == [
        stride_blocks - 2 + (10240 // 128 + 64 - 64) % 2,
        stride_blocks - 2 + (10240 // 128 + 65 - 64) % 2,
    ]
    assert metadata.copy_sfa_tail_lengths[1].cpu().tolist() == [127, 0]
    assert metadata.copy_sfa_device_slots[4:].cpu().tolist() == [8192 + (10367 + p) % 256 for p in range(4)]


def test_short_lifecycle_minus3_minus2_minus1():
    builder = make_builder()
    # Short rows always run -3 (dense, C == 0): the row content is provided
    # by the PD dense D2D / the eager full-row fill, not by a -2 rebuild.
    cm = common([4], [5000], pools=(1,), generations=(11,))
    for _ in range(3):
        assert populate(builder, cm).lim_request_state.cpu().tolist() == [-3]
    # Growth past the hot budget: the cache flip (0 -> hot) forces the -2
    # offload init, then -1 steady state.
    cm.seq_lens[0] = 8196
    assert populate(builder, cm).lim_request_state.cpu().tolist() == [-2]
    assert populate(builder, cm).lim_request_state.cpu().tolist() == [-1]
    # The physical layer restores the computed prefix on NPU before its
    # next dense attention; the target runner does not own this transition.
    cm.seq_lens[0] = 5000
    restored = populate(builder, cm)
    assert restored.lim_request_state.cpu().tolist() == [COPY_SFA_REQUEST_STATE_DENSE_RESTORE]
    assert restored.lim_cache_tokens.cpu().tolist() == [4996]
    assert restored.copy_sfa_cache_tokens.cpu().tolist() == [0]
    assert populate(builder, cm).lim_request_state.cpu().tolist() == [-3]


def test_tiny_row_stays_minus3_without_legal_init():
    builder = make_builder()
    # L < 2048 has no legal -2 (operator contract), so the row must stay -3;
    # in PD the read thread has already populated the dense content.
    cm = common([4], [500], pools=(1,), generations=(11,))
    for _ in range(3):
        assert populate(builder, cm).lim_request_state.cpu().tolist() == [-3]


def test_eager_sp_padding_uses_private_tail_and_exact_query_count():
    builder = make_builder()
    cm = common([4, 5], [10371, 8321])
    cm.num_input_tokens = 8  # padded for TP8, only five actual query rows
    metadata = populate(builder, cm)
    assert metadata.num_decode_tokens == 5
    assert metadata.slot_mapping.cpu().tolist() == [128, 129, 130, 131, 132, -1, -1, -1]
    private_start = builder.copy_sfa_pool_capacity * (8192 + 256)
    assert metadata.copy_sfa_device_slots[5:].min().item() >= private_start
    assert metadata.copy_sfa_device_slots[:5].max().item() < private_start


def test_main_slots_mask_inactive_requests_and_padding():
    builder = make_builder()
    # This branch scatters the split indexer cache with the main SFA metadata;
    # both consume this ownership mask, preserving their physical block ids.
    cm = common([2, 4, 5], [10371, 0, 500], pools=(1, 0, 0), generations=(11, -1, 12))
    cm.num_input_tokens = 8
    metadata = populate(builder, cm)
    assert metadata.slot_mapping.cpu().tolist() == [128, 129, -1, -1, 132, -1, -1, -1]
    assert metadata.copy_sfa_logical_lens.cpu().tolist() == [8195, 0, 500]
    assert metadata.copy_sfa_reuse_logical_lens is None
    assert metadata.copy_sfa_tail_lengths[1].count_nonzero().item() == 0


def test_pool_ownership_survives_compaction_and_dummy_run():
    slots = np.zeros(4, dtype=np.int32)
    generations = np.zeros(4, dtype=np.int64)
    request_slots: dict[str, int] = {}
    generation = 0
    slot_generations: dict[int, int] = {}
    last_prefixes: dict[int, int] = {}

    def prepare(req_ids, *, dummy=False):
        nonlocal request_slots, generation
        request_slots, generation, restore_tails, dense_fills = prepare_copy_sfa_request_slots(
            req_ids=req_ids,
            live_req_ids=req_ids,
            slots=slots,
            generations=generations,
            request_slots=request_slots,
            slot_generations=slot_generations,
            last_prefixes=last_prefixes,
            generation=generation,
            prebound_slots={},
            computed_tokens=None,
            padded_reqs=3,
            block_size=128,
            hot_tokens=8192,
            dummy=dummy,
        )
        assert restore_tails is False
        assert dense_fills == {}

    prepare(["a", "b"])
    assert slots[:3].tolist() == [0, 1, 6]
    assert generations[:3].tolist() == [1, 2, -1]
    # Removing a compacts b; new c may reuse a's slot, with a new generation.
    prepare(["b", "c"])
    assert slots[:3].tolist() == [1, 0, 6]
    assert generations[:3].tolist() == [2, 3, -1]
    prepare(["b", "c"], dummy=True)
    assert slots[:3].tolist() == [4, 5, 6]
    assert generations[:3].tolist() == [-1, -1, -1]
    assert request_slots == {"b": 1, "c": 0}
    prepare(["b", "c"])
    assert generations[:3].tolist() == [2, 3, -1]
    # The draft-only dummy path needs no ownership maps or model runner.
    prepare_copy_sfa_dummy_slots(slots, generations, 3)
    assert slots[:3].tolist() == [4, 5, 6]
    assert generations[:3].tolist() == [-1, -1, -1]
    assert request_slots == {"b": 1, "c": 0}


def test_normal_decode_skips_restore_descriptors_and_reuses_block_table():
    builder = make_builder()
    cm = common([4], [10371], pools=(1,), generations=(11,))
    cm.copy_sfa_restore_tails = False
    md = populate(builder, cm)
    assert md.copy_sfa_copy_src_offsets is None
    assert md.copy_sfa_source_block_table.data_ptr() == cm.block_table_tensor.data_ptr()
    assert md.copy_sfa_logical_lens.cpu().tolist() == [8323]
    cm.copy_sfa_restore_tails = True
    restored = populate(builder, cm)
    assert restored.copy_sfa_copy_src_offsets is not None
    assert restored.copy_sfa_tail_lengths.cpu().tolist() == [[127, 0]]
    cm.copy_sfa_restore_tails = False
    assert populate(builder, cm).copy_sfa_copy_src_offsets is None


def test_host_buffers_are_persistent_and_device_lengths_are_authoritative():
    builder = make_builder()
    cm = common([4], [10371], pools=(1,), generations=(11,))
    md = populate(builder, cm)
    addresses = {
        name: (buffers[0].cpu.data_ptr(), buffers[0].gpu.data_ptr()) for name, buffers in builder.copy_sfa_host.items()
    }
    cm.seq_lens_cpu.fill_(1)
    cm._seq_lens_cpu.fill_(1048576)
    cm.seq_lens.fill_(10368)
    changed = populate(builder, cm)
    assert changed.copy_sfa_seq_lens.cpu().tolist() == [10368]
    assert changed.copy_sfa_prefix_lens.cpu().tolist() == [10240]
    assert changed.copy_sfa_logical_lens.cpu().tolist() == [8320]
    assert changed.copy_sfa_device_slots.cpu().tolist() == [8448 + 8192 + p % 256 for p in range(10364, 10368)]
    assert changed.copy_sfa_query_ends.data_ptr() == md.copy_sfa_query_ends.data_ptr()
    assert addresses == {
        name: (buffers[0].cpu.data_ptr(), buffers[0].gpu.data_ptr()) for name, buffers in builder.copy_sfa_host.items()
    }


def test_mtp_sparse_to_dense_recovery_uses_physical_bank_without_cpu_prefix_rollback():
    builder = make_builder(num_mtp_layers=1)
    old = common([1], [8193], pools=(0,), generations=(7,))
    old.copy_sfa_restore_tails = False
    old.max_query_len = 1
    assert populate(builder, old, draft_index=2, reuse_topk=False).lim_request_state.tolist() == [-2]
    current = common([4], [8195], pools=(0,), generations=(7,))
    current.copy_sfa_restore_tails = False
    with patch.object(torch.Tensor, "item", side_effect=AssertionError("unexpected device scalar read")):
        restored = populate(builder, current, draft_index=0, reuse_topk=False)
    assert restored.lim_request_state.tolist() == [COPY_SFA_REQUEST_STATE_DENSE_RESTORE]
    assert restored.lim_cache_tokens.tolist() == [8191]
    assert restored.copy_sfa_cache_tokens.tolist() == [0]
    assert restored.copy_sfa_copy_src_offsets is None
    assert restored.copy_sfa_device_slots.tolist() == [8191, 8192, 8193, 8194]
    assert builder.lim_last_cache[0, 0] == 0  # target bank unchanged
    assert populate(builder, current, draft_index=0, reuse_topk=False).lim_request_state.tolist() == [-3]


def test_dense_recovery_is_private_to_the_physical_mtp_layer_and_generation():
    builder = make_builder(num_mtp_layers=2)
    old = common([1], [8193], pools=(0,), generations=(7,))
    old.max_query_len = 1
    populate(builder, old, draft_index=2, reuse_topk=False)  # bank 1
    current = common([4], [8195], pools=(0,), generations=(7,))
    assert populate(builder, current, draft_index=1, reuse_topk=False).lim_request_state.tolist() == [-3]  # bank 2
    current.req_topk_buffer_generations[0] = 8
    assert populate(builder, current, draft_index=0, reuse_topk=False).lim_request_state.tolist() == [-3]


def test_mixed_fallback_then_pure_decode_restores_identity_or_cold_fills():
    builder = make_builder()
    cm = common([1, 2], [5000, 10371])
    populate(builder, cm)
    with patch(MODULE + ".split_decodes_and_prefills", return_value=(1, 1, 1, 1)):
        fallback = builder._populate_offload_metadata(SimpleNamespace(), cm)
    assert not fallback.fused_copy_sfa_enabled
    cm.copy_sfa_restore_tails = False
    restored = populate(builder, cm)
    assert restored.lim_request_state.tolist() == [COPY_SFA_REQUEST_STATE_DENSE_RESTORE, -2]
    assert restored.lim_cache_tokens.tolist() == [4999, 8192]
    assert restored.copy_sfa_cache_tokens.tolist() == [0, 8192]
    assert populate(builder, cm).lim_request_state.tolist() == [-3, -1]
