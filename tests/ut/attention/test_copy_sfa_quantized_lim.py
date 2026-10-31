# SPDX-License-Identifier: Apache-2.0
"""Quantized LIM dispatch through the current split-cache serving interface."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from tests.ut.attention.test_copy_sfa_metadata import common, make_builder, populate
from vllm_ascend.attention.sfa_kv_offload import AscendSFAKVOffloadImpl
from vllm_ascend.attention.sfa_v1 import AscendSFAImpl

MODULE = "vllm_ascend.attention.sfa_kv_offload"
HEADS = 32
HEAD_DIM = 128
TOKENS = 3
BLOCKS = 128


def inputs(*, draft_index=None, scale_ndim=4):
    builder = make_builder()
    populate(builder, common([1, 3], [9000, 9000]), draft_index=draft_index)
    cm = common([1, 3], [9001, 500])
    metadata = populate(builder, cm, draft_index=draft_index)
    metadata.block_table = cm.block_table_tensor
    impl = AscendSFAKVOffloadImpl.__new__(AscendSFAKVOffloadImpl)
    impl.enable_sparse_li_c8 = True
    impl.enable_sparse_sfa_c8 = False
    impl.n_head = HEADS
    impl._copy_sfa_metadata = metadata
    impl.lim_slot_map = torch.empty((8, BLOCKS * HEAD_DIM), dtype=torch.int32)
    for name, shape in (
        ("lim_topk_src", (TOKENS, 1, 2048)),
        ("lim_topk_dst", (TOKENS, 1, 2048)),
        ("lim_topk_misses", (TOKENS,)),
        ("lim_miss_src", (2, 32768)),
        ("lim_miss_dst", (2, 32768)),
        ("lim_misses", (2,)),
        ("copy_sfa_reuse_logical_lens", (2,)),
        ("copy_sfa_reuse_cache_tokens", (2,)),
    ):
        setattr(impl, name, torch.zeros(shape, dtype=torch.int32))
    query = torch.ones((TOKENS * HEADS, HEAD_DIM), dtype=torch.int8)
    query_scale = torch.arange(1, TOKENS * HEADS + 1, dtype=torch.float16) / HEADS
    key = torch.ones((BLOCKS, HEAD_DIM, 1, HEAD_DIM), dtype=torch.int8)
    scale_shape = (BLOCKS, HEAD_DIM, 1, 1) if scale_ndim == 4 else (BLOCKS, HEAD_DIM, 1)
    key_scale = torch.arange(1, BLOCKS * HEAD_DIM + 1, dtype=torch.float16).reshape(scale_shape) / BLOCKS
    weights = torch.ones((TOKENS, HEADS), dtype=torch.bfloat16)
    return impl, metadata, query, query_scale, key, key_scale, weights


@pytest.mark.parametrize("draft_index", [None, 0], ids=["target", "draft0"])
@pytest.mark.parametrize("scale_ndim", [3, 4])
def test_split_cache_dispatch_passes_real_scales_and_preserves_mtp_reuse(draft_index, scale_ndim):
    impl, md, query, query_scale, key, key_scale, weights = inputs(draft_index=draft_index, scale_ndim=scale_ndim)
    with (
        patch(MODULE + ".torch.ops._C_ascend.npu_fused_quant_lightning_indexer_manage", create=True) as quant,
        patch(MODULE + ".torch.ops._C_ascend.npu_fused_lightning_indexer_manage", create=True) as bf16,
        patch.object(torch.Tensor, "item", side_effect=AssertionError("unexpected device scalar read")),
    ):
        result = impl._select_indexer_topk(
            query, query_scale, (TOKENS, HEADS, HEAD_DIM), weights, (None, None, key, key_scale), md, None, None
        )
    bf16.assert_not_called()
    quant.assert_called_once()
    args = quant.call_args.args
    torch.testing.assert_close(args[1], query_scale.reshape(TOKENS, HEADS))
    torch.testing.assert_close(args[2], query.reshape(TOKENS, HEADS, HEAD_DIM))
    torch.testing.assert_close(args[3], key_scale.reshape(BLOCKS, HEAD_DIM, 1))
    assert args[1].data_ptr() == query_scale.data_ptr()
    assert args[3].data_ptr() == key_scale.data_ptr()
    assert args[9].data_ptr() == md.copy_sfa_cache_tokens.data_ptr()
    assert args[10].tolist() == [-1, -3]
    assert args[9].tolist() == [8192, 0]
    assert md.copy_sfa_cache_tokens.tolist() == [8192, 0]
    assert result.data_ptr() == impl.lim_topk_src.data_ptr()
    if draft_index == 0:
        assert impl.copy_sfa_reuse_cache_tokens.tolist() == [8192, 0]
        # Quantized LIM must also preserve the 41-token circular tail.
        assert impl.copy_sfa_reuse_logical_lens.tolist() == [8192 + 41, 500]


@pytest.mark.parametrize(
    ("field", "dtype", "error"),
    [
        ("query_scale", None, "requires query and index-key dequant scales"),
        ("key_scale", None, "requires query and index-key dequant scales"),
        ("query_scale", torch.float32, "requires FP16"),
        ("key_scale", torch.float32, "requires FP16"),
        ("query", torch.bfloat16, "requires INT8"),
        ("index_cache", torch.bfloat16, "requires INT8"),
        ("weights", torch.float16, "requires BF16"),
    ],
)
def test_invalid_quantized_inputs_fail_before_native_dispatch(field, dtype, error):
    impl, md, query, query_scale, key, key_scale, weights = inputs()
    values = dict(query=query, weights=weights, index_cache=key, query_scale=query_scale, key_scale=key_scale)
    values[field] = None if dtype is None else values[field].to(dtype)
    with (
        patch(MODULE + ".torch.ops._C_ascend.npu_fused_quant_lightning_indexer_manage", create=True) as quant,
        pytest.raises(ValueError, match=error),
    ):
        impl._lim_select(block_table=md.block_table, **values)
    quant.assert_not_called()


@pytest.mark.parametrize("dtype", [torch.int8, torch.float8_e4m3fn], ids=["int8", "fp8"])
def test_c8_initialization_allows_int8_and_rejects_fp8(dtype):
    class ReachedDeviceSetup(Exception):
        pass

    def init_base(self, *args, **kwargs):
        self.enable_dsa_cp = False
        self.enable_sparse_sfa_c8 = False
        self.enable_sparse_li_c8 = True
        self.c8_k_cache_dtype = dtype
        self.vllm_config = SimpleNamespace(
            cache_config=SimpleNamespace(block_size=128),
            scheduler_config=SimpleNamespace(max_num_seqs=2, max_num_batched_tokens=16),
        )

    config = SimpleNamespace(
        sparse_kv_offload_config=SimpleNamespace(
            use_fused_overlap=False, use_fused_copy_sfa=True, topk_buffer_size=8192
        )
    )
    with (
        patch.object(AscendSFAImpl, "__init__", init_base),
        patch(MODULE + ".get_ascend_config", return_value=config),
        patch(MODULE + ".torch.device", side_effect=ReachedDeviceSetup),
        pytest.raises(ReachedDeviceSetup if dtype == torch.int8 else NotImplementedError),
    ):
        AscendSFAKVOffloadImpl(1, 128, 1.0, 1, None, None, "auto", None, "decoder", None)
