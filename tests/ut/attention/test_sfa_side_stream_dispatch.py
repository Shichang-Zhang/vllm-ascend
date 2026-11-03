# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from tests.ut.attention.test_copy_sfa_fallback import make_manager
from tests.ut.attention.test_copy_sfa_metadata import common, make_builder, populate
from tests.ut.attention.test_copy_sfa_quantized_lim import inputs
from vllm_ascend.attention.sfa_kv_offload import AscendSFAKVOffloadImpl
from vllm_ascend.attention.sfa_v1 import AscendSFAImpl

MODULE = "vllm_ascend.attention.sfa_kv_offload"


def test_quantized_lim_launches_member_copy_before_resident_owner_full_context_indexer():
    layer, md, query, scales, key, key_scales, weights = inputs()
    layer.use_side_stream_onboard, layer.lim_full_resident = True, True
    layer.lim_share_members = (object(),)
    layer._in_graph_runtime = lambda: True
    sequence = []
    onboarder = SimpleNamespace(launch=MagicMock(side_effect=lambda *a, **kw: sequence.append("copy")))
    token = object()
    with (
        patch(MODULE + ".get_forward_context", return_value=token),
        patch(MODULE + ".get_sparse_kv_offload_manager", return_value=SimpleNamespace(lim_group_onboarder=onboarder)),
        patch(
            MODULE + ".torch.ops._C_ascend.npu_fused_quant_lightning_indexer_manage",
            create=True,
            side_effect=lambda *a: sequence.append("lim"),
        ),
        patch.object(AscendSFAImpl, "_select_indexer_topk", side_effect=lambda *a: sequence.append("full-li")),
    ):
        layer._select_indexer_topk(query, scales, (3, 32, 128), weights, (None, None, key, key_scales), md, None, None)
    assert sequence == ["lim", "copy", "full-li"]
    onboarder.launch.assert_called_once_with(layer, md, token, capturing=True)


@pytest.mark.parametrize("mode", ["member", "inline", "draft_reuse"])
def test_attention_waits_before_read_and_only_completed_member_copy_disables_inline_misses(mode):
    manager = make_manager(keep_device=False)
    manager.ensure_copy_sfa_host_visibility = MagicMock()
    manager.k_caches_cpu = [torch.zeros(128, 128, 1, 3, dtype=torch.bfloat16)]
    manager.v_caches_cpu = [torch.zeros(128, 128, 1, 1, dtype=torch.bfloat16)]
    md = populate(make_builder(), common([1], [9000]))
    layer = AscendSFAKVOffloadImpl.__new__(AscendSFAKVOffloadImpl)
    layer.skip_topk, layer.has_indexer = True, mode == "draft_reuse"
    layer.kv_lora_rank, layer.qk_rope_head_dim, layer.scale = 3, 1, 1.0
    layer._in_graph_runtime = lambda: True
    owner = SimpleNamespace(
        lim_topk_dst=torch.zeros(1, 1, 2048, dtype=torch.int32),
        lim_topk_src=torch.zeros(1, 1, 2048, dtype=torch.int32),
        lim_topk_misses=torch.ones(1, dtype=torch.int32),
        lim_miss_src=torch.zeros(1, 32768, dtype=torch.int32),
        lim_miss_dst=torch.zeros(1, 32768, dtype=torch.int32),
        lim_misses=torch.ones(1, dtype=torch.int32),
    )
    layer.lim_indexer_owner = owner
    zeros = torch.zeros(1, dtype=torch.int32)
    order = []

    def wait(*args):
        order.append("wait")
        manager.topk_buffers_k[0].fill_(23)
        return zeros, zeros

    onboarder = SimpleNamespace(member_owners={"layer.0": "owner"}, wait=MagicMock(side_effect=wait))
    manager.lim_group_onboarder = None if mode == "inline" else onboarder
    if mode == "draft_reuse":
        for name, value in vars(owner).items():
            setattr(layer, name, value)
        layer.lim_indexer_owner = layer
        layer.copy_sfa_reuse_cache_tokens = md.copy_sfa_cache_tokens
        layer.copy_sfa_reuse_logical_lens = md.copy_sfa_logical_lens
        layer.lim_reuse_topk_misses = layer.lim_reuse_misses = zeros

    def attention(*args):
        order.append("attention")
        assert args[7].tolist() == args[10].tolist() == ([1] if mode == "inline" else [0])
        if mode == "member":
            assert torch.all(args[14] == 23)
        args[-1].fill_(23)

    with (
        patch(MODULE + ".get_forward_context", return_value=object()),
        patch(
            MODULE + ".torch.ops._C_ascend.npu_fused_scatter_copy_sparse_flash_attention",
            create=True,
            side_effect=attention,
        ),
    ):
        result = layer._copy_sfa_attention(torch.ones(1, 8, 3), torch.ones(1, 8, 1), None, md, manager, "layer.0")
    assert torch.all(result == 23)
    assert order == (["wait", "attention"] if mode == "member" else ["attention"])
