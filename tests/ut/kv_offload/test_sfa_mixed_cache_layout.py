# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.v1.core.kv_cache_utils import get_kv_cache_config_from_groups, get_kv_cache_groups
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, KVCacheTensor, UniformTypeKVCacheSpecs

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload import sparse_kv_offload_manager as manager_module
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sfa_layer_residency import (
    SfaLayerResidency,
    SfaResidencyPlan,
    build_sfa_cache_layout,
)
from vllm_ascend.worker.model_runner_v1 import MLAAttention, NPUModelRunner

OWNER = "model.layers.5.attn"
MEMBER = "model.layers.9.attn"
DRAFT = "draft.layers.0.attn"
INDEXER = "model.layers.5.indexer.k_cache"


def fixture_layout(*, uniform=False, c8=True, draft=True):
    layers = [SfaLayerResidency(OWNER, OWNER, False, False), SfaLayerResidency(MEMBER, OWNER, False, True)]
    if draft:
        layers.append(SfaLayerResidency(DRAFT, DRAFT, True, True))
    plan = SfaResidencyPlan(tuple(layers), ((OWNER, (MEMBER,)),))
    specs = {
        layer.name: AscendMLAAttentionSpec(
            block_size=128, num_kv_heads=1, head_size=576, dtype=torch.bfloat16, store_on_host=layer.offloaded
        )
        for layer in layers
    }
    specs[INDEXER] = AscendSFAIndexerCacheSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.int8 if c8 else torch.bfloat16,
        scale_dim=1 if c8 else 0,
        scale_dtype=torch.float16 if c8 else torch.int8,
        cache_sparse_li_c8=c8,
    )
    names_by_group = [list(specs)] if uniform else [[INDEXER], [OWNER], [MEMBER, *([DRAFT] if draft else [])]]
    groups = []
    for names in names_by_group:
        spec = UniformTypeKVCacheSpecs.from_specs({name: specs[name] for name in names})
        assert spec is not None
        groups.append(KVCacheGroupSpec(layer_names=names, kv_cache_spec=spec))
    config = KVCacheConfig(
        num_blocks=3,
        kv_cache_groups=groups,
        kv_cache_tensors=[
            KVCacheTensor(size=spec.page_size_bytes * 3, shared_by=[name]) for name, spec in specs.items()
        ],
    )
    return plan, specs, config


@pytest.mark.parametrize("uniform", [False, True])
def test_layout_flattens_specs_and_preserves_actual_group_ids(uniform):
    plan, _, config = fixture_layout(uniform=uniform)
    layout = build_sfa_cache_layout(config, plan)
    assert layout.host_group_id == (0 if uniform else 2)
    assert {layer.name for layer in layout.layers if layer.store_on_host} == {MEMBER, DRAFT}
    assert {layer.name for layer in layout.layers if layer.is_indexer} == {INDEXER}
    if not uniform:
        config.kv_cache_groups.reverse()
        assert build_sfa_cache_layout(config, plan).host_group_id == 0


@pytest.mark.parametrize("fault", ["host_groups", "missing", "placement", "block_size", "alias", "tensor", "duplicate"])
def test_invalid_mixed_layout_fails_before_allocation(fault):
    plan, specs, config = fixture_layout()
    if fault == "host_groups":
        config.kv_cache_groups[-1] = KVCacheGroupSpec(layer_names=[MEMBER], kv_cache_spec=specs[MEMBER])
        config.kv_cache_groups.append(KVCacheGroupSpec(layer_names=[DRAFT], kv_cache_spec=specs[DRAFT]))
    elif fault == "missing":
        config.kv_cache_groups[-1].layer_names.remove(DRAFT)
    elif fault == "placement":
        config.kv_cache_groups[1].kv_cache_spec = replace(specs[OWNER], store_on_host=True)
    elif fault == "block_size":
        config.kv_cache_groups[0].kv_cache_spec = replace(specs[INDEXER], block_size=64)
    elif fault == "alias":
        config.kv_cache_tensors[0].shared_by.append(MEMBER)
    elif fault == "tensor":
        config.kv_cache_tensors.pop()
    else:
        config.kv_cache_groups.append(config.kv_cache_groups[0])
    with pytest.raises(ValueError):
        build_sfa_cache_layout(config, plan)


def make_runner(plan, config, tp_rank):
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.device = torch.device("cpu")
    runner.use_sparse = True
    runner.use_compress = False
    runner.use_hybrid_blocks = len(config.kv_cache_groups) > 1
    runner.sparse_kv_offload_enabled = True
    runner.sparse_kv_offload_config = SimpleNamespace(offload_shared_indexer_only=True, keep_device_kv_cache=False)
    runner._sfa_layer_residency_plan = plan
    runner.tp_rank = tp_rank
    runner.runner_only_attn_layers = set()
    runner.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context={}),
        kv_transfer_config=None,
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(kv_lora_rank=512, qk_rope_head_dim=64)),
    )
    runner.model_config = SimpleNamespace(use_mla=True)
    runner._get_attention_kv_cache_dims = lambda name, spec: (512, 64)
    runner.sparse_kv_offload_manager = SimpleNamespace(
        allocate_host_kv_tensors=MagicMock(
            side_effect=lambda sizes, alignment: [torch.empty(s, dtype=torch.int8) for s in sizes]
        )
    )
    backend = SimpleNamespace(
        get_kv_cache_shape=lambda blocks, size, heads, dim: (blocks, size, heads, dim),
        get_supported_kernel_block_sizes=lambda: [128],
    )
    runner._kv_cache_spec_attn_group_iterator = lambda: [
        SimpleNamespace(layer_names=group.layer_names, kv_cache_spec=group.kv_cache_spec, backend=backend)
        for group in config.kv_cache_groups
    ]
    return runner


@pytest.mark.parametrize("tp_rank", [0, 1])
@pytest.mark.parametrize("uniform,c8", [(False, False), (False, True), (True, True)])
def test_real_runner_allocates_resident_kv_separately_from_host_and_quantized_indexer(tp_rank, uniform, c8):
    plan, _, config = fixture_layout(uniform=uniform, c8=c8)
    runner = make_runner(plan, config, tp_rank)
    with patch.object(
        manager_module,
        "allocate_kv_offload_topk_buffer_pair",
        side_effect=lambda *args: (torch.empty(2, 8448, 1, 512), torch.empty(2, 8448, 1, 64)),
    ) as allocate_hot:
        raw = runner._allocate_kv_cache_tensors(config)
        caches = runner._reshape_kv_cache_tensors(config, raw)
    assert len(caches[OWNER]) == 2
    assert caches[OWNER][0].shape == (3, 128, 1, 512)
    assert caches[OWNER][1].shape == (3, 128, 1, 64)
    assert caches[OWNER][0].dtype == torch.bfloat16
    assert len(caches[MEMBER]) == len(caches[DRAFT]) == 6
    assert caches[MEMBER][0] is caches[MEMBER][1] is None
    assert caches[MEMBER][2].shape == caches[OWNER][0].shape
    assert caches[MEMBER][2].data_ptr() != caches[OWNER][0].data_ptr()
    assert caches[DRAFT][2].data_ptr() != caches[MEMBER][2].data_ptr()
    assert caches[INDEXER][0].dtype == (torch.int8 if c8 else torch.bfloat16)
    assert len(caches[INDEXER]) == (2 if c8 else 1)
    if c8:
        assert caches[INDEXER][1].dtype == torch.float16
        assert caches[INDEXER][1].shape == (3, 128, 1, 1)
    assert allocate_hot.call_count == 2
    assert runner.sparse_kv_offload_manager.allocate_host_kv_tensors.call_count == 2


@pytest.mark.parametrize("draft", [False, True])
def test_registration_uses_classification_and_actual_mtp_membership(draft):
    plan, _, config = fixture_layout(draft=draft)
    manager = manager_module.SparseKVOffloadManager.__new__(manager_module.SparseKVOffloadManager)
    manager.residency_plan = plan
    manager.sparse_kv_offload_config = SimpleNamespace(offload_shared_indexer_only=True)
    assert manager._infer_group_block_sizes(config) == 128
    manager.num_target_layers = 64  # Neither total layer count nor last cache identifies the MTP.
    manager.tp_rank = 1
    manager._register_offload_layers(dict.fromkeys(reversed([layer.name for layer in manager.sfa_cache_layout.layers])))
    assert manager.offload_layer_names == [MEMBER, *([DRAFT] if draft else [])]
    assert manager.resident_layer_names == (OWNER,)
    assert manager.mtp_layer_ids == ((1,) if draft else ())
    assert manager.mtp_layer_id == (1 if draft else -1)
    with pytest.raises(KeyError):
        manager._get_offload_layer_id(OWNER)


def test_resident_owner_cost_is_in_device_budget_and_profile_only_reserves_offloaded_rows():
    plan, specs, config = fixture_layout()
    runner = make_runner(plan, config, 0)
    runner.vllm_config.model_config.max_model_len = 16384
    runner.vllm_config.parallel_config = SimpleNamespace(decode_context_parallel_size=1)
    runner.vllm_config.scheduler_config = SimpleNamespace(max_num_seqs=8)
    budget = manager_module.plan_sparse_kv_offload_memory(specs, runner.vllm_config, 1 << 20, 1 << 30, False)
    device_page = specs[OWNER].page_size_bytes + specs[INDEXER].page_size_bytes
    assert budget.npu_limit_blocks == (1 << 20) // device_page
    assert budget.planned_device_bytes == budget.final_num_blocks * device_page
    assert budget.planned_host_bytes == budget.final_num_blocks * (
        specs[MEMBER].page_size_bytes + specs[DRAFT].page_size_bytes
    )
    with patch.object(
        manager_module, "allocate_kv_offload_topk_buffer_pair", return_value=(torch.empty(1), torch.empty(1))
    ) as allocate:
        buffers = manager_module.allocate_kv_offload_topk_profile_buffers(
            specs, runner.vllm_config, runner.sparse_kv_offload_config
        )
    assert len(buffers) == allocate.call_count == 2


def test_runner_spec_placement_requires_the_bound_plan():
    plan, _, config = fixture_layout()
    runner = make_runner(plan, config, 0)
    assert not runner._sfa_store_on_host(OWNER)
    assert runner._sfa_store_on_host(MEMBER) and runner._sfa_store_on_host(DRAFT)
    with pytest.raises(ValueError, match="classified"):
        runner._sfa_store_on_host("unbound")
    runner.sparse_kv_offload_config.offload_shared_indexer_only = False
    assert runner._sfa_store_on_host(OWNER)
    runner.sparse_kv_offload_enabled = False
    assert not runner._sfa_store_on_host(OWNER)


def test_real_runner_spec_generation_uses_residency_before_memory_planning():
    plan, _, config = fixture_layout()
    runner = make_runner(plan, config, 0)
    runner.block_size = 128
    runner.kv_cache_dtype = torch.bfloat16
    runner.vllm_config.cache_config = SimpleNamespace(cache_dtype="auto")
    runner.model_config.hf_text_config = runner.vllm_config.model_config.hf_text_config
    modules = {}
    for layer in plan.layers:
        module = MLAAttention.__new__(MLAAttention)
        torch.nn.Module.__init__(module)
        module.impl = SimpleNamespace(enable_sparse_sfa_c8=False)
        modules[layer.name] = module
    with (
        patch("vllm_ascend.worker.model_runner_v1.has_ec_transfer", return_value=False),
        patch("vllm_ascend.worker.model_runner_v1.get_layers_from_vllm_config", return_value=modules),
    ):
        specs = runner.get_kv_cache_spec()
    assert not specs[OWNER].store_on_host
    assert specs[MEMBER].store_on_host and specs[DRAFT].store_on_host
    assert {spec.page_size_bytes for spec in specs.values()} == {128 * 576 * 2}


def test_real_cache_planner_keeps_layer_sizes_and_private_storage_for_mixed_specs():
    plan, specs, _ = fixture_layout()
    cfg = SimpleNamespace(
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        cache_config=SimpleNamespace(num_gpu_blocks_override=None),
    )
    groups = get_kv_cache_groups(cfg, specs)
    assert len(groups) == 1
    cache_config = get_kv_cache_config_from_groups(
        cfg, groups, 7 * sum(spec.page_size_bytes for spec in specs.values())
    )
    layout = build_sfa_cache_layout(cache_config, plan)
    assert cache_config.num_blocks == 7 and layout.host_group_id == 0
    for tensor in cache_config.kv_cache_tensors:
        assert len(tensor.shared_by) == 1
        assert tensor.size == specs[tensor.shared_by[0]].page_size_bytes * 7
