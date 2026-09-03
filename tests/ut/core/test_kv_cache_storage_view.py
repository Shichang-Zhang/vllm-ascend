from types import SimpleNamespace

import torch
from vllm.utils.math_utils import cdiv

from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    get_storage_cp_world_size,
    kv_cache_spec_uses_unified_host_view,
)
from vllm_ascend.utils import (
    enable_sfa_dcp_replicated_indexer,
    kv_offload_decode_enabled,
)


def test_nested_store_on_host_is_unified_view() -> None:
    spec = SimpleNamespace(
        store_on_host=False,
        kv_cache_specs={"main": SimpleNamespace(store_on_host=True, block_size=128)},
    )
    assert kv_cache_spec_uses_unified_host_view(spec)
    assert get_storage_cp_world_size(spec, 8) == 1
    assert cdiv(131072, 128 * get_storage_cp_world_size(spec, 8)) == 1024


def test_device_local_group_keeps_runtime_cp() -> None:
    spec = SimpleNamespace(store_on_host=False, kv_cache_specs={"main": SimpleNamespace(store_on_host=False)})
    assert not kv_cache_spec_uses_unified_host_view(spec)
    assert get_storage_cp_world_size(spec, 8) == 8
    assert cdiv(131072, 128 * 8) == 128


def test_top_level_store_on_host() -> None:
    spec = SimpleNamespace(store_on_host=True)
    assert kv_cache_spec_uses_unified_host_view(spec)
    assert get_storage_cp_world_size(spec, 16) == 1


def test_indexer_sibling_of_host_uses_storage_cp_1() -> None:
    from vllm_ascend.core.kv_cache_interface import kv_cache_groups_share_unified_scheduler_pages

    host = SimpleNamespace(
        kv_cache_spec=SimpleNamespace(
            store_on_host=False,
            kv_cache_specs={"main": SimpleNamespace(store_on_host=True, block_size=128)},
        )
    )
    indexer = SimpleNamespace(kv_cache_spec=SimpleNamespace(store_on_host=False, block_size=128))
    groups = [host, indexer]
    assert kv_cache_groups_share_unified_scheduler_pages(groups)
    assert not kv_cache_spec_uses_unified_host_view(indexer.kv_cache_spec)
    assert get_storage_cp_world_size(indexer.kv_cache_spec, 8) == 8
    assert get_storage_cp_world_size(
        indexer.kv_cache_spec, 8, scheduler_uses_unified_pages=True
    ) == 1
    assert cdiv(131072, 128 * 1) == 1024


def _host_mla_spec(**kwargs):
    return AscendMLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.bfloat16,
        cache_dtype_str="auto",
        store_on_host=True,
        **kwargs,
    )


def test_host_spec_max_memory_does_not_divide_by_dcp():
    spec = _host_mla_spec()
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=131072),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=8,
            prefill_context_parallel_size=1,
        ),
    )
    pages = spec.max_memory_usage_bytes(vllm_config) // spec.page_size_bytes
    assert pages == 1024


def test_device_spec_max_memory_still_divides_by_dcp():
    spec = AscendMLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.bfloat16,
        cache_dtype_str="auto",
        store_on_host=False,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=131072),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=8,
            prefill_context_parallel_size=1,
        ),
    )
    pages = spec.max_memory_usage_bytes(vllm_config) // spec.page_size_bytes
    assert pages == 128


def test_kv_offload_decode_enabled_reads_additional_config():
    on = SimpleNamespace(additional_config={"kv_offload_decode_config": {"enabled": True}})
    off = SimpleNamespace(additional_config={"kv_offload_decode_config": {"enabled": False}})
    assert kv_offload_decode_enabled(on) is True
    assert kv_offload_decode_enabled(off) is False


def test_replicated_indexer_disabled_when_decode_offload_enabled(monkeypatch):
    monkeypatch.setattr("vllm_ascend.utils.model_uses_sfa_sparse", lambda cfg: True)
    common = dict(
        model_config=object(),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=8,
            prefill_context_parallel_size=1,
        ),
    )
    offload = SimpleNamespace(
        additional_config={"kv_offload_decode_config": {"enabled": True}},
        **common,
    )
    no_offload = SimpleNamespace(
        additional_config={"kv_offload_decode_config": {"enabled": False}},
        **common,
    )
    assert enable_sfa_dcp_replicated_indexer(offload) is False
    assert enable_sfa_dcp_replicated_indexer(no_offload) is True
