from types import SimpleNamespace

import torch
from vllm.utils.math_utils import cdiv

from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    KVCacheAddressingLayout,
    get_kv_cache_group_layout,
    get_storage_cp_world_size,
    kv_cache_spec_uses_unified_host_view,
)
from vllm_ascend.utils import (
    enable_sfa_dcp_replicated_indexer,
    fused_sfa_host_offload_decode_enabled,
    is_pd_decode_kv_consumer,
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


def _sfa_dcp_cfg(*, offload: bool, consumer: bool, kv_role: str | None = None):
    flags = dict(
        is_kv_consumer=consumer,
        is_kv_producer=not consumer,
    )
    if kv_role is not None:
        flags["kv_role"] = kv_role
    return SimpleNamespace(
        additional_config={"kv_offload_decode_config": {"enabled": offload}},
        kv_transfer_config=SimpleNamespace(**flags),
        model_config=object(),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=8,
            prefill_context_parallel_size=1,
        ),
    )


def test_replicated_indexer_off_only_for_decode_consumer_offload_sfa(monkeypatch):
    monkeypatch.setattr("vllm_ascend.utils.model_uses_sfa_sparse", lambda cfg: True)
    decode = _sfa_dcp_cfg(offload=True, consumer=True, kv_role="kv_consumer")
    prefill = _sfa_dcp_cfg(offload=True, consumer=False, kv_role="kv_producer")
    decode_no_offload = _sfa_dcp_cfg(offload=False, consumer=True, kv_role="kv_consumer")

    assert is_pd_decode_kv_consumer(decode) is True
    assert fused_sfa_host_offload_decode_enabled(decode) is True
    assert enable_sfa_dcp_replicated_indexer(decode) is False

    assert is_pd_decode_kv_consumer(prefill) is False
    assert fused_sfa_host_offload_decode_enabled(prefill) is False
    assert enable_sfa_dcp_replicated_indexer(prefill) is True

    assert fused_sfa_host_offload_decode_enabled(decode_no_offload) is False
    assert enable_sfa_dcp_replicated_indexer(decode_no_offload) is True


def test_offload_config_alone_does_not_disable_replicated_indexer(monkeypatch):
    monkeypatch.setattr("vllm_ascend.utils.model_uses_sfa_sparse", lambda cfg: True)
    cfg = SimpleNamespace(
        additional_config={"kv_offload_decode_config": {"enabled": True}},
        kv_transfer_config=None,
        model_config=object(),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=8,
            prefill_context_parallel_size=1,
        ),
    )
    assert kv_offload_decode_enabled(cfg) is True
    assert fused_sfa_host_offload_decode_enabled(cfg) is False
    assert enable_sfa_dcp_replicated_indexer(cfg) is True


def test_group_layout_host_main_and_indexer_sibling() -> None:
    host = SimpleNamespace(
        kv_cache_spec=SimpleNamespace(
            store_on_host=False,
            kv_cache_specs={"main": SimpleNamespace(store_on_host=True, block_size=128)},
        )
    )
    indexer = SimpleNamespace(kv_cache_spec=SimpleNamespace(store_on_host=False, block_size=128))
    assert get_kv_cache_group_layout(host) is KVCacheAddressingLayout.UNIFIED_HOST_MAIN
    assert (
        get_kv_cache_group_layout(indexer, scheduler_uses_unified_pages=True)
        is KVCacheAddressingLayout.GLOBAL_LOGICAL_INDEXER
    )
    assert get_kv_cache_group_layout(indexer) is KVCacheAddressingLayout.DEVICE_LOCAL_CP
    assert get_storage_cp_world_size(indexer.kv_cache_spec, 8, scheduler_uses_unified_pages=True) == 1
    assert get_storage_cp_world_size(indexer.kv_cache_spec, 8) == 8
