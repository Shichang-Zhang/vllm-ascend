from types import SimpleNamespace

from vllm.utils.math_utils import cdiv

from vllm_ascend.core.kv_cache_interface import (
    get_storage_cp_world_size,
    kv_cache_spec_uses_unified_host_view,
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
