# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cache-manager group validation for Mooncake mixed SFA residency."""

from vllm.v1.kv_cache_interface import KVCacheConfig, UniformTypeKVCacheSpecs

from .mooncake_dsa_metadata import DSA_GROUPED_TRANSFER_VERSION, DsaCacheGroupBlocks


def find_dsa_host_group(config: KVCacheConfig) -> int:
    host_groups = set()
    for gid, group in enumerate(config.kv_cache_groups):
        for name in group.layer_names:
            spec = group.kv_cache_spec
            if isinstance(spec, UniformTypeKVCacheSpecs):
                spec = spec.kv_cache_specs[name]
            if spec.block_size != 128:
                raise ValueError("shared-only Mooncake requires block_size=128 in every local group")
            if getattr(spec, "store_on_host", False):
                host_groups.add(gid)
    if len(host_groups) != 1:
        raise ValueError("shared-only Mooncake requires exactly one Host KV group")
    return next(iter(host_groups))


def make_dsa_cache_groups(block_ids, block_sizes) -> tuple[DsaCacheGroupBlocks, ...]:
    if not isinstance(block_sizes, (list, tuple)) or len(block_sizes) != len(block_ids):
        raise ValueError("DSA group block sizes must cover all manager groups")
    return tuple(
        DsaCacheGroupBlocks(gid, size, ids) for gid, (size, ids) in enumerate(zip(block_sizes, block_ids, strict=True))
    )


def parse_dsa_source_groups(params, block_ids) -> tuple[DsaCacheGroupBlocks, ...]:
    version = params.get("dsa_protocol_version")
    if isinstance(version, bool) or not isinstance(version, int) or version != DSA_GROUPED_TRANSFER_VERSION:
        raise ValueError(
            "shared-only Mooncake requires DSA protocol v2; upgrade all Prefill workers and schedulers first"
        )
    return make_dsa_cache_groups(block_ids, params.get("dsa_group_block_sizes"))
