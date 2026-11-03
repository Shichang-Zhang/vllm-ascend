# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit layer ownership before sparse KV cache planning/allocation."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from vllm.v1.kv_cache_interface import KVCacheConfig, UniformTypeKVCacheSpecs

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec


@dataclass(frozen=True)
class SfaLayerResidency:
    name: str
    owner_name: str
    is_draft: bool
    offloaded: bool


@dataclass(frozen=True)
class SfaResidencyPlan:
    layers: tuple[SfaLayerResidency, ...]
    groups: tuple[tuple[str, tuple[str, ...]], ...]

    @property
    def offloaded_names(self) -> tuple[str, ...]:
        return tuple(layer.name for layer in self.layers if layer.offloaded)

    @property
    def resident_names(self) -> tuple[str, ...]:
        return tuple(layer.name for layer in self.layers if not layer.offloaded)

    @property
    def draft_names(self) -> tuple[str, ...]:
        return tuple(layer.name for layer in self.layers if layer.is_draft)


@dataclass(frozen=True)
class SfaCacheLayer:
    name: str
    group_id: int
    store_on_host: bool
    is_indexer: bool


@dataclass(frozen=True)
class SfaCacheLayout:
    layers: tuple[SfaCacheLayer, ...]
    host_group_id: int
    block_size: int


def build_sfa_cache_layout(config: KVCacheConfig, plan: SfaResidencyPlan) -> SfaCacheLayout:
    """Validate the shared-only physical layout before allocating any memory.

    Full-length caches may share a logical KV group, but every layer needs
    private storage. A single Host group is supported, with independently
    numbered resident/indexer groups. HMA tensor aliasing would invalidate
    both the Host budget and rank-local registration, so reject it explicitly.
    """
    expected = {layer.name: layer.offloaded for layer in plan.layers}
    main_names: set[str] = set()
    seen: set[str] = set()
    layers = []
    host_groups: set[int] = set()
    block_sizes: set[int] = set()
    for group_id, group in enumerate(config.kv_cache_groups):
        for name in group.layer_names:
            if name in seen:
                raise ValueError(f"SFA cache layer occurs in multiple groups: {name}")
            seen.add(name)
            spec = group.kv_cache_spec
            if isinstance(spec, UniformTypeKVCacheSpecs):
                spec = spec.kv_cache_specs[name]
            is_indexer = isinstance(spec, AscendSFAIndexerCacheSpec)
            if not is_indexer:
                if not isinstance(spec, AscendMLAAttentionSpec) or name not in expected:
                    raise ValueError(f"shared-only cache has an unclassified main layer: {name}")
                if spec.store_on_host != expected[name]:
                    raise ValueError(f"SFA cache placement disagrees with the layer residency plan: {name}")
                main_names.add(name)
            on_host = getattr(spec, "store_on_host", False)
            if on_host:
                host_groups.add(group_id)
            layers.append(SfaCacheLayer(name, group_id, on_host, is_indexer))
            block_sizes.add(spec.block_size)
    if main_names != expected.keys():
        raise ValueError("SFA cache specs omit layers from the residency plan")
    if block_sizes != {128}:
        raise ValueError("shared-only Copy-SFA requires block_size=128 for every cache group")
    if len(host_groups) != 1:
        raise ValueError("shared-only Copy-SFA requires exactly one Host KV group")
    physical_names = []
    for tensor in config.kv_cache_tensors:
        if len(tensor.shared_by) != 1:
            raise ValueError(
                "shared-only Copy-SFA requires private per-layer cache storage; HMA aliasing is unsupported"
            )
        physical_names.extend(tensor.shared_by)
    if len(physical_names) != len(seen) or set(physical_names) != seen:
        raise ValueError("SFA cache tensors must cover each layer exactly once")
    return SfaCacheLayout(tuple(layers), next(iter(host_groups)), 128)


def bind_sfa_cache_groups(implementations: Mapping[str, Any], layout: SfaCacheLayout) -> None:
    """Bind main and indexer groups by actual registered cache-layer identity."""
    layers = {layer.name: layer for layer in layout.layers}
    bindings = []
    for name, impl in implementations.items():
        main = layers[name]
        if main.is_indexer:
            raise ValueError(f"SFA main attention cannot bind an indexer cache: {name}")
        indexer_group = None
        if impl.has_indexer:
            indexer = layers.get(impl.indexer.k_cache.prefix)
            if indexer is None or not indexer.is_indexer:
                raise ValueError(f"SFA indexer cache is missing from the actual cache groups: {name}")
            indexer_group = indexer.group_id
        bindings.append((impl, main.group_id, indexer_group))
    for impl, main_group, indexer_group in bindings:
        impl.sfa_main_group_id = main_group
        impl.sfa_indexer_group_id = indexer_group


def bind_sfa_layer_residency(
    implementations: Mapping[str, Any],
    *,
    draft_layer_names: frozenset[str],
    shared_only: bool,
) -> SfaResidencyPlan:
    """Bind shared selection owners in model execution order.

    ``draft_layer_names`` comes from actual drafter-module membership, not
    layer-index/name heuristics: draft models may start numbering at zero or
    append an MTP layer to the target's numbering. Reusing a selection buffer
    for a later owner starts another group, matching the execution ordering.
    Validate the complete mapping before mutating any attention instance.
    """
    if not draft_layer_names.issubset(implementations):
        raise ValueError("draft layer membership references an unknown SFA layer")
    owners: dict[int, str] = {}
    members: dict[str, list[str]] = {}
    layers = []
    for name, impl in implementations.items():
        if not isinstance(name, str) or not name:
            raise ValueError("SFA layer names must be nonempty strings")
        if shared_only and getattr(impl, "layer_name", None) not in (None, name):
            raise ValueError("SFA attention layer name differs from the forward-context identity")
        shared = impl.topk_indices_buffer
        skip_topk = impl.skip_topk
        if not isinstance(skip_topk, bool):
            raise TypeError("SFA skip_topk must be bool")
        if skip_topk:
            if shared is None or shared.data_ptr() not in owners:
                raise ValueError(f"SFA shared layer {name!r} precedes or lacks its indexer owner")
            owner_name = owners[shared.data_ptr()]
            if (name in draft_layer_names) != (owner_name in draft_layer_names):
                raise ValueError("SFA selection sharing cannot cross target/draft forward lifetimes")
            members[owner_name].append(name)
        else:
            owner_name = name
            members[name] = []
            if shared is not None:
                if shared.numel() == 0:
                    raise ValueError("SFA selection buffer must have nonzero storage")
                owners[shared.data_ptr()] = name
        is_draft = name in draft_layer_names
        layers.append(SfaLayerResidency(name, owner_name, is_draft, not shared_only or skip_topk or is_draft))
    if shared_only and not any(layer.name != layer.owner_name and not layer.is_draft for layer in layers):
        raise ValueError("offload_shared_indexer_only requires at least one target shared-indexer layer")
    plan = SfaResidencyPlan(tuple(layers), tuple((name, tuple(group)) for name, group in members.items() if group))
    for layer in plan.layers:
        impl = implementations[layer.name]
        impl.layer_name = layer.name
        impl.lim_full_resident = not layer.offloaded
        impl.lim_is_draft = layer.is_draft
        impl.lim_indexer_owner = implementations[layer.owner_name]
        impl.lim_share_members = tuple(implementations[name] for name in members.get(layer.name, ()))
    return plan
