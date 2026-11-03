# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.attention.sfa_kv_offload import AscendSFAKVOffloadImpl
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sfa_layer_residency import bind_sfa_layer_residency
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


def impl(*, shared=None, skip=False, name=None):
    return SimpleNamespace(topk_indices_buffer=shared, skip_topk=skip, layer_name=name)


def make_layers():
    # The target uses a single reusable selection buffer across owners. The
    # MTP has its own buffer, even when its layer index happens to be zero.
    shared = torch.empty(4, 2048, dtype=torch.int32)
    draft = torch.empty_like(shared)
    return {
        "model.layers.2.attn": impl(shared=shared),
        "model.layers.3.attn": impl(shared=shared, skip=True),
        "model.layers.6.attn": impl(shared=shared),
        "model.layers.7.attn": impl(shared=shared, skip=True),
        "model.layers.8.attn": impl(shared=shared, skip=True),
        "model.layers.12.attn": impl(),
        "draft.layers.0.attn": impl(shared=draft),
    }


@pytest.mark.parametrize("shared_only", [False, True])
def test_explicit_owner_member_and_draft_placement_with_non_contiguous_layer_names(shared_only):
    layers = make_layers()
    plan = bind_sfa_layer_residency(
        layers, draft_layer_names=frozenset({"draft.layers.0.attn"}), shared_only=shared_only
    )
    assert plan.groups == (
        ("model.layers.2.attn", ("model.layers.3.attn",)),
        ("model.layers.6.attn", ("model.layers.7.attn", "model.layers.8.attn")),
    )
    assert plan.draft_names == ("draft.layers.0.attn",)
    assert plan.resident_names == (
        ("model.layers.2.attn", "model.layers.6.attn", "model.layers.12.attn") if shared_only else ()
    )
    assert plan.offloaded_names == (
        ("model.layers.3.attn", "model.layers.7.attn", "model.layers.8.attn", "draft.layers.0.attn")
        if shared_only
        else tuple(layers)
    )
    for layer in plan.layers:
        assert layers[layer.name].layer_name == layer.name
        assert layers[layer.name].lim_indexer_owner is layers[layer.owner_name]
        assert layers[layer.name].lim_full_resident is (not layer.offloaded)
    assert layers["model.layers.2.attn"].lim_share_members == (layers["model.layers.3.attn"],)
    assert layers["model.layers.12.attn"].lim_share_members == ()


def test_plan_is_immutable_and_draft_runtime_reuse_does_not_change_residency():
    layers = make_layers()
    plan = bind_sfa_layer_residency(layers, draft_layer_names=frozenset({"draft.layers.0.attn"}), shared_only=True)
    with pytest.raises(FrozenInstanceError):
        plan.layers[0].offloaded = True
    draft = layers["draft.layers.0.attn"]
    draft.skip_topk = True
    assert draft.lim_is_draft and not draft.lim_full_resident
    assert "draft.layers.0.attn" in plan.offloaded_names


def test_rebinding_does_not_duplicate_group_members():
    layers = make_layers()
    first = bind_sfa_layer_residency(layers, draft_layer_names=frozenset({"draft.layers.0.attn"}), shared_only=True)
    second = bind_sfa_layer_residency(layers, draft_layer_names=frozenset({"draft.layers.0.attn"}), shared_only=True)
    assert first == second
    assert len(layers["model.layers.6.attn"].lim_share_members) == 2


@pytest.mark.parametrize("with_buffer", [False, True])
def test_orphan_shared_layer_is_rejected_before_any_impl_is_modified(with_buffer):
    layers = make_layers()
    layers["orphan"] = impl(shared=torch.empty(4, 2048) if with_buffer else None, skip=True)
    with pytest.raises(ValueError, match="lacks its indexer owner"):
        bind_sfa_layer_residency(layers, draft_layer_names=frozenset(), shared_only=True)
    assert all(not hasattr(layer, "lim_full_resident") for layer in layers.values())
    assert all(layer.layer_name is None for layer in layers.values())


def test_member_before_owner_is_rejected():
    shared = torch.empty(4, 2048)
    layers = {"member": impl(shared=shared, skip=True), "owner": impl(shared=shared)}
    with pytest.raises(ValueError, match="precedes"):
        bind_sfa_layer_residency(layers, draft_layer_names=frozenset(), shared_only=True)


def test_sharing_selection_across_target_and_draft_is_rejected():
    layers = make_layers()
    with pytest.raises(ValueError, match="cross target/draft"):
        bind_sfa_layer_residency(layers, draft_layer_names=frozenset({"model.layers.3.attn"}), shared_only=True)


def test_draft_may_reuse_buffer_storage_with_its_own_independent_selection():
    layers = make_layers()
    layers["draft.layers.0.attn"].topk_indices_buffer = layers["model.layers.2.attn"].topk_indices_buffer
    plan = bind_sfa_layer_residency(layers, draft_layer_names=frozenset({"draft.layers.0.attn"}), shared_only=True)
    draft = layers["draft.layers.0.attn"]
    assert draft.lim_indexer_owner is draft
    assert not draft.lim_full_resident and draft.lim_is_draft
    assert plan.groups[0] == ("model.layers.2.attn", ("model.layers.3.attn",))


@pytest.mark.parametrize("with_draft", [False, True])
def test_shared_only_without_target_shared_layers_fails_before_memory_allocation(with_draft):
    layers = {"target": impl()}
    draft_names = frozenset()
    if with_draft:
        layers["mtp"] = impl()
        draft_names = frozenset({"mtp"})
    with pytest.raises(ValueError, match="at least one target shared"):
        bind_sfa_layer_residency(layers, draft_layer_names=draft_names, shared_only=True)
    # The old all-offloaded configuration remains valid without sharing.
    assert bind_sfa_layer_residency(layers, draft_layer_names=draft_names, shared_only=False).resident_names == ()


def test_unknown_draft_membership_and_conflicting_name_are_rejected():
    layers = make_layers()
    with pytest.raises(ValueError, match="unknown SFA layer"):
        bind_sfa_layer_residency(layers, draft_layer_names=frozenset({"unknown"}), shared_only=True)
    layers["model.layers.2.attn"].layer_name = "wrong.name"
    with pytest.raises(ValueError, match="forward-context identity"):
        bind_sfa_layer_residency(layers, draft_layer_names=frozenset(), shared_only=True)


def test_actual_runner_uses_module_membership_for_mtp_instead_of_name_or_numeric_index():
    prototypes = make_layers()
    layers = {}
    for name, prototype in prototypes.items():
        attention = torch.nn.Module()
        attention.impl = AscendSFAKVOffloadImpl.__new__(AscendSFAKVOffloadImpl)
        attention.impl.topk_indices_buffer = prototype.topk_indices_buffer
        attention.impl.skip_topk = prototype.skip_topk
        attention.impl.layer_name = None
        layers[name] = attention
    # Deliberately give the real draft a normal target-style index/name.
    draft_attention = layers.pop("draft.layers.0.attn")
    layers["model.layers.0.attn"] = draft_attention
    drafter_model = torch.nn.Module()
    drafter_model.add_module("attention", draft_attention)
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.sparse_kv_offload_enabled = True
    runner.sparse_kv_offload_config = SimpleNamespace(use_fused_copy_sfa=True, offload_shared_indexer_only=True)
    runner.compilation_config = SimpleNamespace(static_forward_context={**layers, "unrelated": torch.nn.Module()})
    runner.drafter = SimpleNamespace(model=drafter_model)
    runner._configure_sparse_layer_residency()
    plan = runner._sfa_layer_residency_plan
    assert plan.draft_names == ("model.layers.0.attn",)
    assert draft_attention.impl.lim_is_draft and not draft_attention.impl.lim_full_resident
    assert len(plan.resident_names) == 3 and len(plan.offloaded_names) == 4


@pytest.mark.parametrize("enabled,fused", [(False, False), (False, True), (True, False)])
def test_runner_preserves_other_modes_without_inspecting_model_modules(enabled, fused):
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.sparse_kv_offload_enabled = enabled
    runner.sparse_kv_offload_config = SimpleNamespace(use_fused_copy_sfa=fused)
    runner._configure_sparse_layer_residency()
    assert not hasattr(runner, "_sfa_layer_residency_plan")
