# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.distributed.kv_transfer.sparse_kv_offload import lim_group_onboard as module
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.lim_group_onboard import LimGroupOnboarder
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sfa_layer_residency import (
    SfaCacheLayer,
    SfaCacheLayout,
    SfaLayerResidency,
    SfaResidencyPlan,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import SparseKVOffloadManager


class Stream:
    def __init__(self, runtime):
        self.runtime, self.queue, self.position = runtime, [], 0

    def wait_event(self, event):
        self.runtime.record(lambda: self.wait_event(event))
        self.runtime.log.append(("wait", self, event))
        event.stream.drain(event.position)

    def drain(self, end):
        while self.position < end:
            action = self.queue[self.position]
            self.position += 1
            action()

    def synchronize(self):
        self.drain(len(self.queue))


class Event:
    def __init__(self, runtime):
        self.runtime = runtime

    def record(self, stream):
        self.runtime.record(lambda: self.record(stream))
        self.runtime.log.append(("record", stream, self))
        self.stream, self.position = stream, len(stream.queue)


class Runtime:
    def __init__(self):
        self.main = Stream(self)
        self.current = self.main
        self.capture = None
        self.log = []

    def record(self, action):
        if self.capture is not None:
            self.capture.append(action)

    def Stream(self, device):
        return Stream(self)

    def Event(self):
        return Event(self)

    def current_stream(self):
        return self.current

    @contextmanager
    def stream(self, stream):
        original, self.current = self.current, stream
        try:
            yield
        finally:
            self.current = original

    def enqueue(self, action):
        stream = self.current
        self.record(lambda: stream.queue.append(action))
        stream.queue.append(action)


def fixture_groups(*, change=None):
    owners = [
        SimpleNamespace(
            layer_name=f"owner{i}",
            skip_topk=False,
            lim_full_resident=True,
            lim_miss_src=torch.tensor([[0, 5], [4, 1], [3, 2]], dtype=torch.int32),
            lim_miss_dst=torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.int32),
            lim_misses=torch.full((3,), 2, dtype=torch.int32),
            lim_topk_misses=torch.full((6,), 2, dtype=torch.int32),
        )
        for i in range(2)
    ]
    names = ("member0", "member1", "member2")
    layers = (
        tuple(SfaLayerResidency(owner.layer_name, owner.layer_name, False, False) for owner in owners)
        + tuple(SfaLayerResidency(name, "owner0" if i < 2 else "owner1", False, True) for i, name in enumerate(names))
        + (SfaLayerResidency("draft", "draft", True, True),)
    )
    plan = SfaResidencyPlan(layers, (("owner0", names[:2]), ("owner1", names[2:]), ("draft", ())))
    hosts = [
        torch.arange(6 * 4 * 16, dtype=torch.float32).reshape(6, 4, 1, 16).to(torch.bfloat16) + 1000 * i
        for i in range(3)
    ]
    ropes = [tensor[..., :8].contiguous() for tensor in hosts]
    manager = SimpleNamespace(
        residency_plan=plan,
        sfa_cache_layout=SfaCacheLayout(tuple(SfaCacheLayer(name, 2, True, False) for name in names), 2, 4),
        topk_buffers_k=[torch.full((3, 16, 1, 16), -7, dtype=torch.bfloat16) for _ in names],
        topk_buffers_v=[torch.full((3, 16, 1, 8), -7, dtype=torch.bfloat16) for _ in names],
        k_caches_cpu=hosts,
        v_caches_cpu=ropes,
        copy_sfa_host_bases=[
            torch.tensor([kv.data_ptr(), rope.data_ptr()], dtype=torch.int64).view(2, 1)
            for kv, rope in zip(hosts, ropes, strict=True)
        ],
        _get_offload_layer_id=names.index,
        block_size=4,
        topk_buffer_size=8,
        ensure_copy_sfa_host_visibility=MagicMock(),
    )
    runtime = Runtime()
    pointer_map = {kv.data_ptr(): i for i, kv in enumerate(hosts)}
    calls = []

    def operation(src, dst, counts, table, pools, pointers, kv, rope, params, geometry):
        calls.append((runtime.current, geometry, table, kv))

        def copy():
            active, width = geometry.tolist()
            assert width == table.shape[1] or active == 0
            index = pointer_map[int(pointers[0])]
            for row in range(active):
                pool = int(pools[row])
                if pool < 0:
                    continue
                for entry in range(int(counts[row])):
                    token, slot = int(src[row, entry]), int(dst[row, entry])
                    physical, offset = int(table[row, token // 4]), token % 4
                    kv[pool, slot].copy_(hosts[index][physical, offset])
                    rope[pool, slot].copy_(ropes[index][physical, offset])

        runtime.enqueue(copy)

    if change is not None:
        change(manager, owners)
    onboarder = LimGroupOnboarder(manager, {owner.layer_name: owner for owner in owners}, operation, runtime=runtime)
    calls.clear()
    md = SimpleNamespace(
        copy_sfa_pool_entries=torch.tensor([2, 0, -1], dtype=torch.int32),
        sfa_cache_group_tables=(
            torch.full((3, 3), 5, dtype=torch.int32),
            torch.full((3, 3), 4, dtype=torch.int32),
            torch.tensor([[1, 3, 0], [2, 5, 1], [0, 1, 2]], dtype=torch.int32),
        ),
    )
    return onboarder, manager, owners, md, runtime, calls


def test_each_owner_has_private_geometry_and_member_waits_only_for_its_copy():
    onboarder, manager, owners, md, runtime, calls = fixture_groups()
    token = object()
    onboarder.launch(owners[0], md, token, capturing=True)
    smaller = SimpleNamespace(
        copy_sfa_pool_entries=md.copy_sfa_pool_entries[:1], sfa_cache_group_tables=md.sfa_cache_group_tables
    )
    onboarder.launch(owners[1], smaller, token, capturing=True)
    assert [call[1].tolist() for call in calls] == [[3, 3], [3, 3], [1, 3]]
    assert all(call[2].data_ptr() == md.sfa_cache_group_tables[2].data_ptr() for call in calls)
    assert all(torch.all(t == -7) for t in manager.topk_buffers_k)
    zeros = onboarder.wait(owners[0], "member0", token)
    assert not zeros[0].any() and not zeros[1].any()
    torch.testing.assert_close(manager.topk_buffers_k[0][2, 0], manager.k_caches_cpu[0][1, 0])
    assert torch.all(manager.topk_buffers_k[1] == -7) and torch.all(manager.topk_buffers_k[2] == -7)
    onboarder.wait(owners[0], "member1", token)
    onboarder.wait(owners[1], "member2", token)
    torch.testing.assert_close(manager.topk_buffers_k[2][2, 1], manager.k_caches_cpu[2][3, 1])
    assert torch.all(manager.topk_buffers_k[2][0] == -7)  # second group has one active request
    assert manager.ensure_copy_sfa_host_visibility.call_count == 2
    assert calls[0][1].data_ptr() != calls[2][1].data_ptr()
    assert runtime.log[0][0] == "record"


def test_captured_operations_replay_without_python_launch_checks_and_switch_capture_sizes(monkeypatch):
    onboarder, manager, owners, md, runtime, _ = fixture_groups()
    geometry_storage = {group.geometry.untyped_storage().data_ptr() for group in onboarder.groups.values()}
    original_fill = torch.Tensor.fill_

    def fill(tensor, value):
        if tensor.untyped_storage().data_ptr() in geometry_storage:
            runtime.record(lambda: original_fill(tensor, value))
        return original_fill(tensor, value)

    monkeypatch.setattr(torch.Tensor, "fill_", fill)
    graphs = {}
    for count in (1, 3):
        runtime.capture = []
        current = SimpleNamespace(
            copy_sfa_pool_entries=md.copy_sfa_pool_entries[:count], sfa_cache_group_tables=md.sfa_cache_group_tables
        )
        token = object()
        onboarder.launch(owners[0], current, token, capturing=True)
        onboarder.wait(owners[0], "member0", token)
        onboarder.wait(owners[0], "member1", token)
        graphs[count], runtime.capture = runtime.capture, None
    # Replay executes the captured geometry writes, records, copies and waits.
    # The production Python launch/wait methods must not be called on replay.
    monkeypatch.setattr(onboarder, "launch", MagicMock(side_effect=AssertionError("Python launch on replay")))
    monkeypatch.setattr(onboarder, "wait", MagicMock(side_effect=AssertionError("Python wait on replay")))
    for count, physical in ((3, 2), (1, 4), (3, 1), (1, 3)):
        md.sfa_cache_group_tables[2][0, 0] = physical
        owners[0].lim_miss_src[0, 0] = 1
        for tensor in manager.topk_buffers_k:
            tensor.fill_(-7)
        for action in graphs[count]:
            action()
        torch.testing.assert_close(manager.topk_buffers_k[0][2, 0], manager.k_caches_cpu[0][physical, 1])
        assert bool(torch.all(manager.topk_buffers_k[0][0] == -7)) == (count == 1)


def test_wrong_forward_or_skipped_owner_cannot_consume_previous_copy_event():
    onboarder, _, owners, md, _, _ = fixture_groups()
    token = object()
    onboarder.launch(owners[0], md, token, capturing=False)
    with pytest.raises(RuntimeError, match="this forward"):
        onboarder.wait(owners[0], "member0", object())
    owners[0].skip_topk = True
    with pytest.raises(RuntimeError, match="unconditional"):
        onboarder.launch(owners[0], md, object(), capturing=False)


def test_runtime_copy_errors_propagate_and_close_drains_before_release():
    onboarder, manager, owners, md, _, _ = fixture_groups()
    onboarder.launch(owners[0], md, object(), capturing=False)
    assert torch.all(manager.topk_buffers_k[0] == -7)
    onboarder.close()
    assert not torch.all(manager.topk_buffers_k[0] == -7)
    onboarder, _, owners, md, _, _ = fixture_groups()
    onboarder.operation = MagicMock(side_effect=RuntimeError("DMA failed"))
    with pytest.raises(RuntimeError, match="DMA failed"):
        onboarder.launch(owners[0], md, object(), capturing=False)


@pytest.mark.parametrize("fault", ["tables", "count", "dtype", "table_rows"])
def test_invalid_forward_geometry_fails_before_side_copy(fault):
    onboarder, _, owners, md, _, calls = fixture_groups()
    if fault == "tables":
        md.sfa_cache_group_tables = None
    elif fault == "count":
        md.copy_sfa_pool_entries = torch.zeros(4, dtype=torch.int32)
    elif fault == "dtype":
        md.copy_sfa_pool_entries = md.copy_sfa_pool_entries.long()
    else:
        md.sfa_cache_group_tables = (*md.sfa_cache_group_tables[:2], md.sfa_cache_group_tables[2][:1])
    with pytest.raises((ValueError, RuntimeError)):
        onboarder.launch(owners[0], md, object(), capturing=False)
    assert not calls


@pytest.mark.parametrize("soc,available", [(module.AscendDeviceType.A2, True), (module.AscendDeviceType.A3, False)])
def test_only_missing_kernel_or_unsupported_soc_falls_back_to_inline(soc, available, monkeypatch):
    monkeypatch.setattr(module, "get_ascend_device_type", lambda: soc)
    monkeypatch.setattr(module, "enable_custom_op", lambda: True)
    monkeypatch.setattr(
        torch.ops._C_ascend, "npu_lim_copy_descriptors", MagicMock() if available else None, raising=False
    )
    assert module.resolve_lim_copy_operator() is None


def test_old_operator_abi_is_rejected_instead_of_silent_fallback(monkeypatch):
    monkeypatch.setattr(module, "get_ascend_device_type", lambda: module.AscendDeviceType.A3)
    monkeypatch.setattr(module, "enable_custom_op", lambda: True)
    monkeypatch.setattr(
        torch.ops._C_ascend,
        "npu_lim_copy_descriptors",
        SimpleNamespace(
            default=SimpleNamespace(
                _schema=SimpleNamespace(arguments=[SimpleNamespace(name="dev_ptrs")]),
            )
        ),
        raising=False,
    )
    with pytest.raises(RuntimeError, match="ABI mismatch"):
        module.resolve_lim_copy_operator()


def test_lazy_extension_is_loaded_before_operator_availability_is_checked(monkeypatch):
    operation = SimpleNamespace(
        _schema=SimpleNamespace(
            arguments=[SimpleNamespace(name=name) for name in module.LIM_COPY_ARGUMENT_NAMES],
        )
    )
    monkeypatch.setattr(module, "get_ascend_device_type", lambda: module.AscendDeviceType.A3)
    monkeypatch.setattr(torch.ops._C_ascend, "npu_lim_copy_descriptors", None, raising=False)

    def enable():
        monkeypatch.setattr(torch.ops._C_ascend, "npu_lim_copy_descriptors", SimpleNamespace(default=operation))

    monkeypatch.setattr(module, "enable_custom_op", enable)
    assert module.resolve_lim_copy_operator() is operation


def test_manager_closes_side_stream_before_host_allocator(monkeypatch):
    manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
    order = []
    manager.lim_group_onboarder = SimpleNamespace(close=lambda: order.append("side-drained"))
    manager._host_kv_allocator = SimpleNamespace(close=lambda: order.append("host-freed"))
    manager.close()
    assert order == ["side-drained", "host-freed"]


@pytest.mark.parametrize("fault", ["head", "dtype", "stride", "capacity", "host", "skip", "group"])
def test_invalid_registered_geometry_is_rejected_before_warmup(fault):
    def change(manager, owners):
        if fault == "head":
            manager.topk_buffers_k[0] = torch.zeros((3, 16, 1, 513), dtype=torch.bfloat16)
        elif fault == "dtype":
            manager.topk_buffers_k[0] = manager.topk_buffers_k[0].float()
        elif fault == "stride":
            manager.topk_buffers_k[0] = manager.topk_buffers_k[0][:, ::2]
        elif fault == "capacity":
            manager.topk_buffers_k[0] = manager.topk_buffers_k[0][:1]
            manager.topk_buffers_v[0] = manager.topk_buffers_v[0][:1]
        elif fault == "host":
            manager.v_caches_cpu[0] = manager.v_caches_cpu[0][:2]
        elif fault == "skip":
            owners[0].skip_topk = True
        else:
            manager.sfa_cache_layout = SfaCacheLayout(manager.sfa_cache_layout.layers, 0, 4)

    with pytest.raises(ValueError):
        fixture_groups(change=change)


def test_draft_group_never_uses_target_side_events():
    def change(manager, owners):
        layers = tuple(
            SfaLayerResidency(
                layer.name, layer.owner_name, layer.name in ("owner0", "member0", "member1"), layer.offloaded
            )
            for layer in manager.residency_plan.layers
        )
        manager.residency_plan = SfaResidencyPlan(layers, manager.residency_plan.groups)

    onboarder, _, owners, md, _, calls = fixture_groups(change=change)
    assert "owner0" not in onboarder.groups
    assert "member0" not in onboarder.member_owners
    owners[0].skip_topk = True
    onboarder.launch(owners[0], md, object(), capturing=True)
    assert calls == []


def test_manager_retains_host_allocation_if_side_drain_fails():
    manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
    manager.lim_group_onboarder = SimpleNamespace(close=MagicMock(side_effect=RuntimeError("stream failed")))
    manager._host_kv_allocator = SimpleNamespace(close=MagicMock())
    with pytest.raises(RuntimeError, match="stream failed"):
        manager.close()
    manager._host_kv_allocator.close.assert_not_called()
