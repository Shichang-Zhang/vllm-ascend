# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Content and failure tests for Mooncake's deferred block-copy executor."""

import ctypes
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Barrier
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload import delayed_d2h_executor as module
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.delayed_d2h import DelayedD2HPlanner
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.delayed_d2h_executor import (
    D2HCopyComponent,
    MooncakeDelayedD2HExecutor,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import SparseKVOffloadManager


class FakeEvent:
    def __init__(self, runtime):
        self.runtime = runtime
        self.records = 0
        self.fail_wait = False

    def record(self, stream):
        self.records += 1
        self.runtime.log.append(("record", self, stream))

    def synchronize(self):
        self.runtime.log.append(("synchronize", self))
        if self.fail_wait:
            raise RuntimeError("test event failure")
        self.runtime.drain()


class FakeStream:
    def __init__(self, runtime):
        self.runtime = runtime

    def wait_event(self, event):
        assert event.records
        self.runtime.log.append(("wait", event))

    def synchronize(self):
        self.runtime.drain()


class FakeRuntime:
    def __init__(self):
        self.log = []
        self.pending = []
        self.capturing = False
        self.compute_stream = FakeStream(self)

    def Stream(self):
        return FakeStream(self)

    def Event(self):
        return FakeEvent(self)

    def current_stream(self):
        return self.compute_stream

    def is_current_stream_capturing(self):
        return self.capturing

    @contextmanager
    def stream(self, stream):
        self.log.append(("enter", stream))
        yield
        self.log.append(("exit", stream))

    def drain(self):
        for callback in self.pending:
            callback()
        self.pending.clear()

    def copy_blocks(self, sources, destinations, lengths):
        self.log.append(("copy", sources.numel()))

        def copy():
            # These addresses are validated views of live CPU tensors, with
            # the exact same byte geometry used by the production kernel.
            for source, destination, size in zip(
                sources.tolist(), destinations.tolist(), lengths.tolist(), strict=True
            ):
                ctypes.memmove(destination, source, size)

        self.pending.append(copy)


@pytest.fixture(autouse=True)
def cpu_pinned_allocation(monkeypatch):
    original = torch.empty

    def empty(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return original(*args, **kwargs)

    monkeypatch.setattr(module.torch, "empty", empty)


def make_executor(rank=0, size=1, components=None):
    planner = DelayedD2HPlanner(capacity=3, block_size=128, hot_tokens=256, host_num_blocks=16)
    planner.bind(np.array([0, 2]), np.array([10, 20]), np.array([0, 256]))
    if components is None:
        components = []
        # Non-contiguous model layer names and different K/V widths.
        for layer in ("layers.2", "layers.9", "mtp.0"):
            for component, width in (("k", 4), ("v", 2)):
                resident = torch.empty((6, 512, 1, width), dtype=torch.bfloat16)
                values = torch.arange(512).view(1, -1, 1, 1) + 512 * torch.arange(6).view(-1, 1, 1, 1)
                resident.copy_(values + len(components) * 4096)
                host = torch.full((16, 128, 1, width), -1, dtype=torch.bfloat16)
                components.append(D2HCopyComponent(f"{layer}.{component}", resident, host))
    runtime = FakeRuntime()
    return MooncakeDelayedD2HExecutor(
        planner,
        components,
        runtime=runtime,
        tp_group=SimpleNamespace(cpu_group=SimpleNamespace(rank=rank)),
        tp_rank=rank,
        tp_size=size,
        copy_blocks=runtime.copy_blocks,
    )


def freeze(executor):
    return executor.planner.plan(
        computed_tokens=np.array([384, 128]),
        pool_slots=np.array([2, 0]),
        generations=np.array([20, 10]),
        max_write_tokens=np.array([7, 7]),
        block_table=np.array([[2, 3, 11], [7, 8, 9]]),
    )


def prefixes(executor):
    return executor.planner.prefixes(np.array([0, 2]), np.array([10, 20])).tolist()


def test_all_layer_local_addresses_copy_correct_contents_only_after_retire():
    executor = make_executor()
    freeze(executor)
    executor.record_source_ready()
    assert executor.launch()
    assert prefixes(executor) == [0, 256]
    for component in executor.components:
        assert torch.all(component.host == -1)
        # Upcoming target/draft writes are disjoint from both frozen blocks.
        component.resident[0, 128:135].fill_(-7)
        component.resident[2, 384:391].fill_(-9)
    assert executor.retire() == 2
    assert prefixes(executor) == [128, 384]
    for component in executor.components:
        torch.testing.assert_close(component.host[7], component.resident[0, :128])
        torch.testing.assert_close(component.host[11], component.resident[2, 256:384])
        assert torch.all(component.host[6] == -1)
    actions = [entry[0] for entry in executor.runtime.log]
    assert actions.index("wait") < actions.index("copy") < actions.index("synchronize")
    assert executor.retire() == 0


def test_inflight_source_event_and_pinned_descriptors_cannot_be_reused():
    executor = make_executor()
    freeze(executor)
    executor.record_source_ready()
    executor.launch()
    event = executor._source_events[executor._inflight_source_slot]
    staging = executor._descriptors_host.clone()
    for _ in range(5):
        executor.record_source_ready()
    assert event.records == 1
    with pytest.raises(RuntimeError, match="already submitted"):
        executor.launch()
    torch.testing.assert_close(staging, executor._descriptors_host)
    executor.retire()


def test_consecutive_batches_reuse_staging_and_rotate_source_events():
    executor = make_executor()
    staging_ptr = executor._descriptors_host.data_ptr()
    freeze(executor)
    executor.record_source_ready()
    executor.launch()
    first_source_slot = executor._inflight_source_slot
    executor.record_source_ready()
    executor.retire()
    first_results = [component.host[7].clone() for component in executor.components]
    executor.planner.plan(
        computed_tokens=np.array([512, 256]),
        pool_slots=np.array([2, 0]),
        generations=np.array([20, 10]),
        max_write_tokens=np.array([7, 7]),
        block_table=np.array([[2, 3, 11, 12], [7, 8, 9, 10]]),
    )
    executor.launch()
    assert executor._inflight_source_slot != first_source_slot
    assert executor._descriptors_host.data_ptr() == staging_ptr
    executor.record_source_ready()
    assert executor.retire() == 2
    assert prefixes(executor) == [256, 512]
    for component, first in zip(executor.components, first_results, strict=True):
        torch.testing.assert_close(component.host[7], first)
        torch.testing.assert_close(component.host[8], component.resident[0, 128:256])
        torch.testing.assert_close(component.host[12], component.resident[2, 384:512])


@pytest.mark.parametrize("failure", ["missing_source", "submission", "completion", "unlaunched"])
def test_failure_preserves_plan_prefix_and_slot_ownership(failure):
    executor = make_executor()
    batch = freeze(executor)
    if failure != "missing_source":
        executor.record_source_ready()
    if failure == "submission":

        def fail(*_args):
            raise RuntimeError("test submission error")

        executor.copy_blocks = fail
    if failure == "completion":
        executor._complete_event.fail_wait = True
    if failure != "unlaunched":
        executor.launch()
    with pytest.raises(RuntimeError, match="TP completion"):
        executor.retire()
    assert executor.planner.inflight is batch
    assert prefixes(executor) == [0, 256]
    with pytest.raises(RuntimeError, match="in-flight"):
        executor.planner.release(np.array([0]), np.array([10]))


@pytest.mark.parametrize("operation", ["record_source_ready", "launch", "retire"])
def test_control_is_forbidden_during_capture(operation):
    executor = make_executor()
    executor.runtime.capturing = True
    with pytest.raises(RuntimeError, match="outside graph capture"):
        getattr(executor, operation)()


def run_tp_retire(monkeypatch, executors):
    barrier = Barrier(len(executors), timeout=5)
    statuses = [None] * len(executors)

    def gather(output, status, group):
        statuses[group.rank] = status
        barrier.wait()
        output[:] = statuses
        barrier.wait()

    monkeypatch.setattr(module.dist, "all_gather_object", gather)

    def retire(executor):
        try:
            return executor.retire()
        except RuntimeError as exc:
            return str(exc)

    with ThreadPoolExecutor(max_workers=len(executors)) as pool:
        return list(pool.map(retire, executors))


def test_tp_nonwriter_waits_for_writer_and_empty_rounds_stay_collective(monkeypatch):
    executors = [make_executor(rank, 2) for rank in range(2)]
    for executor in executors:
        freeze(executor)
        executor.record_source_ready()
        executor.launch()
    assert not executors[1].runtime.pending
    assert executors[1].stream is None
    assert run_tp_retire(monkeypatch, executors) == [2, 2]
    assert [prefixes(executor) for executor in executors] == [[128, 384], [128, 384]]
    assert run_tp_retire(monkeypatch, executors) == [0, 0]


@pytest.mark.parametrize("divergence", ["empty_root", "empty_peer", "epoch", "destinations", "failed_writer"])
def test_tp_divergence_or_failure_never_publishes_on_any_rank(monkeypatch, divergence):
    executors = [make_executor(rank, 2) for rank in range(2)]
    for rank, executor in enumerate(executors):
        if (divergence == "empty_root" and rank == 0) or (divergence == "empty_peer" and rank == 1):
            continue
        if divergence == "epoch" and rank == 1:
            executor.planner._next_epoch = 1
        batch = freeze(executor)
        if divergence == "destinations" and rank == 1:
            batch.destination_blocks.flags.writeable = True
            batch.destination_blocks[0] = 12
            batch.destination_blocks.flags.writeable = False
        executor.record_source_ready()
        executor.launch()
    if divergence == "failed_writer":
        executors[0]._complete_event.fail_wait = True
    results = run_tp_retire(monkeypatch, executors)
    assert all("TP completion" in result for result in results)
    assert [prefixes(executor) for executor in executors] == [[0, 256], [0, 256]]


def test_close_drains_partial_submission_without_publishing():
    executor = make_executor()
    freeze(executor)
    executor.record_source_ready()
    copy = executor.copy_blocks

    def submit_then_fail(*args):
        copy(*args)
        raise RuntimeError("error after enqueue")

    executor.copy_blocks = submit_then_fail
    assert not executor.launch()
    # Compute's final hook must still allow all ranks to reach retire.
    executor.record_source_ready()
    with pytest.raises(RuntimeError, match="submission failed"):
        executor.retire()
    assert executor.runtime.pending
    executor.close()
    assert not executor.runtime.pending
    assert prefixes(executor) == [0, 256]
    executor.close()
    with pytest.raises(RuntimeError, match="closed"):
        executor.launch()


def test_failed_collective_poison_keeps_resources_even_on_retry(monkeypatch):
    executor = make_executor(0, 2)
    freeze(executor)
    executor.record_source_ready()
    executor.launch()

    def fail(*_args, **_kwargs):
        raise RuntimeError("test communication failure")

    monkeypatch.setattr(module.dist, "all_gather_object", fail)
    with pytest.raises(RuntimeError, match="collective failed"):
        executor.retire()
    assert prefixes(executor) == [0, 256]
    with pytest.raises(RuntimeError, match="poisoned"):
        executor.launch()


def test_manager_close_keeps_host_pool_when_copy_stream_drain_fails():
    executor = make_executor()
    manager = object.__new__(SparseKVOffloadManager)
    manager.delayed_d2h_executor = executor
    released = []
    manager._host_kv_allocator = SimpleNamespace(close=lambda: released.append(True))

    def fail():
        raise RuntimeError("stream cannot be drained")

    executor.stream.synchronize = fail
    with pytest.raises(RuntimeError, match="cannot be drained"):
        manager.close()
    assert not released


@pytest.mark.parametrize("field", ["host_capacity", "row_stride", "width", "dtype", "contiguity", "duplicate"])
def test_bad_registered_geometry_fails_before_submission(field):
    executor = make_executor()
    component = executor.components[0]
    resident, host = component.resident, component.host
    if field == "host_capacity":
        host = host[:-1]
    elif field == "row_stride":
        resident = torch.empty((6, 511, 1, 4), dtype=torch.bfloat16)
    elif field == "width":
        host = torch.empty((16, 128, 1, 5), dtype=torch.bfloat16)
    elif field == "dtype":
        host = host.float()
    elif field == "contiguity":
        resident = resident.transpose(0, 1)
    components = [D2HCopyComponent("kv", resident, host)]
    if field == "duplicate":
        components *= 2
    with pytest.raises(ValueError):
        make_executor(components=components)


@pytest.mark.parametrize("mixed_uniform_group", [False, True])
def test_manager_binds_actual_host_group_and_noncontiguous_layer_names(mixed_uniform_group):
    executor = make_executor()
    manager = object.__new__(SparseKVOffloadManager)
    manager._uses_mooncake_host_pool = lambda: True
    manager.use_fused_copy_sfa = True
    manager.sparse_kv_offload_config = SimpleNamespace(keep_device_kv_cache=False)
    manager.offload_layer_names = ["layers.2", "layers.9", "mtp.0"]
    manager.kv_cache_config = SimpleNamespace(
        num_blocks=16,
        kv_cache_groups=[
            SimpleNamespace(layer_names=["indexer"], kv_cache_spec=SimpleNamespace(store_on_host=False)),
            SimpleNamespace(layer_names=manager.offload_layer_names, kv_cache_spec=SimpleNamespace(store_on_host=True)),
        ],
    )
    manager.max_num_reqs, manager.block_size, manager.topk_buffer_size = 1, 128, 256
    manager.topk_buffers_k = [component.resident for component in executor.components[::2]]
    manager.topk_buffers_v = [component.resident for component in executor.components[1::2]]
    manager.k_caches_cpu = [component.host for component in executor.components[::2]]
    manager.v_caches_cpu = [component.host for component in executor.components[1::2]]
    manager._npu_runtime = executor.runtime
    manager.tp_rank, manager.tp_size, manager.tp_group = 0, 1, None
    if mixed_uniform_group:
        specs = dict.fromkeys(
            manager.offload_layer_names,
            AscendMLAAttentionSpec(
                block_size=128, num_kv_heads=1, head_size=6, dtype=torch.bfloat16, store_on_host=True
            ),
        )
        specs["indexer"] = AscendSFAIndexerCacheSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.int8,
            scale_dim=1,
            scale_dtype=torch.float16,
            cache_sparse_li_c8=True,
        )
        uniform = UniformTypeKVCacheSpecs.from_specs(specs)
        assert uniform is not None
        manager.kv_cache_config.kv_cache_groups = [SimpleNamespace(layer_names=list(specs), kv_cache_spec=uniform)]
    manager._initialize_delayed_d2h()
    assert manager.d2h_host_group_id == (0 if mixed_uniform_group else 1)
    assert [component.name for component in manager.delayed_d2h_executor.components] == [
        component.name for component in executor.components
    ]
    manager.kv_cache_config.kv_cache_groups.append(manager.kv_cache_config.kv_cache_groups[-1])
    with pytest.raises(ValueError, match="exactly one Host"):
        manager._initialize_delayed_d2h()
