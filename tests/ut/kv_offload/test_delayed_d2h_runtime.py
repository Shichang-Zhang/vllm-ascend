# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.ut.kv_offload.test_delayed_d2h_executor import cpu_pinned_allocation, make_executor  # noqa: F401
from tests.ut.kv_offload.test_mooncake_dsa_decode_worker import make_receive_worker
from tests.ut.worker.test_device_metadata_runner import (  # noqa: F401
    _make_execute_runner,
    _make_executor,
    forward_patches,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p import mooncake_connector
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_dsa_decode_worker import DsaDecodeWorkerState
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_dsa_metadata import (
    DsaConnectorMetadata,
    DsaDecodeCommand,
    DsaDecodeResultKind,
    DsaLocalResult,
    DsaLocalResultKind,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload import delayed_d2h_executor
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.delayed_d2h_runtime import DelayedD2HRuntime
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import SparseKVOffloadManager
from vllm_ascend.worker import model_runner_v1


def setup_runtime(ready=100, rank=0, size=1):
    executor = make_executor(rank=rank, size=size)
    executor.planner.release(np.array([0, 2]), np.array([10, 20]))
    state = DsaDecodeWorkerState(tp_rank=rank, block_size=128)
    state.apply_metadata(DsaConnectorMetadata(decode_commands=(DsaDecodeCommand("request", 1, ready),)))
    manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
    manager.topk_buffer_size = 256
    manager.block_size = 128
    manager.kv_cache_config = SimpleNamespace(num_blocks=16)
    manager.topk_buffers_k = [executor.components[0].resident]
    trace = []

    def restore(sources, destinations, lengths):
        trace.append(("restore", sources.tolist(), destinations.tolist(), lengths.tolist()))
        for component in executor.components:
            for source, destination, length in zip(sources, destinations, lengths, strict=True):
                host = component.host.view(-1, component.host.shape[-1])
                resident = component.resident.view(-1, component.resident.shape[-1])
                resident[destination : destination + length].copy_(host[source : source + length])

    def synchronize():
        trace.append("compute-drain")
        executor.runtime.drain()

    manager._copy_sfa_local_ranges = restore
    runtime = DelayedD2HRuntime(
        executor, restore_rows=manager.restore_delayed_d2h_rows, synchronize_compute=synchronize
    )
    return runtime, executor, state, manager, trace


def prepare(runtime, computed, *, ids=("request",), table=None, writes=None):
    count = len(ids)
    if table is None:
        table = np.array([[5, 2, 9, 7, 11]] * count)
    if writes is None:
        writes = np.full(count, 7)
    runtime.prepare_frontier(np.array(computed, dtype=np.int64))
    capacity = runtime.executor.planner.capacity
    slots = np.zeros(capacity, dtype=np.int32)
    generations = np.zeros(capacity, dtype=np.int64)
    prefixes = np.full(capacity, -1, dtype=np.int32)
    runtime.prepare_batch(
        ids,
        block_table=table,
        max_write_tokens=writes,
        slots=slots,
        generations=generations,
        stable_prefixes=prefixes,
        padded_requests=capacity,
    )
    return slots, generations, prefixes


def test_full_dense_block_copied_only_after_retire_and_last_request_released_without_forward():
    runtime, executor, state, _, trace = setup_runtime(124)
    for component in executor.components:
        component.resident.fill_(-9)
        component.host.fill_(-1)
        component.host[5, :124] = torch.arange(124, dtype=torch.float32).view(-1, 1, 1)
    runtime.begin_step(state)
    slots, _, prefixes = prepare(runtime, [124])
    assert prefixes.tolist() == [0, 0, 0]
    assert slots.tolist() == [0, 4, 5]
    for component in executor.components:
        torch.testing.assert_close(component.resident[0, :124], component.host[5, :124])
        component.resident[0, 124:128] = torch.arange(124, 128, dtype=torch.float32).view(-1, 1, 1)
    runtime.finish_forward()
    runtime.begin_step(state)
    prepare(runtime, [128])
    for component in executor.components:
        component.resident[0, 128:135].fill_(-7)
        assert torch.all(component.host[5, 124:] == -1)
    runtime.finish_forward()
    state.apply_metadata(DsaConnectorMetadata(decode_commands=(DsaDecodeCommand("request", 1, release=True),)))
    runtime.begin_step(state)
    for component in executor.components:
        torch.testing.assert_close(component.host[5], component.resident[0, :128])
        assert torch.all(component.host[2] == -1)
    assert runtime.slots.bound_slots() == {}
    assert trace[-1] == "compute-drain"
    results = state.take_results()
    assert {result.kind for result in results} == {
        DsaDecodeResultKind.PREFIX_READY,
        DsaDecodeResultKind.RELEASE_COMPLETE,
    }
    assert all(result.stable_prefix == 128 for result in results)


@pytest.mark.parametrize("computed,expected", [(127, (5 * 128, 0, 127)), (385, (7 * 128, 384, 1))])
def test_new_row_restore_uses_its_host_group_physical_ids_and_correct_dense_or_ring_offset(computed, expected):
    runtime, executor, state, _, trace = setup_runtime(computed)
    runtime.begin_step(state)
    _, _, prefixes = prepare(runtime, [computed])
    assert trace == [("restore", [expected[0]], [expected[1]], [expected[2]])]
    assert prefixes[0] == computed // 128 * 128
    assert executor.planner.inflight is None


def test_block_aligned_received_prompt_can_recompute_last_token_with_conservative_geometry():
    runtime, executor, state, _, _ = setup_runtime(256)
    runtime.begin_step(state)
    _, _, prefixes = prepare(runtime, [255])
    assert prefixes[0] == 128
    assert state.acquire("request").stable_prefix == 256
    runtime.finish_forward()
    runtime.begin_step(state)
    prepare(runtime, [256])
    runtime.finish_forward()
    runtime.begin_step(state)
    assert state.acquire("request").stable_prefix == 256
    assert state.take_results() == ()


def test_async_frontier_matches_previous_corrected_batch_with_rejection_reorder_and_new_request():
    runtime, _, state, _, _ = setup_runtime()
    runtime.begin_step(state)
    runtime.prepare_frontier(np.array([100, 200]))
    runtime.prepare_frontier(
        np.array([204, 104, 300]),
        previous_positions=np.array([1, 0, -1]),
        previous_drafts=np.array([3, 3, 0]),
        valid_sampled_counts=np.array([1, 2, 0]),
    )
    assert runtime.computed_tokens.tolist() == [202, 101, 300]
    runtime.prepare_frontier(
        np.array([108, 208]),
        previous_positions=np.array([1, 0]),
        previous_drafts=np.array([3, 3, 0]),
        valid_sampled_counts=np.array([4, 1, 1]),
    )
    assert runtime.computed_tokens.tolist() == [102, 206]


def test_unscheduled_rows_remain_owned_and_reordered_requests_keep_slots_and_generations():
    runtime, _, state, _, _ = setup_runtime()
    state.apply_metadata(DsaConnectorMetadata(decode_commands=(DsaDecodeCommand("second", 2, 256),)))
    runtime.begin_step(state)
    slots, generations, prefixes = prepare(runtime, [100, 256], ids=("request", "second"))
    assert slots[:2].tolist() == [0, 1]
    runtime.finish_forward()
    runtime.begin_step(state)
    slots, generations, prefixes = prepare(runtime, [256], ids=("second",))
    assert slots[0] == 1 and generations[0] == 2 and prefixes[0] == 256
    assert runtime.slots.bound_slots() == {"request": 0, "second": 1}
    runtime.finish_forward()
    runtime.begin_step(state)
    slots, generations, prefixes = prepare(runtime, [256, 100], ids=("second", "request"))
    assert slots[:2].tolist() == [1, 0]
    assert generations[:2].tolist() == [2, 1]
    assert prefixes[:2].tolist() == [256, 0]


def test_row_capacity_cannot_recycle_unscheduled_request():
    runtime, _, state, _, _ = setup_runtime()
    state.apply_metadata(
        DsaConnectorMetadata(
            decode_commands=tuple(
                DsaDecodeCommand(name, generation) for generation, name in ((2, "second"), (3, "third"), (4, "fourth"))
            )
        )
    )
    runtime.begin_step(state)
    prepare(runtime, [0, 0, 0], ids=("request", "second", "third"))
    runtime.finish_forward()
    runtime.begin_step(state)
    with pytest.raises(RuntimeError, match="remain owned"):
        prepare(runtime, [0], ids=("fourth",))
    assert runtime.slots.bound_slots() == {"request": 0, "second": 1, "third": 2}


def test_unverified_initial_frontier_and_mtp_write_overlap_fail_closed():
    runtime, executor, state, _, _ = setup_runtime(0)
    runtime.begin_step(state)
    with pytest.raises(RuntimeError, match="verified Host"):
        prepare(runtime, [100])
    assert runtime.slots.bound_slots() == {}
    prepare(runtime, [0])
    runtime.finish_forward()
    runtime.begin_step(state)
    with pytest.raises(RuntimeError, match="overlap"):
        prepare(runtime, [250], writes=np.array([7]))
    assert executor.planner.inflight is None


def test_retire_failure_does_not_release_or_publish_and_shutdown_does_not_claim_success():
    runtime, executor, state, _, _ = setup_runtime(0)
    runtime.begin_step(state)
    prepare(runtime, [0])
    runtime.finish_forward()
    runtime.begin_step(state)
    prepare(runtime, [128])
    runtime.finish_forward()
    state.apply_metadata(DsaConnectorMetadata(decode_commands=(DsaDecodeCommand("request", 1, release=True),)))
    executor._complete_event.fail_wait = True
    with pytest.raises(RuntimeError, match="TP completion failed"):
        runtime.begin_step(state)
    assert runtime.slots.bound_slots() == {"request": 0}
    assert state.take_results() == ()
    runtime.close()
    assert runtime.slots.bound_slots() == {"request": 0}
    assert state.take_results() == ()
    runtime.close()


@pytest.mark.parametrize("fail", [False, True])
def test_shutdown_drains_decode_runtime_before_unregister_and_retains_registration_on_error(monkeypatch, fail):
    worker, _, _, _ = make_receive_worker()
    trace = []
    worker.kv_send_thread = None
    worker.kv_recv_thread = SimpleNamespace(
        request_queue=SimpleNamespace(join=lambda: trace.append("receive-drain"), put=lambda _: None),
        join=lambda: None,
        executor=SimpleNamespace(shutdown=lambda wait: None),
    )

    def close():
        trace.append("runtime-drain")
        if fail:
            raise RuntimeError("DMA drain failed")

    monkeypatch.setattr(
        mooncake_connector,
        "get_sparse_kv_offload_manager",
        lambda: SimpleNamespace(delayed_d2h_runtime=SimpleNamespace(close=close)),
    )
    monkeypatch.setattr(mooncake_connector.global_te, "unregister_buffer", lambda: trace.append("unregister"))
    if fail:
        with pytest.raises(RuntimeError, match="DMA drain failed"):
            worker.shutdown()
    else:
        worker.shutdown()
    assert trace == ["receive-drain", "runtime-drain", *([] if fail else ["unregister"])]


def test_actual_runner_empty_step_retires_before_update_and_still_returns_connector_results(request, monkeypatch):
    trace = request.getfixturevalue("forward_patches")
    runner = _make_execute_runner(_make_executor(False, trace), trace)
    runtime, executor, state, _, _ = setup_runtime()
    runner._delayed_d2h_runtime = runtime
    begin_step = runtime.begin_step

    def begin(state):
        trace.events.append("retire")
        begin_step(state)

    runtime.begin_step = begin
    connector = SimpleNamespace(
        prepare_delayed_decode_step=lambda _: state,
        handle_preemptions=lambda _: trace.events.append("handle-preemptions"),
    )
    monkeypatch.setattr(model_runner_v1, "has_kv_transfer_group", lambda: True)
    monkeypatch.setattr(model_runner_v1, "get_kv_transfer_group", lambda: connector)
    runner._update_states.side_effect = lambda _: trace.events.append("update-states")
    runner.kv_connector_no_forward = lambda *args: "connector-output"
    scheduler_output = SimpleNamespace(total_num_scheduled_tokens=0, kv_connector_metadata=DsaConnectorMetadata())
    assert runner.execute_model(scheduler_output) == "connector-output"
    assert trace.events == ["retire", "handle-preemptions", "update-states"]


def test_actual_runner_metadata_hook_uses_host_group_and_full_mtp_write_budget(monkeypatch):
    runtime, executor, state, manager, trace = setup_runtime(127)
    runtime.begin_step(state)
    runtime.prepare_frontier(np.array([127]))
    runner = model_runner_v1.NPUModelRunner.__new__(model_runner_v1.NPUModelRunner)
    runner._delayed_d2h_runtime = runtime
    runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[object(), object()])
    runner.sparse_kv_offload_enabled = True
    runner.sparse_kv_offload_config = SimpleNamespace(use_fused_copy_sfa=True)
    runner.sparse_kv_offload_manager = manager
    manager.d2h_host_group_id = 1
    runner._offload_stable_prefixes = SimpleNamespace(np=np.full(3, -1, dtype=np.int32))
    runner._offload_pool_slots = SimpleNamespace(np=np.zeros(3, dtype=np.int32))
    runner._offload_pool_generations = SimpleNamespace(np=np.zeros(3, dtype=np.int64))
    runner.input_batch = SimpleNamespace(
        req_ids=["request"],
        block_table=[
            SimpleNamespace(get_numpy_array=lambda: np.full((1, 5), -1)),
            SimpleNamespace(get_numpy_array=lambda: np.array([[5, 2, 9, 7, 11]])),
        ],
    )
    runner.query_start_loc = runner._offload_req_ids_tensor = runner._offload_token_to_req = None
    runner.num_spec_tokens = 3
    runner.device_metadata_executor = None

    class AfterRuntimeHook:
        def numpy(self):
            raise RuntimeError("runtime hook completed")

    runner.optimistic_seq_lens_cpu = AfterRuntimeHook()
    monkeypatch.setattr(model_runner_v1, "update_sparse_kv_offload_metadata", lambda *args: None)
    with pytest.raises(RuntimeError, match="runtime hook completed"):
        runner._build_attention_metadata(4, 1, 4, num_scheduled_tokens_np=np.array([4]))
    assert trace == [("restore", [5 * 128], [0], [127])]
    assert runner._offload_stable_prefixes.np[0] == 0
    assert not runner._copy_sfa_need_eager_tail_restore
    runtime.finish_forward()
    runtime.begin_step(state)
    # Reject a subsequent target+MTP=7 write when only 6 tail slots remain.
    runtime.prepare_frontier(np.array([250]))
    with pytest.raises(RuntimeError, match="overlap"):
        runner._build_attention_metadata(4, 1, 4, num_scheduled_tokens_np=np.array([4]))
    assert executor.planner.inflight is None


def test_tp_out_of_order_notify_completion_preserves_identical_free_order_and_future_d2h_plan(monkeypatch):
    barrier = Barrier(2, timeout=5)
    statuses = [None, None]

    def gather(output, status, group):
        statuses[group.rank] = status
        barrier.wait()
        output[:] = statuses
        barrier.wait()

    monkeypatch.setattr(delayed_d2h_executor.dist, "all_gather_object", gather)
    runtimes, states, notifications = [], [], []
    for rank in range(2):
        runtime, _, state, _, _ = setup_runtime(0, rank=rank, size=2)
        _, _, receive, _ = make_receive_worker(rank)
        first = replace(receive, notify_only=True, main_host_block_ids=(), indexer_hbm_block_ids=())
        second = replace(first, request_id="second", generation=2)
        state.apply_metadata(
            DsaConnectorMetadata(
                requests=(first, second),
                decode_commands=(DsaDecodeCommand("second", 2), DsaDecodeCommand("third", 3)),
            )
        )
        runtimes.append(runtime)
        states.append(state)
        notifications.append((first, second))

    def begin_all():
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda pair: pair[0].begin_step(pair[1]), zip(runtimes, states)))

    def finish_notification(rank, index):
        command = notifications[rank][index]
        states[rank].receive_finished(
            command,
            DsaLocalResult(
                command.request_id, rank, DsaLocalResultKind.RECEIVE_COMPLETE, generation=command.generation
            ),
        )

    begin_all()
    for runtime, state in zip(runtimes, states, strict=True):
        prepare(runtime, [0, 0, 0], ids=("request", "second", "third"))
        runtime.finish_forward()
        state.apply_metadata(
            DsaConnectorMetadata(
                decode_commands=(
                    DsaDecodeCommand("request", 1, release=True),
                    DsaDecodeCommand("second", 2, release=True),
                )
            )
        )
    finish_notification(0, 0)
    finish_notification(1, 1)
    begin_all()
    assert all(runtime.slots.bound_slots() == {"request": 0, "second": 1, "third": 2} for runtime in runtimes)
    assert all(state.take_results() == () for state in states)
    finish_notification(0, 1)
    finish_notification(1, 0)
    begin_all()
    for runtime, state in zip(runtimes, states, strict=True):
        assert runtime.slots.bound_slots() == {"third": 2}
        assert len(state.take_results()) == 2
        state.apply_metadata(DsaConnectorMetadata(decode_commands=(DsaDecodeCommand("fourth", 4),)))
        slots, _, _ = prepare(runtime, [0], ids=("fourth",))
        assert slots[0] == 0
        runtime.finish_forward()
    begin_all()
    for runtime in runtimes:
        prepare(runtime, [128], ids=("fourth",))
        runtime.finish_forward()
    begin_all()
    assert all(state.acquire("fourth").stable_prefix == 128 for state in states)
