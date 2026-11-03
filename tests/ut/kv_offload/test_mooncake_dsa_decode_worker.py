# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue
import threading
from dataclasses import replace

import pytest

from tests.ut.kv_offload.test_mooncake_dsa_lifecycle import delayed_scheduler
from tests.ut.kv_offload.test_mooncake_dsa_scheduler import _request
from tests.ut.kv_offload.test_mooncake_dsa_shared_pool import make_worker
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_dsa_decode_worker import DsaDecodeWorkerState
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_dsa_metadata import (
    DsaConnectorMetadata,
    DsaDecodeCommand,
    DsaDecodeResultKind,
    DsaLocalResult,
    DsaLocalResultKind,
    DsaTransferPhase,
    DsaWorkerResultMetadata,
)


def metadata(*commands, requests=()):
    return DsaConnectorMetadata(requests=requests, decode_commands=commands)


def make_state(ready=130):
    state = DsaDecodeWorkerState(tp_rank=0, block_size=128)
    state.apply_metadata(metadata(DsaDecodeCommand("request", 1, ready)))
    return state


def release(state):
    state.apply_metadata(metadata(DsaDecodeCommand("request", 1, release=True)))


def make_receive_worker(rank=0):
    worker, thread, command = make_worker(rank=rank, ptp=8, dcp=8, dtp=2)
    state = DsaDecodeWorkerState(tp_rank=rank, block_size=4)
    worker._dsa_decode_state = state
    worker._dsa_decode_receives = queue.SimpleQueue()
    command = replace(command, generation=1)
    return worker, thread, command, state


def take_tasks(thread):
    tasks = []
    while not thread.request_queue.empty():
        task = thread.request_queue.get_nowait()
        thread._mark_request_task_submitted(task)
        tasks.append(task)
    return tasks


def test_no_receive_request_requires_restored_residency_before_publication():
    state = make_state()
    request = state.acquire("request")
    assert request.host_ready_tokens == 130 and request.stable_prefix == 128
    assert not request.resident_bound
    with pytest.raises(RuntimeError, match="unready"):
        state.publish("request", 1, 256)
    with pytest.raises(RuntimeError, match="generation"):
        state.bind_resident("request", 2)
    state.bind_resident("request", 1)
    state.publish("request", 1, 256)
    state.publish("request", 1, 384)
    state.publish("request", 1, 128)
    (result,) = state.take_results()
    assert result.kind is DsaDecodeResultKind.PREFIX_READY and result.stable_prefix == 384
    assert state.take_results() == ()
    assert state.acquire("request").resident_bound


def test_recomputed_last_prompt_block_never_retracts_verified_host_prefix():
    state = make_state(256)
    state.bind_resident("request", 1)
    state.publish("request", 1, 128)
    assert state.take_results() == ()
    assert state.acquire("request").stable_prefix == 256
    release(state)
    state.confirm_drained("request", 1)
    assert state.take_results()[0].stable_prefix == 256


@pytest.mark.parametrize("prefix", [-1, 1, True, 1.5])
def test_invalid_completion_cannot_advance_prefix(prefix):
    state = make_state()
    state.bind_resident("request", 1)
    with pytest.raises(ValueError):
        state.publish("request", 1, prefix)
    assert state.acquire("request").stable_prefix == 128
    assert state.take_results() == ()


def test_cancel_requires_explicit_drain_and_preserves_final_completed_prefix():
    state = make_state()
    state.bind_resident("request", 1)
    with pytest.raises(RuntimeError, match="live"):
        state.confirm_drained("request", 1)
    release(state)
    with pytest.raises(RuntimeError, match="terminal"):
        state.acquire("request")
    state.publish("request", 1, 256)
    (pending,) = state.releasable()
    assert pending.resident_bound and pending.stable_prefix == 256
    state.confirm_drained("request", 1)
    results = state.take_results()
    assert [result.kind for result in results] == [
        DsaDecodeResultKind.PREFIX_READY,
        DsaDecodeResultKind.RELEASE_COMPLETE,
    ]
    assert all(result.stable_prefix == 256 for result in results)
    assert state.releasable() == ()
    state.confirm_drained("request", 1)
    assert state.take_results() == ()


def test_failed_dma_retains_owner_and_cannot_release_or_publish():
    state = make_state()
    state.bind_resident("request", 1)
    state.fail("request", 1, "copy failed")
    release(state)
    assert state.releasable() == ()
    with pytest.raises(RuntimeError, match="failed"):
        state.confirm_drained("request", 1)
    with pytest.raises(RuntimeError, match="failed"):
        state.publish("request", 1, 256)
    with pytest.raises(RuntimeError, match="previous owner"):
        state.apply_metadata(metadata(DsaDecodeCommand("request", 2)))
    (result,) = state.take_results()
    assert result.kind is DsaDecodeResultKind.FAILED and result.stable_prefix == 128


def test_watermark_rejects_late_registration_without_request_id_tombstones():
    state = make_state()
    release(state)
    state.confirm_drained("request", 1)
    state.take_results()
    state.apply_metadata(metadata(DsaDecodeCommand("request", 1, 130)))
    with pytest.raises(RuntimeError, match="no registered"):
        state.acquire("request")
    state.apply_metadata(metadata(DsaDecodeCommand("request", 2)))
    state.apply_metadata(metadata(DsaDecodeCommand("request", 1, release=True)))
    state.publish("request", 1, 1024)
    state.confirm_drained("request", 1)
    assert state.acquire("request").generation == 2
    assert state.acquire("request").stable_prefix == 0
    assert state.take_results() == ()
    with pytest.raises(RuntimeError, match="ahead"):
        state.publish("request", 3, 128)


def test_same_batch_generation_order_differs_from_request_order_and_validation_is_atomic():
    state = DsaDecodeWorkerState(tp_rank=0, block_size=128)
    state.apply_metadata(metadata(DsaDecodeCommand("a", 2), DsaDecodeCommand("z", 1)))
    assert state.acquire("a").generation == 2
    assert state.acquire("z").generation == 1
    with pytest.raises(RuntimeError, match="release has no registered owner"):
        state.apply_metadata(metadata(DsaDecodeCommand("b", 3), DsaDecodeCommand("future", 4, release=True)))
    with pytest.raises(RuntimeError, match="no registered"):
        state.acquire("b")
    state.apply_metadata(metadata(DsaDecodeCommand("b", 3)))
    assert state.acquire("b").generation == 3


@pytest.mark.parametrize("rank", [0, 1])
def test_actual_p8_dcp8_d2_receive_then_decode_publication_survives_receive_tracker_removal(rank):
    worker, thread, command, state = make_receive_worker(rank)
    batch = metadata(DsaDecodeCommand("request", 1), requests=(command,))
    # The early runner hook and normal connector dispatch may see the same
    # command. Registration is idempotent and does not reset residency.
    worker.prepare_decode_step(batch)
    worker.start_load_kv(batch)
    tasks = take_tasks(thread)
    with pytest.raises(RuntimeError, match="still receiving"):
        state.acquire("request")
    for task in reversed(tasks):
        thread._handle_dsa_request(task)
    assert "request" not in worker._dsa_active_commands
    # Background callbacks alone cannot mutate the execution-thread state.
    with pytest.raises(RuntimeError, match="still receiving"):
        state.acquire("request")
    wire = worker.build_connector_worker_meta()
    assert wire.results[0].generation == 1 and wire.results[0].tp_rank == rank
    assert wire.decode_results == ()
    assert state.acquire("request").host_ready_tokens == 32
    state.bind_resident("request", 1)
    state.publish("request", 1, 36)
    release(state)
    state.confirm_drained("request", 1)
    wire = worker.build_connector_worker_meta()
    assert wire.results == ()
    assert {result.kind for result in wire.decode_results} == {
        DsaDecodeResultKind.PREFIX_READY,
        DsaDecodeResultKind.RELEASE_COMPLETE,
    }
    assert worker.build_connector_worker_meta() is None


@pytest.mark.parametrize("failed", [False, True])
def test_cancel_during_receive_defers_decode_ack_until_all_endpoint_tasks_drain(failed):
    worker, thread, command, state = make_receive_worker()
    batch = metadata(DsaDecodeCommand("request", 1), requests=(command,))
    worker.start_load_kv(batch)
    tasks = take_tasks(thread)
    if failed:
        thread.engine.batch_transfer_sync_read.return_value = -1
    worker.prepare_decode_step(metadata(DsaDecodeCommand("request", 1, release=True)))
    thread._handle_dsa_request(tasks[-1])
    assert state.releasable() == ()
    with pytest.raises(RuntimeError, match="receiving"):
        state.confirm_drained("request", 1)
    for task in tasks[:-1]:
        thread._handle_dsa_request(task)
    wire = worker.build_connector_worker_meta()
    assert not wire.decode_results
    (pending,) = state.releasable()
    assert not pending.resident_bound
    assert pending.stable_prefix == (0 if failed else 32)
    state.confirm_drained("request", 1)
    assert worker.build_connector_worker_meta().decode_results[0].kind is DsaDecodeResultKind.RELEASE_COMPLETE


def test_cancel_blocked_read_does_not_acknowledge_while_transfer_can_still_write():
    worker, thread, command, state = make_receive_worker()
    worker.start_load_kv(metadata(DsaDecodeCommand("request", 1), requests=(command,)))
    tasks = take_tasks(thread)
    entered, unblock = threading.Event(), threading.Event()

    def read(*args):
        entered.set()
        assert unblock.wait(5)
        return 0

    thread.engine.batch_transfer_sync_read.side_effect = read
    running = threading.Thread(target=thread._handle_dsa_request, args=(tasks[0],))
    running.start()
    try:
        assert entered.wait(5)
        worker.start_load_kv(
            DsaConnectorMetadata(
                cancelled_requests=("request",), decode_commands=(DsaDecodeCommand("request", 1, release=True),)
            )
        )
        for task in tasks[1:]:
            thread._handle_dsa_request(task)
        assert worker.build_connector_worker_meta() is None
        assert state.releasable() == ()
    finally:
        unblock.set()
        running.join(5)
    assert not running.is_alive()
    worker.build_connector_worker_meta()
    assert len(state.releasable()) == 1


def test_apc_notification_allows_decode_but_holds_release_until_notification_thread_finishes():
    worker, thread, command, state = make_receive_worker()
    command = replace(command, notify_only=True, main_host_block_ids=(), indexer_hbm_block_ids=())
    batch = metadata(DsaDecodeCommand("request", 1, 32), requests=(command,))
    worker.start_load_kv(batch)
    state.bind_resident("request", 1)
    state.publish("request", 1, 36)
    release(state)
    assert state.releasable() == ()
    for task in take_tasks(thread):
        thread._handle_dsa_request(task)
    wire = worker.build_connector_worker_meta()
    assert wire.results == ()
    assert wire.decode_results[0].stable_prefix == 36
    state.confirm_drained("request", 1)
    assert worker.build_connector_worker_meta().decode_results[0].stable_prefix == 36
    thread.engine.batch_transfer_sync_read.assert_not_called()


def test_receive_result_identity_and_disabled_worker_mismatch_are_rejected():
    worker, _, command, state = make_receive_worker()
    batch = metadata(DsaDecodeCommand("request", 1), requests=(command,))
    worker.prepare_decode_step(batch)
    result = DsaLocalResult("request", 1, DsaLocalResultKind.RECEIVE_COMPLETE, generation=1)
    with pytest.raises(ValueError, match="local decode owner"):
        state.receive_finished(command, result)
    with pytest.raises(RuntimeError, match="still receiving"):
        state.acquire("request")
    worker._dsa_decode_state = None
    with pytest.raises(RuntimeError, match="every worker"):
        worker.prepare_decode_step(batch)


def test_notification_failure_does_not_invalidate_apc_content():
    worker, _, command, state = make_receive_worker()
    command = replace(command, notify_only=True, main_host_block_ids=(), indexer_hbm_block_ids=())
    state.apply_metadata(metadata(DsaDecodeCommand("request", 1, 32), requests=(command,)))
    state.receive_finished(
        command,
        DsaLocalResult("request", 0, DsaLocalResultKind.TRANSFER_FAILED, DsaTransferPhase.MAIN_D2RH, generation=1),
    )
    assert state.acquire("request").host_ready_tokens == 32


def test_decode_results_aggregate_across_tp_without_second_receive_event():
    merged = DsaWorkerResultMetadata()
    for rank in range(2):
        worker, _, _, state = make_receive_worker(rank)
        worker.prepare_decode_step(metadata(DsaDecodeCommand("request", 1, 32)))
        state.bind_resident("request", 1)
        state.publish("request", 1, 36)
        release(state)
        state.confirm_drained("request", 1)
        merged = merged.aggregate(worker.build_connector_worker_meta())
    assert merged.results == ()
    assert len(merged.decode_results) == 4
    assert {result.tp_rank for result in merged.decode_results} == {0, 1}


def test_real_scheduler_cancel_before_allocation_sends_notification_without_decode_owner():
    scheduler = delayed_scheduler()
    request = _request()
    assert scheduler.request_finished(request, ()) == (False, None)
    batch = scheduler.build_connector_meta(None)
    assert batch.requests[0].generation == 0 and batch.requests[0].notify_only
    assert not batch.decode_commands
    worker, thread, _ = make_worker(ptp=2, dcp=1, dtp=2)
    worker._dsa_decode_state = DsaDecodeWorkerState(tp_rank=0, block_size=2)
    worker._dsa_decode_receives = queue.SimpleQueue()
    worker.start_load_kv(batch)
    tasks = take_tasks(thread)
    assert tasks
    for task in tasks:
        thread._handle_dsa_request(task)
    thread.engine.batch_transfer_sync_read.assert_not_called()
    assert thread._send_done_recv_signal.call_count == len(tasks)
    assert worker.build_connector_worker_meta() is None
    assert worker._dsa_decode_state.releasable() == ()


@pytest.mark.parametrize("new_generation", [False, True])
def test_late_receive_dispatch_cannot_rewrite_completed_or_reused_resident_owner(new_generation):
    worker, thread, command, state = make_receive_worker()
    batch = metadata(DsaDecodeCommand("request", 1), requests=(command,))
    worker.start_load_kv(batch)
    for task in take_tasks(thread):
        thread._handle_dsa_request(task)
    worker.build_connector_worker_meta()
    if new_generation:
        release(state)
        state.confirm_drained("request", 1)
        state.take_results()
        state.apply_metadata(metadata(DsaDecodeCommand("request", 2, 32)))
    generation = 2 if new_generation else 1
    state.bind_resident("request", generation)
    thread.engine.batch_transfer_sync_read.reset_mock()
    worker.start_load_kv(batch)
    assert take_tasks(thread) == []
    assert not worker._dsa_active_commands
    assert state.acquire("request").generation == generation
    thread.engine.batch_transfer_sync_read.assert_not_called()


def test_late_cancel_cannot_cancel_new_generation_receive():
    worker, thread, command, state = make_receive_worker()
    worker.start_load_kv(metadata(DsaDecodeCommand("request", 1), requests=(command,)))
    for task in take_tasks(thread):
        thread._handle_dsa_request(task)
    worker.build_connector_worker_meta()
    release(state)
    state.confirm_drained("request", 1)
    state.take_results()
    second = replace(command, generation=2)
    worker.start_load_kv(metadata(DsaDecodeCommand("request", 2), requests=(second,)))
    tasks = take_tasks(thread)
    worker.start_load_kv(
        DsaConnectorMetadata(
            cancelled_requests=("request",), decode_commands=(DsaDecodeCommand("request", 1, release=True),)
        )
    )
    assert not worker._dsa_cancel_events["request"].is_set()
    for task in tasks:
        thread._handle_dsa_request(task)
    wire = worker.build_connector_worker_meta()
    assert wire.results[0].generation == 2 and wire.results[0].kind is DsaLocalResultKind.RECEIVE_COMPLETE
    assert state.acquire("request").generation == 2


def test_replay_between_callback_active_pop_and_completion_queue_cannot_dispatch_twice():
    worker, thread, command, state = make_receive_worker()
    batch = metadata(DsaDecodeCommand("request", 1), requests=(command,))
    worker.start_load_kv(batch)

    class ReplayAfterPop(dict):
        def pop(self, key, default=None):
            value = super().pop(key, default)
            # Deterministically reproduce the race: main-thread replay after
            # callback removal but before callback puts its completion queue.
            worker.start_load_kv(batch)
            return value

    worker._dsa_active_commands = ReplayAfterPop(worker._dsa_active_commands)
    tasks = take_tasks(thread)
    for task in tasks:
        thread._handle_dsa_request(task)
    assert take_tasks(thread) == []
    assert not worker._dsa_active_commands
    assert not state.should_dispatch_receive(command)
    wire = worker.build_connector_worker_meta()
    assert len(wire.results) == 1
    assert state.acquire("request").host_ready_tokens == 32
