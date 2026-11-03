# SPDX-License-Identifier: Apache-2.0
"""Generation-safe Host publication and decode release acknowledgements."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from tests.ut.kv_offload.test_mooncake_dsa_scheduler import _blocks, _request, _scheduler
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_dsa_lifecycle import DsaDecodeLedger
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_dsa_metadata import (
    DsaConnectorMetadata,
    DsaDecodeCommand,
    DsaDecodeResult,
    DsaDecodeResultKind,
    DsaLocalResult,
    DsaLocalResultKind,
    DsaTransferPhase,
    DsaWorkerResultMetadata,
)


def report(rank, prefix, *, generation=1, kind=DsaDecodeResultKind.PREFIX_READY, request="request", error=None):
    return DsaDecodeResult(request, generation, rank, kind, prefix, error)


def make_ledger(*, pending=False):
    ledger = DsaDecodeLedger(tp_size=2, block_size=128)
    assert ledger.allocate("request", host_ready_tokens=130, receive_pending=pending) == 1
    assert ledger.build_commands() == (DsaDecodeCommand("request", 1, 130),)
    return ledger


def test_prefix_requires_every_rank_and_does_not_regress_on_delayed_reports():
    ledger = make_ledger()
    assert ledger.completed_prefix("request") == 128
    ledger.apply_results((report(0, 384),))
    assert ledger.completed_prefix("request") == 128
    ledger.apply_results((report(1, 256),))
    assert ledger.completed_prefix("request") == 256
    ledger.apply_results((report(1, 384), report(0, 256)))
    assert ledger.completed_prefix("request") == 384
    assert ledger.completed_prefix("unknown") == 0
    assert ledger.take_released() == set()


def test_receive_completion_keeps_decode_owned_and_release_waits_for_all_ranks():
    ledger = make_ledger(pending=True)
    ledger.receive_finished("request", 1, host_ready_tokens=260, success=True)
    assert ledger.completed_prefix("request") == 256
    assert ledger.generation("request") == 1
    assert ledger.request_release("request")
    assert ledger.request_release("request")
    assert ledger.build_commands() == (DsaDecodeCommand("request", 1, release=True),)
    assert ledger.build_commands() == ()
    ledger.apply_results((report(0, 256, kind=DsaDecodeResultKind.RELEASE_COMPLETE),))
    assert ledger.take_released() == set()
    ledger.apply_results((report(1, 256, kind=DsaDecodeResultKind.RELEASE_COMPLETE),))
    assert ledger.take_released() == {"request"}
    assert ledger.take_released() == set()
    assert ledger.generation("request") is None


@pytest.mark.parametrize("success", [False, True])
def test_release_ack_before_receive_completion_cannot_release_destination(success):
    ledger = make_ledger(pending=True)
    ledger.request_release("request")
    ledger.build_commands()
    ledger.apply_results(tuple(report(rank, 128, kind=DsaDecodeResultKind.RELEASE_COMPLETE) for rank in range(2)))
    assert ledger.take_released() == set()
    ledger.receive_finished("request", 1, host_ready_tokens=260, success=success)
    assert ledger.take_released() == {"request"}


def test_failed_receive_cannot_publish_speculative_prefix_reports():
    ledger = make_ledger(pending=True)
    ledger.apply_results((report(0, 512), report(1, 512)))
    assert ledger.completed_prefix("request") == 128
    ledger.receive_finished("request", 1, host_ready_tokens=512, success=False)
    ledger.apply_results((report(0, 512), report(1, 512)))
    assert ledger.completed_prefix("request") == 128


def test_new_generation_ignores_late_prefix_and_release_from_previous_owner():
    ledger = make_ledger()
    ledger.request_release("request")
    ledger.build_commands()
    old = tuple(report(rank, 128, kind=DsaDecodeResultKind.RELEASE_COMPLETE) for rank in range(2))
    ledger.apply_results(old)
    assert ledger.take_released() == {"request"}
    assert ledger.allocate("request", host_ready_tokens=0, receive_pending=False) == 2
    ledger.build_commands()
    ledger.apply_results((*old, report(0, 99968), report(1, 99968)))
    assert ledger.completed_prefix("request") == 0
    assert ledger.take_released() == set()
    assert ledger.generation("request") == 2


def test_failed_copy_never_publishes_or_releases_and_poison_survives_retry():
    ledger = make_ledger()
    with pytest.raises(RuntimeError, match="copy failed"):
        ledger.apply_results((report(0, 256), report(1, 256, kind=DsaDecodeResultKind.FAILED, error="copy failed")))
    assert ledger.completed_prefix("request") == 128
    ledger.request_release("request")
    ledger.build_commands()
    with pytest.raises(RuntimeError, match="copy failed"):
        ledger.apply_results((report(0, 256, kind=DsaDecodeResultKind.RELEASE_COMPLETE),))
    assert ledger.take_released() == set()


@pytest.mark.parametrize("invalid", [report(2, 256), report(1, 256, generation=2), report(1, 255)])
def test_invalid_batch_is_atomic_for_prefix_and_release(invalid):
    ledger = make_ledger()
    ledger.request_release("request")
    ledger.build_commands()
    with pytest.raises(ValueError):
        ledger.apply_results((report(0, 256, kind=DsaDecodeResultKind.RELEASE_COMPLETE), invalid))
    assert ledger.completed_prefix("request") == 128
    ledger.apply_results((report(1, 128, kind=DsaDecodeResultKind.RELEASE_COMPLETE),))
    assert ledger.take_released() == set()


def test_premature_or_conflicting_release_is_rejected():
    ledger = make_ledger()
    with pytest.raises(RuntimeError, match="before a release command"):
        ledger.apply_results((report(0, 128, kind=DsaDecodeResultKind.RELEASE_COMPLETE),))
    ledger.apply_results((report(0, 256),))
    ledger.request_release("request")
    ledger.build_commands()
    with pytest.raises(RuntimeError, match="rolled back"):
        ledger.apply_results((report(0, 128, kind=DsaDecodeResultKind.RELEASE_COMPLETE),))
    ledger.apply_results((report(0, 256, kind=DsaDecodeResultKind.RELEASE_COMPLETE),))
    with pytest.raises(RuntimeError, match="after worker release"):
        ledger.apply_results((report(0, 384),))
    ledger.apply_results((report(0, 128),))


@pytest.mark.parametrize("duplicate_prefix", [128, 384])
def test_repeated_release_cannot_change_final_prefix_across_batches(duplicate_prefix):
    ledger = make_ledger()
    ledger.request_release("request")
    ledger.build_commands()
    released = report(0, 256, kind=DsaDecodeResultKind.RELEASE_COMPLETE)
    ledger.apply_results((released, report(1, 512)))
    assert ledger.completed_prefix("request") == 256
    ledger.apply_results((released,))
    with pytest.raises(RuntimeError, match="duplicate decode release"):
        ledger.apply_results((replace(released, stable_prefix=duplicate_prefix),))
    assert ledger.completed_prefix("request") == 256
    assert ledger.take_released() == set()
    ledger.apply_results((report(1, 512, kind=DsaDecodeResultKind.RELEASE_COMPLETE),))
    assert ledger.take_released() == {"request"}


def delayed_scheduler():
    scheduler = _scheduler()
    scheduler._decode_ledger = DsaDecodeLedger(tp_size=2, block_size=2)
    scheduler._dsa_cached_prefixes = {}
    return scheduler


def allocate(scheduler, *, cached=0):
    request = _request()
    external, _ = scheduler.get_num_new_matched_tokens(request, cached)
    scheduler.update_state_after_alloc(request, _blocks([30, 31], [40, 41]), external)
    metadata = scheduler.build_connector_meta(None)
    return request, metadata


def output(*, receive=(), decode=(), raw_receive=(), raw_send=()):
    return SimpleNamespace(
        kv_connector_worker_meta=DsaWorkerResultMetadata(receive, decode),
        finished_recving=set(raw_receive),
        finished_sending=set(raw_send),
    )


def receive_results(generation=1, *, failed=False):
    return tuple(
        DsaLocalResult(
            "request",
            rank,
            DsaLocalResultKind.TRANSFER_FAILED if failed else DsaLocalResultKind.RECEIVE_COMPLETE,
            DsaTransferPhase.MAIN_D2RH if failed else None,
            generation=generation,
        )
        for rank in range(2)
    )


def test_scheduler_decode_lifetime_outlives_receive_tracker():
    scheduler = delayed_scheduler()
    request, metadata = allocate(scheduler)
    assert metadata.requests[0].generation == metadata.decode_commands[0].generation == 1
    received = output(receive=receive_results())
    scheduler.update_connector_output(received)
    assert received.finished_recving == {"request"}
    assert "request" not in scheduler._dsa_requests
    assert scheduler.owns_decode_blocks("request")
    assert scheduler.get_completed_host_prefix("request") == 4
    scheduler.update_connector_output(output(decode=(report(0, 6), report(1, 6))))
    assert scheduler.get_completed_host_prefix("request") == 6
    assert scheduler.request_finished(request, ([30, 31], [40, 41])) == (True, None)
    (command,) = scheduler.build_connector_meta(None).decode_commands
    assert command.release
    first = output(decode=(report(0, 6, kind=DsaDecodeResultKind.RELEASE_COMPLETE),))
    scheduler.update_connector_output(first)
    assert not first.finished_sending and not first.finished_recving
    last = output(decode=(report(1, 6, kind=DsaDecodeResultKind.RELEASE_COMPLETE),))
    scheduler.update_connector_output(last)
    assert last.finished_sending == {"request"}
    assert not last.finished_recving
    assert not scheduler.owns_decode_blocks("request")


@pytest.mark.parametrize("failed", [False, True])
@pytest.mark.parametrize("receive_first", [False, True])
def test_cancel_during_receive_needs_both_receive_drain_and_decode_ack(failed, receive_first):
    scheduler = delayed_scheduler()
    request, _ = allocate(scheduler)
    assert scheduler.request_finished(request, ()) == (True, None)
    metadata = scheduler.build_connector_meta(None)
    assert metadata.cancelled_requests == ("request",)
    assert metadata.decode_commands[0].release
    messages = [
        output(receive=receive_results(failed=failed)),
        output(decode=tuple(report(rank, 0, kind=DsaDecodeResultKind.RELEASE_COMPLETE) for rank in range(2))),
    ]
    if not receive_first:
        messages.reverse()
    for index, message in enumerate(messages):
        scheduler.update_connector_output(message)
        assert message.finished_sending == ({"request"} if index else set())
        assert not message.finished_recving


def test_apc_notification_only_still_allocates_a_decode_lifetime():
    scheduler = delayed_scheduler()
    request, metadata = allocate(scheduler, cached=4)
    assert metadata.requests[0].notify_only
    assert metadata.decode_commands == (DsaDecodeCommand("request", 1, 4),)
    assert scheduler.get_completed_host_prefix("request") == 4
    assert "request" not in scheduler._dsa_requests
    assert scheduler.request_finished(request, ()) == (True, None)


def test_cached_request_without_remote_receive_gets_registration():
    scheduler = delayed_scheduler()
    request = _request()
    request.kv_transfer_params = {}
    assert scheduler.get_num_new_matched_tokens(request, 2) == (0, False)
    scheduler.update_state_after_alloc(request, _blocks([30], [40]), 0)
    metadata = scheduler.build_connector_meta(None)
    assert not metadata.requests
    assert metadata.decode_commands == (DsaDecodeCommand("request", 1, 2),)
    assert scheduler._dsa_cached_prefixes == {}


def test_rejected_before_allocation_has_no_decode_resources_to_hold():
    scheduler = delayed_scheduler()
    request = _request()
    assert scheduler.request_finished(request, ()) == (False, None)
    metadata = scheduler.build_connector_meta(None)
    assert metadata.requests[0].notify_only
    assert not metadata.decode_commands
    assert scheduler._dsa_cached_prefixes == {}


def test_raw_notifications_and_late_receive_generation_cannot_free_current_lifetime():
    scheduler = delayed_scheduler()
    allocate(scheduler)
    message = output(receive=receive_results(0), raw_receive=("request",), raw_send=("request",))
    scheduler.update_connector_output(message)
    assert not message.finished_recving and not message.finished_sending
    assert not scheduler._dsa_requests["request"].results_by_rank
    assert scheduler.owns_decode_blocks("request")
    message.kv_connector_worker_meta = None
    message.finished_recving = {"request"}
    scheduler.update_connector_output(message)
    assert not message.finished_recving


def test_new_wire_messages_aggregate_distinct_receive_and_decode_events():
    received = receive_results()[0]
    prefix = report(0, 256)
    release = report(0, 256, kind=DsaDecodeResultKind.RELEASE_COMPLETE)
    merged = DsaWorkerResultMetadata((received,), (prefix,)).aggregate(DsaWorkerResultMetadata((), (release,)))
    assert merged.results == (received,)
    assert merged.decode_results == (prefix, release)
    assert DsaWorkerResultMetadata((received, replace(received, generation=2))).results == (
        received,
        replace(received, generation=2),
    )
    with pytest.raises(ValueError, match="conflicting decode result"):
        merged.aggregate(DsaWorkerResultMetadata(decode_results=(replace(prefix, stable_prefix=384),)))
    bind = DsaDecodeCommand("request", 1, 130)
    terminal = DsaDecodeCommand("request", 1, release=True)
    assert DsaConnectorMetadata(decode_commands=(terminal, bind, bind)).decode_commands == (bind, terminal)
    with pytest.raises(ValueError, match="conflicting decode command"):
        DsaConnectorMetadata(decode_commands=(bind, replace(bind, host_ready_tokens=256)))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"generation": 0},
        {"generation": True},
        {"stable_prefix": -1},
        {"stable_prefix": 1.5},
        {"kind": "prefix_ready"},
        {"error": "unexpected"},
        {"kind": DsaDecodeResultKind.FAILED},
    ],
)
def test_decode_result_strict_validation(kwargs):
    values = dict(
        request_id="request", generation=1, tp_rank=0, kind=DsaDecodeResultKind.PREFIX_READY, stable_prefix=128
    )
    values.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        DsaDecodeResult(**values)
