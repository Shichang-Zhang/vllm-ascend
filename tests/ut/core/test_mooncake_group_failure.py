# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus

from tests.ut.core.test_mooncake_decode_cache_guard import make_manager, make_request, setup_owner
from tests.ut.kv_offload.test_mooncake_dsa_lifecycle import output
from vllm_ascend.core.mooncake_group_failure import install_mooncake_group_failure_handler
from vllm_ascend.core.recompute_scheduler import AsyncRecomputeScheduler, RecomputeScheduler
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_dsa_metadata import DsaDecodeResult, DsaDecodeResultKind
from vllm_ascend.patch.platform.patch_kv_delivery_preemption import KVDeliveryScheduler


def make_scheduler(manager, cls=Scheduler):
    scheduler = cls.__new__(cls)
    scheduler.connector = SimpleNamespace(dsa_grouped_transfer_enabled=True)
    scheduler.kv_cache_manager = manager
    scheduler.block_size = 2
    scheduler.recompute_kv_load_failures = False
    scheduler.running, scheduler.skipped_waiting = [], []
    install_mooncake_group_failure_handler(scheduler)
    return scheduler


@pytest.mark.parametrize("group", [0, 1, 2])
@pytest.mark.parametrize("async_load", [False, True])
def test_failure_in_any_group_truncates_at_first_bad_logical_block(group, async_load):
    manager, request = make_manager(groups=3), make_request()
    manager.allocate_slots(request, 8, delay_cache_blocks=async_load)
    request.num_computed_tokens = 8
    blocks = manager.get_block_ids(request.request_id)
    scheduler = make_scheduler(manager)
    pool_free = manager.block_pool.get_num_free_blocks()
    invalid = {blocks[group][1], blocks[(group + 1) % 3][2]}
    affected, tokens, evicted = scheduler._update_requests_with_invalid_blocks(
        [request], invalid, {request.request_id: 2}, evict_blocks=not async_load
    )
    assert affected == {request.request_id} and tokens == 4
    assert request.num_computed_tokens == 2
    assert evicted == (set() if async_load else {block for ids in blocks for block in ids[1:]})
    assert manager.get_block_ids(request.request_id) == blocks
    assert manager.block_pool.get_num_free_blocks() == pool_free


def test_only_external_interval_is_scanned_and_generator_is_supported():
    manager, request = make_manager(groups=3), make_request()
    manager.allocate_slots(request, 8, delay_cache_blocks=True)
    request.num_computed_tokens = 8
    blocks = manager.get_block_ids(request.request_id)
    scheduler = make_scheduler(manager)
    assert scheduler._update_requests_with_invalid_blocks(iter([request]), {blocks[2][3]}, {request.request_id: 2}) == (
        set(),
        0,
        set(),
    )
    assert request.num_computed_tokens == 8


def test_failed_uncached_receive_preserves_shared_prefix_hashes_and_references():
    manager, first = make_manager(groups=3), make_request()
    manager.allocate_slots(first, 6, delay_cache_blocks=True)
    manager.cache_blocks(first, 2)
    second = make_request("second")
    hits, length, _ = manager.get_computed_blocks(second)
    assert length == 2
    manager.allocate_slots(second, 4, num_new_computed_tokens=length, new_computed_blocks=hits, delay_cache_blocks=True)
    second.num_computed_tokens = 6
    second.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    blocks = manager.get_block_ids(second.request_id)
    scheduler = make_scheduler(manager)
    scheduler.skipped_waiting = [second]
    assert scheduler._handle_invalid_blocks({blocks[2][1]}, {}) == {second.request_id}
    assert manager.get_computed_blocks(make_request("third"))[1] == 2
    assert all(group[0].ref_cnt == 2 for group in hits.blocks)
    manager.free(second)
    assert all(group[0].ref_cnt == 1 for group in hits.blocks)
    assert manager.get_computed_blocks(make_request("fourth"))[1] == 2


def test_shared_bad_cached_block_fails_all_users_and_evicts_all_group_suffixes():
    manager, first = make_manager(groups=3), make_request()
    manager.allocate_slots(first, 6)
    first.num_computed_tokens = 6
    second = make_request("second")
    hits, length, _ = manager.get_computed_blocks(second)
    assert length == 6
    manager.allocate_slots(second, 1, num_new_computed_tokens=length, new_computed_blocks=hits)
    second.num_computed_tokens = 6
    scheduler = make_scheduler(manager)
    scheduler.running = [first, second]
    invalid = {manager.get_block_ids(first.request_id)[1][1]}
    assert scheduler._handle_invalid_blocks(invalid, {}) == {first.request_id, second.request_id}
    assert first.num_computed_tokens == second.num_computed_tokens == 2
    assert manager.get_computed_blocks(make_request("third"))[1] == 2
    # Eviction removes hashes, never releases still-owned physical blocks.
    assert all(block.ref_cnt == 2 for group in hits.blocks for block in group)


def test_group_failure_finish_requests_waits_for_both_decode_release_acknowledgements():
    manager, connector, request = setup_owner(groups=3)
    scheduler = make_scheduler(manager, AsyncRecomputeScheduler)
    scheduler.connector = connector
    request.status, request.num_computed_tokens = RequestStatus.RUNNING, 6
    scheduler.running, scheduler.requests = [request], {request.request_id: request}
    scheduler.waiting = scheduler.skipped_waiting = MagicMock()
    scheduler._inflight_prefills = {request}
    scheduler._connector_finished = lambda req: connector.request_finished(req, manager.get_block_ids(req.request_id))
    scheduler.encoder_cache_manager = MagicMock()
    scheduler.ec_connector = None
    scheduler.finished_req_ids, scheduler.finished_req_ids_dict = set(), None
    scheduler.defer_block_free = False
    blocks = manager.get_block_ids(request.request_id)
    free_before = manager.block_pool.get_num_free_blocks()
    failed = scheduler._handle_invalid_blocks({blocks[2][1]}, {})
    assert failed == {request.request_id}
    scheduler.finish_requests(failed, RequestStatus.FINISHED_ERROR)
    connector.build_connector_meta(None)
    assert request.request_id in scheduler.requests
    generation = connector._decode_ledger.generation(request.request_id)
    for rank in (0, 1):
        scheduler._update_from_kv_xfer_finished(
            output(
                decode=(
                    DsaDecodeResult(
                        request.request_id,
                        generation,
                        rank,
                        DsaDecodeResultKind.RELEASE_COMPLETE,
                        0,
                    ),
                )
            )
        )
        if rank == 0:
            assert manager.get_block_ids(request.request_id) == blocks
            assert manager.block_pool.get_num_free_blocks() == free_before
    assert request.request_id not in scheduler.requests
    assert manager.block_pool.get_num_free_blocks() == free_before + sum(map(len, blocks))


@pytest.mark.parametrize("cls", [Scheduler, AsyncScheduler, RecomputeScheduler, AsyncRecomputeScheduler])
def test_all_actual_scheduler_constructors_install_group_handler_without_delayed_d2h(cls, monkeypatch):
    manager = make_manager(groups=3)
    connector = SimpleNamespace(
        dsa_grouped_transfer_enabled=True, delayed_d2h_enabled=False, requires_kv_delivery=False
    )
    config = SimpleNamespace(kv_transfer_config=SimpleNamespace(is_kv_producer=False))

    def initialize_base(self, *args, **kwargs):
        self.connector, self.kv_cache_manager = connector, manager
        self.vllm_config, self.block_size = config, 2
        self.num_spec_tokens, self.recompute_kv_load_failures = 0, False
        self.parallel_config = SimpleNamespace(pipeline_parallel_size=1)

    monkeypatch.setattr(KVDeliveryScheduler.__bases__[0], "__init__", initialize_base)
    monkeypatch.setattr(
        "vllm_ascend.patch.platform.patch_balance_schedule.init_ascend_config",
        lambda _: SimpleNamespace(
            scheduler_config=SimpleNamespace(short_request_first_config=SimpleNamespace(enabled=False))
        ),
    )
    monkeypatch.setattr(
        "vllm_ascend.patch.platform.patch_balance_schedule._balance_scheduling_enabled", lambda _: False
    )
    scheduler = cls(config, None, None, 2)
    assert scheduler._update_requests_with_invalid_blocks == scheduler._mooncake_group_failure_handler.update_requests
    request = make_request()
    manager.allocate_slots(request, 6, delay_cache_blocks=True)
    request.num_computed_tokens = 6
    invalid = {manager.get_block_ids(request.request_id)[2][0]}
    assert scheduler._update_requests_with_invalid_blocks([request], invalid, {})[0] == {request.request_id}


def test_install_is_idempotent_and_disabled_mode_keeps_upstream_handler():
    scheduler = make_scheduler(make_manager(groups=3))
    original = scheduler._update_requests_with_invalid_blocks
    install_mooncake_group_failure_handler(scheduler)
    assert scheduler._update_requests_with_invalid_blocks == original
    disabled = SimpleNamespace(connector=None, _update_requests_with_invalid_blocks=original)
    install_mooncake_group_failure_handler(disabled)
    assert disabled._update_requests_with_invalid_blocks == original


@pytest.mark.parametrize("fault", ["recompute", "geometry"])
def test_incompatible_scheduler_configuration_is_rejected_before_admission(fault):
    scheduler = SimpleNamespace(
        connector=SimpleNamespace(dsa_grouped_transfer_enabled=True),
        kv_cache_manager=make_manager(groups=3),
        recompute_kv_load_failures=fault == "recompute",
        block_size=4 if fault == "geometry" else 2,
    )
    with pytest.raises(ValueError):
        install_mooncake_group_failure_handler(scheduler)
