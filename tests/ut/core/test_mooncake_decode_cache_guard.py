# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real block-pool/APC tests for the Mooncake completed-prefix fence."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.request import Request, RequestStatus

from tests.ut.kv_offload.test_mooncake_dsa_lifecycle import delayed_scheduler, output
from vllm_ascend.core.mooncake_decode_cache_guard import install_mooncake_decode_cache_guard
from vllm_ascend.core.recompute_scheduler import AsyncRecomputeScheduler, RecomputeScheduler
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_dsa_metadata import DsaDecodeResult, DsaDecodeResultKind
from vllm_ascend.patch.platform.patch_kv_delivery_preemption import KVDeliveryScheduler


def make_request(request_id="first"):
    request = Request(
        request_id=request_id,
        prompt_token_ids=list(range(9)),
        sampling_params=SamplingParams(max_tokens=16),
        pooling_params=None,
    )
    # Equal token prefixes have equal hashes; the actual allocator, per-group
    # hash index, refcounts and cache-hit lookup remain real in these tests.
    request.block_hashes = [BlockHash(bytes([i])) for i in range(4)]
    request.kv_transfer_params = {}
    return request


def make_manager(groups=1, caching=True):
    return KVCacheManager(
        KVCacheConfig(
            num_blocks=64,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    layer_names=[f"layer.{i}"],
                    kv_cache_spec=FullAttentionSpec(
                        block_size=2, num_kv_heads=1, head_size=i + 1, dtype=torch.bfloat16
                    ),
                )
                for i in range(groups)
            ],
        ),
        max_model_len=32,
        scheduler_block_size=2,
        hash_block_size=2,
        enable_caching=caching,
    )


def setup_owner(*, groups=1, caching=True):
    manager = make_manager(groups, caching)
    connector = delayed_scheduler()
    install_mooncake_decode_cache_guard(SimpleNamespace(connector=connector, kv_cache_manager=manager))
    request = make_request()
    manager.allocate_slots(request, 6)
    connector.update_state_after_alloc(request, manager.get_blocks(request.request_id), 0)
    connector.build_connector_meta(None)
    return manager, connector, request


def report(connector, request, prefix, ranks=(0, 1), *, release=False, generation=None):
    if generation is None:
        generation = connector._decode_ledger.generation(request.request_id)
    message = output(
        decode=tuple(
            DsaDecodeResult(
                request.request_id,
                generation,
                rank,
                DsaDecodeResultKind.RELEASE_COMPLETE if release else DsaDecodeResultKind.PREFIX_READY,
                prefix,
            )
            for rank in ranks
        )
    )
    connector.update_connector_output(message)
    return message


@pytest.mark.parametrize("groups", [1, 2])
def test_unwritten_host_block_is_invisible_to_second_request_until_every_tp_completes(groups):
    manager, connector, first = setup_owner(groups=groups)
    second = make_request("second")
    blocks = manager.get_blocks(first.request_id).blocks
    host = torch.full((groups, 64, 2), -1)
    expected = torch.arange(6).reshape(3, 2)
    assert manager.get_computed_blocks(second)[1] == 0
    first.num_computed_tokens = 6
    # allocate_slots publishes through coordinator directly, before compute.
    manager.allocate_slots(first, 1)
    assert manager.get_computed_blocks(second)[1] == 0
    for group, group_blocks in enumerate(blocks):
        host[group, group_blocks[0].block_id] = expected[0]
    report(connector, first, 2, ranks=(0,))
    manager.cache_blocks(first, 6)
    assert manager.get_computed_blocks(second)[1] == 0
    report(connector, first, 2, ranks=(1,))
    manager.cache_blocks(first, 6)
    hits, length, _ = manager.get_computed_blocks(second)
    assert length == 2
    for group, hit in enumerate(hits.blocks):
        assert len(hit) == 1
        assert torch.equal(host[group, hit[0].block_id], expected[0])
    # The next allocation must not hash the uncompleted generated blocks.
    manager.allocate_slots(first, 2)
    assert manager.get_computed_blocks(second)[1] == 2
    for group, group_blocks in enumerate(blocks):
        host[group, group_blocks[1].block_id] = expected[1]
        host[group, group_blocks[2].block_id] = expected[2]
    report(connector, first, 6)
    manager.cache_blocks(first, 6)
    hits, length, _ = manager.get_computed_blocks(second)
    assert length == 6
    for group, hit in enumerate(hits.blocks):
        assert torch.equal(torch.stack([host[group, block.block_id] for block in hit]), expected)


@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.parametrize("caching", [False, True])
def test_free_and_deferred_pop_preserve_ownership_until_all_release_acks(deferred, caching):
    manager, connector, request = setup_owner(caching=caching)
    before = manager.get_block_ids(request.request_id)
    free_before = manager.block_pool.get_num_free_blocks()
    free = manager.pop_blocks_for_free if deferred else manager.free
    connector.request_finished(request, before)
    connector.build_connector_meta(None)
    report(connector, request, 0, ranks=(0,), release=True)
    with pytest.raises(RuntimeError, match="still owns blocks"):
        free(request)
    assert manager.get_block_ids(request.request_id) == before
    assert manager.block_pool.get_num_free_blocks() == free_before
    final = report(connector, request, 0, ranks=(1,), release=True)
    assert final.finished_sending == {request.request_id}
    released = free(request)
    if deferred:
        assert manager.block_pool.get_num_free_blocks() == free_before
        manager.block_pool.free_blocks(reversed(released))
    assert manager.block_pool.get_num_free_blocks() == free_before + 3


def test_cancel_owner_keeps_shared_cached_content_alive_and_late_generation_cannot_publish():
    manager, connector, first = setup_owner()
    first.num_computed_tokens = 6
    report(connector, first, 2)
    manager.cache_blocks(first, 6)
    second = make_request("second")
    hits, length, _ = manager.get_computed_blocks(second)
    manager.allocate_slots(second, 1, num_new_computed_tokens=length, new_computed_blocks=hits)
    shared_block = hits.blocks[0][0]
    assert shared_block.ref_cnt == 2
    connector.request_finished(first, manager.get_block_ids(first.request_id))
    connector.build_connector_meta(None)
    report(connector, first, 2, release=True)
    manager.free(first)
    assert shared_block.ref_cnt == 1
    assert manager.get_computed_blocks(make_request("third"))[1] == 2
    # The same textual ID can start a new generation; old reports must not
    # publish newly allocated blocks whose Host content is still unwritten.
    replacement = make_request(first.request_id)
    manager.allocate_slots(replacement, 6)
    connector.update_state_after_alloc(replacement, manager.get_blocks(replacement.request_id), 0)
    connector.build_connector_meta(None)
    report(connector, replacement, 6, generation=1)
    manager.cache_blocks(replacement, 6)
    assert manager.get_computed_blocks(make_request("fourth"))[1] == 2
    assert shared_block.ref_cnt == 1


@pytest.mark.parametrize("scheduler_cls", [Scheduler, AsyncScheduler, RecomputeScheduler, AsyncRecomputeScheduler])
def test_constructor_installs_guard_for_actual_scheduler_classes(monkeypatch, scheduler_cls):
    manager = make_manager()
    connector = delayed_scheduler()
    connector.requires_kv_delivery = False
    config = SimpleNamespace(kv_transfer_config=SimpleNamespace(is_kv_producer=False))

    def initialize_base(self, *args, **kwargs):
        self.connector = connector
        self.kv_cache_manager = manager
        self.vllm_config = config
        self.num_spec_tokens = 0
        self.parallel_config = SimpleNamespace(pipeline_parallel_size=1)

    # Replace only upstream hardware/config setup; execute the real Ascend
    # constructor chain, including the production guard installation hook.
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
    scheduler_cls(config, None, None, 2)
    request = make_request()
    manager.allocate_slots(request, 6)
    assert manager.get_computed_blocks(make_request("second"))[1] == 0


@pytest.mark.parametrize("scheduler_cls", [RecomputeScheduler, AsyncRecomputeScheduler])
def test_actual_recompute_receive_publication_is_fenced(scheduler_cls):
    manager, connector, request = setup_owner()
    request.num_computed_tokens = 6
    scheduler = scheduler_cls.__new__(scheduler_cls)
    scheduler.kv_cache_manager = manager
    scheduler.connector = connector
    scheduler.failed_recving_kv_req_ids = set()
    scheduler.finished_recving_kv_req_ids = {request.request_id}
    report(connector, request, 2)
    scheduler._update_waiting_for_remote_kv(request)
    assert manager.get_computed_blocks(make_request("second"))[1] == 2


@pytest.mark.parametrize("deferred", [False, True])
def test_actual_preempt_is_rejected_before_block_or_request_mutation(deferred):
    manager, connector, request = setup_owner()
    request.status = RequestStatus.RUNNING
    request.num_computed_tokens = 6
    request.last_sched_seq = 2
    scheduler = AsyncRecomputeScheduler.__new__(AsyncRecomputeScheduler)
    scheduler.kv_cache_manager = manager
    scheduler.defer_block_free = deferred
    scheduler.processed_step_seq = 1
    before = manager.get_block_ids(request.request_id)
    with pytest.raises(RuntimeError, match="Preemption/resume is not supported"):
        scheduler._preempt_request(request, 0)
    assert manager.get_block_ids(request.request_id) == before
    assert request.status == RequestStatus.RUNNING
    assert request.num_computed_tokens == 6


def test_disabled_connector_keeps_original_publication_and_free():
    manager = make_manager()
    original = manager.coordinator.cache_blocks
    install_mooncake_decode_cache_guard(SimpleNamespace(connector=None, kv_cache_manager=manager))
    assert manager.coordinator.cache_blocks == original
    request = make_request()
    manager.allocate_slots(request, 6)
    assert manager.get_computed_blocks(make_request("second"))[1] == 6
    manager.free(request)


def test_install_is_idempotent_and_missing_completion_fails_closed():
    manager = make_manager()
    connector = SimpleNamespace(delayed_d2h_enabled=True, get_completed_host_prefix=lambda _: None)
    scheduler = SimpleNamespace(connector=connector, kv_cache_manager=manager)
    install_mooncake_decode_cache_guard(scheduler)
    original = manager.coordinator.cache_blocks
    install_mooncake_decode_cache_guard(scheduler)
    assert manager.coordinator.cache_blocks == original
    with pytest.raises(RuntimeError, match="completed Host prefix"):
        manager.allocate_slots(make_request(), 6)


def test_pressure_fallback_abort_holds_blocks_through_normal_connector_lifecycle():
    manager, connector, request = setup_owner()
    request.status = RequestStatus.RUNNING
    scheduler = RecomputeScheduler.__new__(RecomputeScheduler)
    scheduler.requests = {request.request_id: request}
    scheduler.running = []
    scheduler.waiting = MagicMock()
    scheduler.skipped_waiting = MagicMock()
    scheduler._inflight_prefills = {request}
    scheduler._connector_finished = lambda req: connector.request_finished(req, manager.get_block_ids(req.request_id))
    scheduler.encoder_cache_manager = MagicMock()
    scheduler.ec_connector = None
    scheduler.finished_req_ids = set()
    scheduler.finished_req_ids_dict = None
    scheduler.kv_cache_manager = manager
    scheduler.defer_block_free = False
    scheduler._finish_recomputed_request(request, [])
    assert request.status == RequestStatus.FINISHED_ABORTED
    assert request.request_id in scheduler.requests
    assert manager.get_block_ids(request.request_id)[0]
    connector.build_connector_meta(None)
    report(connector, request, 0, release=True)
    scheduler._free_blocks(request)
    assert request.request_id not in scheduler.requests
