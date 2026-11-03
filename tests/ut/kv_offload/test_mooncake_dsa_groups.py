# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import msgspec
import pytest
from vllm.v1.request import RequestStatus

from tests.ut.kv_offload.test_mooncake_dsa_scheduler import _request, _scheduler
from tests.ut.kv_offload.test_sfa_mixed_cache_layout import fixture_layout
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import (
    MooncakeAgentMetadata,
    MooncakeConnectorScheduler,
    _MooncakeDsaDecodeScheduler,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_dsa_groups import find_dsa_host_group
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_dsa_metadata import DsaCacheGroupBlocks, DsaStepRequest

MODULE = "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector"


def grouped_scheduler():
    scheduler = _scheduler()
    scheduler._dsa_shared_residency = True
    scheduler.block_size = [128, 128, 128]
    scheduler.main_group_idx, scheduler.indexer_group_idx = 2, 0
    scheduler._main_block_size = 128
    scheduler._dsa_prefill_tp_size = 8
    return scheduler


def grouped_request():
    request = _request()
    request.prompt_token_ids = list(range(256))
    request.kv_transfer_params.update(
        dsa_protocol_version=2,
        dsa_group_block_sizes=[128, 128, 128],
        dsa_main_group_id=0,
        dsa_indexer_group_id=2,
        remote_block_ids=((40, 41), (50, 51), (60, 61)),
        remote_ptp_size=8,
        remote_dcp_size=8,
        remote_block_size=128,
    )
    return request


def groups_for_allocation(*, prefix=0):
    groups = ((10, 11), (20, 21), (30, 31))
    return SimpleNamespace(
        get_block_ids=lambda: groups,
        get_unhashed_block_ids_all_groups=lambda: tuple(group[prefix // 128 :] for group in groups),
    )


@pytest.mark.parametrize("uniform", [False, True])
def test_constructor_selects_host_group_instead_of_first_main_group(uniform):
    _, _, cache_config = fixture_layout(uniform=uniform)
    cfg = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            engine_id="decode",
            kv_port=9000,
            get_from_extra_config=lambda key, default: {"tp_size": 8} if key == "prefill" else default,
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=2, data_parallel_rank=0, data_parallel_size=1),
    )
    with (
        patch(
            MODULE + ".get_ascend_config",
            return_value=SimpleNamespace(
                sparse_kv_offload_config=SimpleNamespace(offload_shared_indexer_only=True, delayed_d2h=True)
            ),
        ),
        patch("vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.scheduler.get_ip", return_value="127.0.0.1"),
    ):
        scheduler = _MooncakeDsaDecodeScheduler(cfg, cache_config)
    try:
        assert scheduler.main_group_idx == (0 if uniform else 2)
        assert scheduler._main_block_size == 128
        assert scheduler.delayed_d2h_enabled
    finally:
        scheduler.executor.shutdown(wait=True)


@pytest.mark.parametrize("prefix", [0, 128, 256])
def test_scheduler_carries_distinct_source_and_destination_groups_and_real_failure_ids(prefix):
    scheduler, request = grouped_scheduler(), grouped_request()
    external, receiving = scheduler.get_num_new_matched_tokens(request, prefix)
    assert external == 256 - prefix and receiving == (prefix < 256)
    scheduler.update_state_after_alloc(request, groups_for_allocation(prefix=prefix), external)
    (command,) = scheduler.build_connector_meta(None).requests
    assert command.source.protocol_version == 2
    assert [(group.group_id, group.block_ids) for group in command.source.cache_groups] == [
        (0, (40, 41)),
        (1, (50, 51)),
        (2, (60, 61)),
    ]
    assert [group.block_ids for group in command.destination_groups] == [(10, 11), (20, 21), (30, 31)]
    assert command.source.remote_main_group_id == 0
    assert command.main_host_block_ids == (30, 31)
    assert command.indexer_hbm_block_ids == (10, 11)
    assert command.invalid_block_ids == (
        (10, 11, 20, 21, 30, 31) if prefix == 0 else (11, 21, 31) if prefix == 128 else ()
    )
    assert command.notify_only == (prefix == 256)
    assert msgspec.msgpack.decode(msgspec.msgpack.encode(command), type=DsaStepRequest) == command


@pytest.mark.parametrize("version", [None, 1, 3, True, "2"])
def test_shared_only_rejects_legacy_or_unknown_protocol_before_creating_receive(version):
    scheduler, request = grouped_scheduler(), grouped_request()
    request.kv_transfer_params["dsa_protocol_version"] = version
    with pytest.raises(ValueError, match="upgrade all Prefill"):
        scheduler.get_num_new_matched_tokens(request, 0)
    assert not scheduler._dsa_requests
    assert request.kv_transfer_params["do_remote_prefill"]


@pytest.mark.parametrize("sizes", [None, [], [128, 128], [128, 0, 128], [128, True, 128]])
def test_grouped_source_requires_valid_block_size_for_every_group(sizes):
    scheduler, request = grouped_scheduler(), grouped_request()
    request.kv_transfer_params["dsa_group_block_sizes"] = sizes
    with pytest.raises((TypeError, ValueError)):
        scheduler.get_num_new_matched_tokens(request, 0)
    assert not scheduler._dsa_requests


def test_every_destination_group_requires_allocated_capacity():
    scheduler, request = grouped_scheduler(), grouped_request()
    scheduler.get_num_new_matched_tokens(request, 0)
    blocks = groups_for_allocation()
    blocks.get_block_ids = lambda: ((10, 11), (20,), (30, 31))
    with pytest.raises(ValueError, match="every DSA destination"):
        scheduler.update_state_after_alloc(request, blocks, 256)
    assert scheduler.build_connector_meta(None).requests == ()


@pytest.mark.parametrize("alias", ["dsa_main_block_size", "dsa_indexer_block_size"])
@pytest.mark.parametrize("size", [64, True, "128"])
def test_grouped_source_rejects_conflicting_legacy_size_alias(alias, size):
    scheduler, request = grouped_scheduler(), grouped_request()
    request.kv_transfer_params[alias] = size
    with pytest.raises(ValueError, match="conflicts with source group"):
        scheduler.get_num_new_matched_tokens(request, 0)
    assert not scheduler._dsa_requests


def test_cancel_before_allocation_is_grouped_notification_without_destinations():
    scheduler, request = grouped_scheduler(), grouped_request()
    scheduler.request_finished(request, ())
    (command,) = scheduler.build_connector_meta(None).requests
    assert command.notify_only and command.source.protocol_version == 2
    assert command.destination_groups == ()
    assert command.main_host_block_ids == command.indexer_hbm_block_ids == ()


def test_group_schema_rejects_conflicting_aliases_and_noncontiguous_group_identity():
    scheduler, request = grouped_scheduler(), grouped_request()
    scheduler.get_num_new_matched_tokens(request, 0)
    source = scheduler._dsa_requests[request.request_id].source
    for changes in (
        {"main_block_ids": (99,)},
        {"remote_main_group_id": 3},
        {"remote_indexer_block_size": 64},
        {"protocol_version": 1},
        {"cache_groups": ()},
        {"cache_groups": (DsaCacheGroupBlocks(1, 128, (1,)),)},
    ):
        with pytest.raises(ValueError):
            replace(source, **changes)
    with pytest.raises(ValueError, match="all destination"):
        DsaStepRequest("request", source, (), ())
    with pytest.raises(ValueError, match="exactly once"):
        DsaStepRequest("request", source, (), (), destination_groups=(DsaCacheGroupBlocks(0, 128, (1,)),) * 2)


def test_no_host_group_is_rejected_at_scheduler_configuration():
    _, _, config = fixture_layout()
    config.kv_cache_groups.pop()
    with pytest.raises(ValueError, match="exactly one Host"):
        find_dsa_host_group(config)


def test_prefill_advertises_versioned_all_group_geometry_without_changing_block_ids():
    _, _, config = fixture_layout()
    scheduler = MooncakeConnectorScheduler.__new__(MooncakeConnectorScheduler)
    scheduler.kv_cache_groups = config.kv_cache_groups
    scheduler.block_size = 128
    scheduler.tp_size, scheduler.pcp_size, scheduler.dcp_size = 8, 1, 8
    scheduler.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
        kv_transfer_config=SimpleNamespace(get_from_extra_config=lambda key, default: key == "dsa_pd_offload"),
    )
    scheduler._reqs_need_send = {}
    scheduler._get_transfer_block_ids = lambda blocks, prompt: blocks
    scheduler._get_swa_transfer_block_ids = lambda blocks: blocks
    scheduler.engine_id = "prefill"
    scheduler.side_channel_host, scheduler.side_channel_port = "127.0.0.1", 9000
    scheduler.multi_nodes_meta_mapping = {}
    request = SimpleNamespace(
        request_id="source",
        prompt_token_ids=list(range(256)),
        output_token_ids=[42],
        kv_transfer_params={"do_remote_decode": True},
        status=RequestStatus.FINISHED_LENGTH_CAPPED,
    )
    groups = ((10, 11), (20, 21), (30, 31))
    held, params = scheduler.request_finished(request, groups)
    assert held and params["remote_block_ids"] == groups
    assert params["dsa_protocol_version"] == 2
    assert params["dsa_group_block_sizes"] == [128, 128, 128]
    assert params["dsa_indexer_group_id"] == 0 and params["dsa_main_group_id"] == 1
    assert params["remote_dcp_size"] == 8


def test_agent_metadata_version_defaults_to_legacy_and_roundtrips_v2():
    metadata = MooncakeAgentMetadata("peer", 9000, {}, 128, [], [], 10, [], [])
    decoder = msgspec.msgpack.Decoder(MooncakeAgentMetadata)
    assert decoder.decode(msgspec.msgpack.encode(metadata)).dsa_protocol_version == 1
    metadata.dsa_protocol_version = 2
    assert decoder.decode(msgspec.msgpack.encode(metadata)).dsa_protocol_version == 2
