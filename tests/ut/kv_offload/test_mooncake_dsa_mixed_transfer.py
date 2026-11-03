# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
import threading
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from tests.ut.kv_offload.test_mooncake_dsa_shared_pool import make_worker, submit
from tests.ut.kv_offload.test_sfa_mixed_cache_layout import DRAFT, INDEXER, MEMBER, OWNER, fixture_layout
from vllm_ascend.distributed.kv_transfer.kv_p2p import mooncake_connector
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import MooncakeConnectorWorker
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_dsa_metadata import (
    DsaCacheGroupBlocks,
    DsaLocalResultKind,
    DsaTransferPhase,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_dsa_transfer import DsaCacheLayout

BLOCK = 4
CAPACITY = 128
POISON = -99
NAMES = (OWNER, MEMBER, INDEXER)


def tensors_for(name, pages):
    if name == INDEXER:
        return (
            torch.full((pages, BLOCK, 2), POISON, dtype=torch.int8),
            torch.full((pages, BLOCK, 1), POISON, dtype=torch.float16),
        )
    return tuple(torch.full((pages, BLOCK, 1), POISON, dtype=torch.bfloat16) for _ in range(2))


def fixture_transfer(dcp=8, prefix=0):
    ptp, dtp, length = 8, 2, 35
    source_ids = (tuple(range(19, 8, -1)), tuple(range(39, 28, -1)), tuple(range(59, 48, -1)))
    destination_ids = (tuple(range(70, 81)), tuple(range(90, 101)), tuple(range(110, 121)))
    # Transfer metadata groups deliberately differ from cache-manager groups,
    # and source manager groups differ from destination manager groups.
    groups = {
        7: ({"layer_names": [INDEXER], "kv_cache_group_id": 2, "kv_cache_spec": {"block_size": BLOCK}}, [2]),
        3: ({"layer_names": [OWNER], "kv_cache_group_id": 0, "kv_cache_spec": {"block_size": BLOCK}}, [0]),
        9: ({"layer_names": [MEMBER], "kv_cache_group_id": 1, "kv_cache_spec": {"block_size": BLOCK}}, [1]),
    }
    sources = []
    for rank in range(ptp):
        caches = {name: tensors_for(name, CAPACITY * (dcp if name == INDEXER else 1)) for name in NAMES}
        for gid, name in enumerate(NAMES):
            for token in range(length):
                ordinal, offset = divmod(token, BLOCK)
                if name != INDEXER and ordinal % dcp != rank % dcp:
                    continue
                block_id = source_ids[gid][ordinal // dcp]
                page = block_id * dcp + ordinal % dcp if name == INDEXER else block_id
                for position, tensor in enumerate(caches[name]):
                    tensor[page, offset] = token + 1 + position * 40
        sources.append(caches)
    host = tensors_for(MEMBER, CAPACITY)
    rank_states = []
    writes = Counter()
    for rank in range(dtp):
        worker, thread, command = make_worker(rank, ptp, dcp, dtp)
        caches = {OWNER: tensors_for(OWNER, CAPACITY), MEMBER: host, INDEXER: tensors_for(INDEXER, CAPACITY)}
        thread.pp_layer_indices = {0: (0, 10)}
        thread._dsa_transformer_layers = {OWNER: 5, MEMBER: 9, INDEXER: 5}
        thread.remote_dsa_protocol_versions = {"prefill": {5000 + p: 2 for p in range(ptp)}}
        for p, remote in enumerate(sources):
            port = 5000 + p
            thread.remote_kv_group2layeridx["prefill"][port] = groups
            for mapping, getter in (
                (thread.kv_caches_base_addr, lambda t: t.data_ptr()),
                (thread.remote_block_stride_per_addr, lambda t: t.stride(0) * t.element_size()),
                (thread.remote_block_len_per_addr, lambda t: t[0].numel() * t.element_size()),
                (thread.remote_block_size_scale, lambda t: t.shape[0] // CAPACITY),
                (thread.remote_cache_dtypes, lambda t: str(t.dtype)),
            ):
                mapping["prefill"][port] = [[getter(t) for t in remote[name]] for name in NAMES]
        layouts = {
            name: [
                DsaCacheLayout(
                    name,
                    pos,
                    t.data_ptr(),
                    t[0].numel() * t.element_size(),
                    t.stride(0) * t.element_size(),
                    1,
                    BLOCK,
                    str(t.dtype),
                    CAPACITY,
                    {INDEXER: 0, OWNER: 1, MEMBER: 2}[name],
                    name == MEMBER,
                )
                for pos, t in enumerate(caches[name])
            ]
            for name in NAMES
        }
        thread._dsa_main_local_layout = layouts[OWNER] + layouts[MEMBER]
        thread._dsa_indexer_local_layout = layouts[INDEXER]
        command = replace(
            command,
            source=replace(
                command.source,
                main_block_ids=source_ids[0],
                indexer_block_ids=source_ids[2],
                remote_main_group_id=0,
                remote_indexer_group_id=2,
                remote_main_block_size=BLOCK,
                remote_indexer_block_size=BLOCK,
                num_computed_tokens=prefix,
                num_external_tokens=length - prefix,
                protocol_version=2,
                cache_groups=tuple(DsaCacheGroupBlocks(i, BLOCK, ids) for i, ids in enumerate(source_ids)),
            ),
            main_host_block_ids=destination_ids[2],
            indexer_hbm_block_ids=destination_ids[0],
            destination_groups=tuple(DsaCacheGroupBlocks(i, BLOCK, ids) for i, ids in enumerate(destination_ids)),
        )

        def copy(session, dsts, srcs, sizes, caches=caches):
            remote = sources[int(session.split(":")[1]) - 7000]
            for dst, src, size in zip(dsts, srcs, sizes, strict=True):
                assert any(
                    t.data_ptr() <= dst < dst + size <= t.data_ptr() + t.nbytes for ts in caches.values() for t in ts
                )
                assert any(
                    t.data_ptr() <= src < src + size <= t.data_ptr() + t.nbytes for ts in remote.values() for t in ts
                )
                ctypes.memmove(dst, src, size)
                writes.update(range(dst, dst + size))
            return 0

        thread.engine.batch_transfer_sync_read = MagicMock(side_effect=copy)
        rank_states.append((worker, thread, command, caches))
    return rank_states, sources, destination_ids, writes, length


@pytest.mark.parametrize("dcp", [1, 8])
@pytest.mark.parametrize("prefix", [0, 5, 32])
def test_all_cp_shards_fill_every_private_owner_and_shared_host_exactly_once(dcp, prefix):
    states, sources, destination_ids, writes, length = fixture_transfer(dcp, prefix)
    for worker, thread, command, caches in states:
        for task in reversed(submit(worker, command)):
            thread._handle_dsa_request(task)
        (result,) = worker.build_connector_worker_meta().results
        assert result.kind is DsaLocalResultKind.RECEIVE_COMPLETE
    # Verify contents after both TP writers have drained, not merely address plans.
    for _, _, _, caches in states:
        for gid, name in enumerate((INDEXER, OWNER, MEMBER)):
            for position, tensor in enumerate(caches[name]):
                expected = torch.full_like(tensor, POISON)
                for token in range(prefix, length):
                    ordinal, offset = divmod(token, BLOCK)
                    expected[destination_ids[gid][ordinal], offset] = token + 1 + position * 40
                torch.testing.assert_close(tensor, expected)
    assert writes and set(writes.values()) == {1}
    assert sources  # Keep remote allocations alive until all transfers finish.


@pytest.mark.parametrize("fault", ["version", "group", "size", "local_size", "capacity", "missing"])
def test_bad_component_metadata_fails_before_endpoint_submits_any_copy(fault):
    states, sources, _, _, _ = fixture_transfer()
    worker, thread, command, _ = states[0]
    if fault == "version":
        thread.remote_dsa_protocol_versions["prefill"] = dict.fromkeys(thread.remote_te_port["prefill"], 1)
    elif fault in ("group", "size", "missing"):
        groups = thread.remote_kv_group2layeridx["prefill"][5000]
        if fault == "group":
            groups[3][0]["kv_cache_group_id"] = 42
        elif fault == "size":
            groups[3][0]["kv_cache_spec"]["block_size"] = 8
        else:
            del groups[3]
    elif fault == "local_size":
        thread._dsa_main_local_layout[0] = replace(thread._dsa_main_local_layout[0], block_tokens=8)
    else:
        thread._dsa_main_local_layout[0] = replace(thread._dsa_main_local_layout[0], capacity=1)
    for task in submit(worker, command):
        thread._handle_dsa_request(task)
    (result,) = worker.build_connector_worker_meta().results
    assert result.kind is DsaLocalResultKind.TRANSFER_FAILED
    thread.engine.batch_transfer_sync_read.assert_not_called()
    assert sources


def test_private_owner_transfer_failure_reports_its_phase_and_waits_for_all_tasks():
    states, sources, _, _, _ = fixture_transfer()
    worker, thread, command, _ = states[0]
    thread.engine.batch_transfer_sync_read.side_effect = [0, 0, -1]
    tasks = submit(worker, command)
    thread._handle_dsa_request(tasks[0])
    assert worker.build_connector_worker_meta() is None
    for task in tasks[1:]:
        thread._handle_dsa_request(task)
    (result,) = worker.build_connector_worker_meta().results
    assert result.kind is DsaLocalResultKind.TRANSFER_FAILED
    assert result.failure_phase is DsaTransferPhase.MAIN_D2D
    assert sources


@pytest.mark.parametrize("uniform", [False, True])
@pytest.mark.parametrize("c8", [False, True])
def test_real_specs_select_host_views_and_register_all_private_components(uniform, c8, monkeypatch):
    _, specs, config = fixture_layout(uniform=uniform, c8=c8)
    worker = MooncakeConnectorWorker.__new__(MooncakeConnectorWorker)
    worker._dsa_shared_residency = worker._dsa_decode = True
    worker.total_layers, worker.num_blocks = 10, 3
    worker.kv_cache_config = config
    worker._get_layer_spec = specs.__getitem__
    owner = tensors_for(OWNER, 3)
    indexer = tensors_for(INDEXER, 3) if c8 else (torch.empty((3, 4, 2), dtype=torch.bfloat16),)
    hosts = {MEMBER: tensors_for(MEMBER, 3), DRAFT: tensors_for(DRAFT, 3)}
    manager = SimpleNamespace(get_local_host_kv_views=MagicMock(side_effect=hosts.__getitem__))
    monkeypatch.setattr(mooncake_connector, "get_sparse_kv_offload_manager", lambda: manager)
    caches = {OWNER: owner, MEMBER: (None,) * 6, DRAFT: (None,) * 6, INDEXER: indexer}
    indexer_layout, main_layout = worker._build_dsa_local_layouts(caches)
    assert {call.args[0] for call in manager.get_local_host_kv_views.call_args_list} == {MEMBER, DRAFT}
    assert {layout.group_id for layout in main_layout if not layout.host_backed} == {0 if uniform else 1}
    private = indexer_layout + [layout for layout in main_layout if not layout.host_backed]
    regions, locations = worker._dsa_consumer_indexer_register_regions(caches, private)
    assert set(regions.ptrs) == {tensor.data_ptr() for tensors in (owner, indexer) for tensor in tensors}
    assert sum(regions.lengths) == sum(t.nbytes for ts in (owner, indexer) for t in ts)
    assert set(locations) == {"*"}
    assert {name for name in caches if worker._dsa_uses_private_cache(name)} == {OWNER, INDEXER}

    # Exercise the actual registration entry point and serialized GET_META,
    # including rank-private owner tensors and skipped Host placeholders.
    worker._dsa_pd_offload = True
    worker._is_hma_required = False
    worker.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            is_deepseek_mla=True,
            hf_text_config=SimpleNamespace(model_type="glm", index_topk=2),
            get_total_num_kv_heads=lambda: 1,
        ),
        speculative_config=None,
    )
    worker.engine_id, worker.engine = "decode", MagicMock()
    worker.tp_rank, worker.tp_size, worker._prefill_pp_size = 0, 2, 1
    worker.te_rpc_port, worker.handshake_port, worker.side_channel_port = 7000, 6000, 6000
    worker.block_size, worker.kv_role = 128, "kv_consumer"
    worker._prefill_pp_layer_partition = None
    ready = threading.Event()
    ready.set()
    monkeypatch.setattr(mooncake_connector.threading, "Event", lambda: ready)
    monkeypatch.setattr(mooncake_connector, "enable_sfa_dcp_replicated_indexer", lambda cfg: False)
    monkeypatch.setattr(mooncake_connector, "get_ip", lambda: "127.0.0.1")
    monkeypatch.setattr(mooncake_connector, "KVCacheRecvingThread", MagicMock())
    register = MagicMock()
    monkeypatch.setattr(mooncake_connector.global_te, "register_buffer", register)
    worker.register_kv_caches(caches)
    registered, sizes, _ = register.call_args.args
    assert set(registered) == set(regions.ptrs)
    assert sum(sizes) == sum(regions.lengths)
    metadata = worker.xfer_handshake_metadata
    assert metadata.dsa_protocol_version == 2
    for spec, indices in metadata.kv_group2layeridx.values():
        for name, index in zip(spec["layer_names"], indices, strict=True):
            expected = caches[name] if name in (OWNER, INDEXER) else ()
            assert metadata.kv_caches_base_addr[index] == [t.data_ptr() for t in expected]
            assert metadata.cache_dtypes[index] == [str(t.dtype) for t in expected]
    assert worker.kv_recv_thread._dsa_main_local_layout == main_layout
