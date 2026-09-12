# SPDX-License-Identifier: Apache-2.0
"""Hardware integration test for a TP-shared Mooncake Host segment."""

from __future__ import annotations

import ctypes
import gc
import multiprocessing as mp
import os
import tempfile
import traceback
from datetime import timedelta
from multiprocessing.connection import Connection
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    AscendSFAIndexerCacheSpec,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_dsa_transfer import (
    MAX_REGISTER_MEMORY_BYTES,
    DsaRegisterAtom,
    collect_bounded_register_regions,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (
    SparseKVOffloadMemoryBudget,
    plan_sparse_kv_offload_memory,
)
from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import (
    GlobalTE,
)

_RUN_ENV = "VLLM_ASCEND_RUN_LARGE_MOONCAKE_INTEGRATION"
_GIB = 1024**3
_SHARED_SEGMENT_BYTES = 100 * _GIB
_TP_SIZE = 2
_OWNER_RANK = 0
_GLM_LAYER_COUNT = 79
_GLM_MAX_MODEL_LEN = 1_000_000
_GLM_MAX_NUM_SEQS = 8
_GLM_DCP_SIZE = 1
_GLM_BLOCK_SIZE = 128
_GLM_K_WIDTH = 512
_GLM_V_WIDTH = 64
_GLM_INDEXER_WIDTH = 128
_BFLOAT16_BYTES = 2
_LAYER_COMPONENT_ALIGNMENT_BYTES = 2 * 1024**2
_COPY_BYTES = 4096
_REGION_PATTERN_SEED = 0x5A
_GPU_MEMORY_UTILIZATION = 0.9
_NPU_KV_BUDGET_GIB_ENV = "VLLM_ASCEND_MOONCAKE_TEST_NPU_KV_GIB"
_DCP_SIZE_ENV = "VLLM_ASCEND_MOONCAKE_TEST_DCP_SIZE"
_COLLECTIVE_TIMEOUT_SECONDS = 90
_PEER_SETUP_TIMEOUT_SECONDS = 120
_PEER_EXIT_TIMEOUT_SECONDS = 15

pytestmark = pytest.mark.skipif(
    os.getenv(_RUN_ENV) != "1",
    reason=f"set {_RUN_ENV}=1 to run the 100 GiB Mooncake integration test",
)


def _initialize_engine(host: str) -> Any:
    from mooncake.engine import TransferEngine

    engine = TransferEngine()
    result = engine.initialize(host, "P2PHANDSHAKE", "ascend", "")
    if result != 0:
        raise RuntimeError(f"TransferEngine.initialize returned {result}")
    return engine


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _glm_layer_bytes(num_blocks: int) -> int:
    tokens = num_blocks * _GLM_BLOCK_SIZE
    k_bytes = tokens * _GLM_K_WIDTH * _BFLOAT16_BYTES
    v_bytes = tokens * _GLM_V_WIDTH * _BFLOAT16_BYTES
    return _align_up(
        k_bytes,
        _LAYER_COMPONENT_ALIGNMENT_BYTES,
    ) + _align_up(
        v_bytes,
        _LAYER_COMPONENT_ALIGNMENT_BYTES,
    )


def _build_planner_specs() -> dict[str, Any]:
    import torch

    main_spec = AscendMLAAttentionSpec(
        block_size=_GLM_BLOCK_SIZE,
        num_kv_heads=1,
        head_size=_GLM_K_WIDTH + _GLM_V_WIDTH,
        dtype=torch.bfloat16,
        cache_dtype_str="auto",
        store_on_host=True,
    )
    indexer_spec = AscendSFAIndexerCacheSpec(
        block_size=_GLM_BLOCK_SIZE,
        num_kv_heads=1,
        head_size=_GLM_INDEXER_WIDTH,
        dtype=torch.bfloat16,
        cache_dtype_str="auto",
    )
    return {
        **{
            f"main.{layer_index}": main_spec
            for layer_index in range(_GLM_LAYER_COUNT)
        },
        **{
            f"indexer.{layer_index}": indexer_spec
            for layer_index in range(_GLM_LAYER_COUNT)
        },
    }


def _standalone_npu_kv_budget_bytes(device_ids: tuple[int, ...]) -> tuple[int, str]:
    """Return a no-model upper bound, or an injected profile-derived budget."""
    import torch
    import torch_npu  # noqa: F401

    configured_gib = os.getenv(_NPU_KV_BUDGET_GIB_ENV)
    if configured_gib is not None:
        configured_bytes = int(float(configured_gib) * _GIB)
        if configured_bytes <= 0:
            raise ValueError(f"{_NPU_KV_BUDGET_GIB_ENV} must be positive")
        return configured_bytes, _NPU_KV_BUDGET_GIB_ENV

    per_device_budgets = []
    for device_id in device_ids:
        torch.npu.set_device(device_id)
        free_bytes, total_bytes = torch.npu.mem_get_info()
        requested_bytes = int(total_bytes * _GPU_MEMORY_UTILIZATION)
        per_device_budgets.append(min(free_bytes, requested_bytes))
    return min(per_device_budgets), "min(free_hbm, total_hbm * 0.9)"


def _plan_sparse_cache(
    available_device_memory_bytes: int,
    dcp_size: int,
) -> SparseKVOffloadMemoryBudget:
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=_GLM_MAX_MODEL_LEN),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=dcp_size,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=_GLM_MAX_NUM_SEQS),
    )
    return plan_sparse_kv_offload_memory(
        kv_cache_spec=_build_planner_specs(),
        vllm_config=vllm_config,
        available_device_memory_bytes=available_device_memory_bytes,
        dram_limit_bytes=_SHARED_SEGMENT_BYTES,
        keep_device_kv_cache=False,
    )


def _glm_layer_atoms(
    base_ptr: int,
    location: str,
    layer_bytes: int,
) -> list[DsaRegisterAtom]:
    allocation = ("shared-host-pool", base_ptr)
    return [
        DsaRegisterAtom(
            start=base_ptr + layer_index * layer_bytes,
            end=base_ptr + (layer_index + 1) * layer_bytes,
            location=location,
            allocation=allocation,
        )
        for layer_index in range(_GLM_LAYER_COUNT)
    ]


def _region_layer_counts(layer_bytes: int) -> list[int]:
    layers_per_region = (MAX_REGISTER_MEMORY_BYTES - 1) // layer_bytes
    return [
        min(layers_per_region, _GLM_LAYER_COUNT - first_layer)
        for first_layer in range(0, _GLM_LAYER_COUNT, layers_per_region)
    ]


def _expected_region_lengths(layer_bytes: int) -> list[int]:
    return [
        layer_count * layer_bytes
        for layer_count in _region_layer_counts(layer_bytes)
    ]


def _region_offsets(region_lengths: list[int]) -> list[int]:
    offsets = []
    current_offset = 0
    for region_length in region_lengths:
        offsets.append(current_offset)
        current_offset += region_length
    return offsets


def _region_pattern_byte(region_index: int) -> int:
    return (_REGION_PATTERN_SEED + region_index) % 256


def _initialize_process_group(rank: int, rendezvous_path: str) -> Any:
    import torch.distributed as dist

    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=_TP_SIZE,
        timeout=timedelta(seconds=_COLLECTIVE_TIMEOUT_SECONDS),
    )
    return dist.group.WORLD


def _create_and_register_rank(
    *,
    rank: int,
    device_id: int,
    host: str,
    segment_name: str,
    rendezvous_path: str,
    num_blocks: int,
) -> tuple[GlobalTE, Any, Any, list[int]]:
    import torch
    import torch_npu  # noqa: F401
    from mooncake.shared_segment import create_shared_segment

    torch.npu.set_device(device_id)
    comm_group = _initialize_process_group(rank, rendezvous_path)
    segment = create_shared_segment(
        segment_name,
        blocks={
            "pool": {
                "count": 1,
                "shape": (_SHARED_SEGMENT_BYTES,),
                "dtype": torch.uint8,
            }
        },
        world_size=_TP_SIZE,
        rank_id=rank,
        owner_rank=_OWNER_RANK,
        device_id=device_id,
        comm_group=comm_group,
        mmap=True,
        host_register=True,
    )
    raw = segment.tensors("pool")[0]
    base_ptr = int(raw.data_ptr())
    location = f"npu:{device_id}"
    layer_bytes = _glm_layer_bytes(num_blocks)
    used_bytes = _GLM_LAYER_COUNT * layer_bytes
    assert used_bytes <= _SHARED_SEGMENT_BYTES
    assert raw.numel() == _SHARED_SEGMENT_BYTES

    regions = collect_bounded_register_regions(
        _glm_layer_atoms(base_ptr, location, layer_bytes)
    )
    expected_lengths = _expected_region_lengths(layer_bytes)
    assert regions.ptrs == [
        base_ptr + offset
        for offset in _region_offsets(expected_lengths)
    ]
    assert regions.lengths == expected_lengths
    assert regions.locations == [location] * len(expected_lengths)
    assert all(size < MAX_REGISTER_MEMORY_BYTES for size in regions.lengths)

    manager = GlobalTE()
    manager.transfer_engine = _initialize_engine(host)
    manager.register_buffer(
        regions.ptrs,
        regions.lengths,
        regions.locations,
    )
    assert manager.is_register_buffer
    return manager, segment, raw, regions.lengths


def _release_rank_resources(
    manager: GlobalTE | None,
    segment: Any,
    raw: Any,
) -> None:
    import torch.distributed as dist

    if manager is not None and manager.is_register_buffer:
        manager.unregister_buffer()
    raw = None
    segment = None
    if manager is not None:
        manager.transfer_engine = None
    gc.collect()
    if dist.is_initialized():
        dist.destroy_process_group()


def _tp_rank_one_main(
    host: str,
    device_id: int,
    segment_name: str,
    rendezvous_path: str,
    num_blocks: int,
    connection: Connection,
) -> None:
    manager = None
    segment = None
    raw = None
    try:
        manager, segment, raw, region_lengths = _create_and_register_rank(
            rank=1,
            device_id=device_id,
            host=host,
            segment_name=segment_name,
            rendezvous_path=rendezvous_path,
            num_blocks=num_blocks,
        )
        host_base = int(segment.base_addr())
        for region_index, region_offset in enumerate(
            _region_offsets(region_lengths)
        ):
            ctypes.memset(
                host_base + region_offset,
                _region_pattern_byte(region_index),
                _COPY_BYTES,
            )
        connection.send(
            {
                "ok": True,
                "session": f"{host}:{manager.transfer_engine.get_rpc_port()}",
                "ptr": int(raw.data_ptr()),
                "region_lengths": region_lengths,
            }
        )
        connection.recv()
    except BaseException:  # noqa: BLE001
        connection.send({"ok": False, "error": traceback.format_exc()})
    finally:
        _release_rank_resources(manager, segment, raw)
        connection.close()


def test_glm_100_gib_shared_segment_is_mapped_and_registered_by_both_tp_ranks() -> None:
    import torch
    import torch_npu  # noqa: F401
    from mooncake.shared_segment import shared_segment_supported
    from vllm.utils.network_utils import get_ip

    owner_device_id = int(os.getenv("VLLM_ASCEND_MOONCAKE_TEST_DEVICE", "0"))
    peer_device_id = int(os.getenv("VLLM_ASCEND_MOONCAKE_TEST_PEER_DEVICE", "1"))
    if owner_device_id == peer_device_id:
        pytest.fail("Mooncake TP test requires two different NPU devices")
    if not shared_segment_supported(mmap=True, host_register=True):
        pytest.skip("Mooncake mmap + HostRegister shared segments are unavailable")

    os.environ.setdefault("HCCL_INTRA_ROCE_ENABLE", "1")
    npu_budget_bytes, npu_budget_source = _standalone_npu_kv_budget_bytes(
        (owner_device_id, peer_device_id)
    )
    dcp_size = int(os.getenv(_DCP_SIZE_ENV, str(_GLM_DCP_SIZE)))
    if dcp_size <= 0:
        pytest.fail(f"{_DCP_SIZE_ENV} must be positive")
    budget = _plan_sparse_cache(npu_budget_bytes, dcp_size)
    assert budget.final_num_blocks > 0
    assert (
        budget.planned_host_bytes + budget.host_alignment_reserve_bytes
        <= _SHARED_SEGMENT_BYTES
    )
    layer_bytes = _glm_layer_bytes(budget.final_num_blocks)
    region_layer_counts = _region_layer_counts(layer_bytes)
    assert len(region_layer_counts) > 1, (
        "planner result does not exercise split registration: "
        f"num_blocks={budget.final_num_blocks}, layer_bytes={layer_bytes}"
    )
    print(
        "Mooncake sparse planner: "
        f"max_model_len={_GLM_MAX_MODEL_LEN}, "
        f"max_num_seqs={_GLM_MAX_NUM_SEQS}, "
        f"dcp_size={dcp_size}, "
        f"dram_gib={_SHARED_SEGMENT_BYTES / _GIB:.2f}, "
        f"npu_kv_budget_gib={npu_budget_bytes / _GIB:.2f}, "
        f"npu_budget_source={npu_budget_source}, "
        f"limits=(npu={budget.npu_limit_blocks}, "
        f"dram={budget.dram_limit_blocks}, "
        f"workload={budget.workload_limit_blocks}), "
        f"final_num_blocks={budget.final_num_blocks}, "
        f"limiting_factor={budget.limiting_factor}, "
        f"layer_bytes={layer_bytes}, "
        f"region_layers={region_layer_counts}"
    )

    torch.npu.set_device(owner_device_id)
    host = os.getenv("VLLM_ASCEND_MOONCAKE_TEST_HOST", get_ip())
    unique_id = os.getpid()
    segment_name = f"vllm_glm_100g_tp_integration_{unique_id}"
    rendezvous_path = str(
        Path(tempfile.gettempdir()) / f"{segment_name}.rendezvous"
    )
    context = mp.get_context("spawn")
    parent_connection, child_connection = context.Pipe()
    peer = context.Process(
        target=_tp_rank_one_main,
        args=(
            host,
            peer_device_id,
            segment_name,
            rendezvous_path,
            budget.final_num_blocks,
            child_connection,
        ),
    )
    manager = None
    segment = None
    raw = None

    try:
        peer.start()
        manager, segment, raw, region_lengths = _create_and_register_rank(
            rank=0,
            device_id=owner_device_id,
            host=host,
            segment_name=segment_name,
            rendezvous_path=rendezvous_path,
            num_blocks=budget.final_num_blocks,
        )
        if not parent_connection.poll(_PEER_SETUP_TIMEOUT_SECONDS):
            pytest.fail("timed out waiting for TP rank 1")
        peer_info = parent_connection.recv()
        if not peer_info["ok"]:
            pytest.fail(f"TP rank 1 setup failed:\n{peer_info['error']}")
        assert peer_info["region_lengths"] == region_lengths

        host_base = int(segment.base_addr())
        local_device_base = int(raw.data_ptr())
        remote_device_base = int(peer_info["ptr"])
        for region_index, region_offset in enumerate(
            _region_offsets(region_lengths)
        ):
            pattern_byte = _region_pattern_byte(region_index)
            expected = bytes([pattern_byte]) * _COPY_BYTES
            source = (ctypes.c_ubyte * _COPY_BYTES).from_address(
                host_base + region_offset
            )
            assert bytes(source) == expected

            destination_offset = region_offset + _COPY_BYTES
            ctypes.memset(host_base + destination_offset, 0, _COPY_BYTES)
            result = manager.transfer_engine.transfer_sync_read(
                peer_info["session"],
                local_device_base + destination_offset,
                remote_device_base + region_offset,
                _COPY_BYTES,
            )
            assert result == 0
            destination = (ctypes.c_ubyte * _COPY_BYTES).from_address(
                host_base + destination_offset
            )
            assert bytes(destination) == expected
    finally:
        try:
            parent_connection.send("stop")
        except (BrokenPipeError, EOFError):
            pass
        _release_rank_resources(manager, segment, raw)
        peer.join(timeout=_PEER_EXIT_TIMEOUT_SECONDS)
        if peer.is_alive():
            peer.terminate()
            peer.join(timeout=_PEER_EXIT_TIMEOUT_SECONDS)
        parent_connection.close()
        child_connection.close()
        Path(rendezvous_path).unlink(missing_ok=True)
