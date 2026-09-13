# SPDX-License-Identifier: Apache-2.0
"""Exercise TP-shared Mooncake Host-memory registration on two A3 dies.

``create_shared_segment`` is collective for ``tp_size > 1``. The owner rank
enters allocation first, then the peer joins and maps the owner's physical
pages. After both mappings exist, the test initializes and registers the two
TransferEngines strictly in rank order.

Run on an A3 with logical dies 0, 1, and 2 visible::

    VLLM_ASCEND_RUN_LARGE_MOONCAKE_INTEGRATION=1 \
    ASCEND_RT_VISIBLE_DEVICES=0,1,2 pytest -sv \
        tests/e2e/nightly/single_node/test_mooncake_shared_segment_registration_two_dies.py

The six parameter cases cover die 0 + die 1 (same chip) and die 0 + die 2
(different chips), and run sequentially. Do not use pytest-xdist.
"""

from __future__ import annotations

import contextlib
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

GIB = 1024**3
MIB = 1024**2
SHARED_SEGMENT_ALIGNMENT = 2 * MIB
MAX_REGISTER_REGION_SIZE = 64 * GIB - 2 * MIB
LARGE_REGISTER_TOTAL_SIZE = 100 * GIB
TP_SIZE = 2
OWNER_RANK = 0
SHARED_MEMORY_PROBE_BYTES = 4096
PROCESS_GROUP_TIMEOUT = timedelta(minutes=5)
CHILD_EVENT_TIMEOUT_SECONDS = 300
CHILD_EXIT_TIMEOUT_SECONDS = 30
RUN_ENV = "VLLM_ASCEND_RUN_LARGE_MOONCAKE_INTEGRATION"

REGISTER_REGION_SIZES = (
    pytest.param((32 * GIB,), id="32GiB"),
    pytest.param((MAX_REGISTER_REGION_SIZE,), id="64GiB-minus-2MiB"),
    pytest.param(
        (
            MAX_REGISTER_REGION_SIZE,
            LARGE_REGISTER_TOTAL_SIZE - MAX_REGISTER_REGION_SIZE,
        ),
        id="100GiB-split",
    ),
)

SAME_CHIP_DEVICE_IDS = (0, 1)
CROSS_CHIP_DEVICE_IDS = (0, 2)

pytestmark = pytest.mark.skipif(
    os.getenv(RUN_ENV) != "1",
    reason=f"set {RUN_ENV}=1 to run the large two-die Mooncake integration test",
)


def _send_event(connection: Connection, event: str, **payload: Any) -> None:
    connection.send({"event": event, **payload})


def _receive_event(
    connection: Connection,
    expected_event: str,
) -> dict[str, Any]:
    if not connection.poll(CHILD_EVENT_TIMEOUT_SECONDS):
        pytest.fail(f"timed out waiting for child event {expected_event!r}")
    message = connection.recv()
    if message["event"] == "error":
        pytest.fail(f"child process failed:\n{message['error']}")
    assert message["event"] == expected_event, message
    return message


def _wait_for_command(connection: Connection, expected_command: str) -> None:
    command = connection.recv()
    if command != expected_command:
        raise RuntimeError(f"expected command {expected_command!r}, got {command!r}")


def _release_resources(
    transfer_engine: Any,
    regions: list[Any],
) -> None:
    import torch.distributed as dist

    try:
        if transfer_engine is not None and transfer_engine.is_register_buffer:
            transfer_engine.unregister_buffer()
    finally:
        while regions:
            regions.pop().release()
        gc.collect()
        if dist.is_initialized():
            dist.destroy_process_group()


def _rank_main(
    rank: int,
    device_id: int,
    host: str,
    segment_name: str,
    rendezvous_path: str,
    region_sizes: tuple[int, ...],
    connection: Connection,
) -> None:
    transfer_engine = None
    regions: list[Any] = []
    try:
        import torch
        import torch.distributed as dist
        import torch_npu  # noqa: F401

        from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.mooncake_host_pool import (
            HostPoolTopology,
            allocate_mooncake_host_region,
        )
        from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import (
            GlobalTE,
        )

        torch.npu.set_device(device_id)
        os.environ["HCCL_INTRA_ROCE_ENABLE"] = "1"
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{rendezvous_path}",
            rank=rank,
            world_size=TP_SIZE,
            timeout=PROCESS_GROUP_TIMEOUT,
        )
        # This mirrors the interface used by vLLM's TP GroupCoordinator. The
        # Mooncake shared-segment API performs its metadata collective through
        # tp_group.cpu_group.
        tp_group = SimpleNamespace(cpu_group=dist.group.WORLD)
        _send_event(connection, "process_group_ready")
        _wait_for_command(connection, "allocate")

        if rank == OWNER_RANK:
            _send_event(connection, "owner_allocation_started")

        total_size_bytes = sum(region_sizes)
        for region_index, size_bytes in enumerate(region_sizes):
            region = allocate_mooncake_host_region(
                size_bytes=size_bytes,
                alignment=SHARED_SEGMENT_ALIGNMENT,
                topology=HostPoolTopology(
                    tp_rank=rank,
                    tp_size=TP_SIZE,
                    owner_rank=OWNER_RANK,
                    device_id=device_id,
                    tp_group=tp_group,
                ),
                name=f"{segment_name}_{total_size_bytes}_{region_index}",
            )
            regions.append(region)
            assert region.tensor.device.type == "npu"
            assert region.tensor.is_contiguous()
            assert region.tensor.numel() * region.tensor.element_size() == size_bytes
            assert region.tensor.data_ptr() % SHARED_SEGMENT_ALIGNMENT == 0

        if rank == OWNER_RANK:
            for region_index, region in enumerate(regions):
                host_ptr = int(region.handle.base_addr()) + region.segment_offset
                ctypes.memset(
                    host_ptr,
                    0x5A + region_index,
                    SHARED_MEMORY_PROBE_BYTES,
                )

        _send_event(
            connection,
            "allocation_complete",
            offsets=[region.segment_offset for region in regions],
        )

        if rank != OWNER_RANK:
            _wait_for_command(connection, "verify_shared_memory")
            for region_index, region in enumerate(regions):
                host_ptr = int(region.handle.base_addr()) + region.segment_offset
                observed = bytes((ctypes.c_ubyte * SHARED_MEMORY_PROBE_BYTES).from_address(host_ptr))
                expected = bytes([0x5A + region_index]) * SHARED_MEMORY_PROBE_BYTES
                assert observed == expected
            _send_event(connection, "shared_memory_verified")

        _wait_for_command(connection, "register")
        transfer_engine = GlobalTE()
        transfer_engine.get_transfer_engine(host, device_name=None)
        transfer_engine.register_buffer(
            [region.tensor.data_ptr() for region in regions],
            list(region_sizes),
            [f"npu:{device_id}"] * len(regions),
        )
        assert transfer_engine.is_register_buffer
        _send_event(connection, "registration_complete")
        _wait_for_command(connection, "cleanup")
    except BaseException:  # noqa: BLE001
        with contextlib.suppress(BrokenPipeError, EOFError, OSError):
            _send_event(connection, "error", error=traceback.format_exc())
    finally:
        try:
            _release_resources(transfer_engine, regions)
        finally:
            connection.close()


def _run_two_dies_shared_segment_registration(
    region_sizes: tuple[int, ...],
    device_ids: tuple[int, int],
) -> None:
    """Map owner pages on two dies, then register rank 0 before rank 1."""
    xdist_worker_count = os.getenv("PYTEST_XDIST_WORKER_COUNT")
    if xdist_worker_count not in (None, "1"):
        pytest.fail("Run this large-memory test sequentially without pytest-xdist (omit -n)")

    torch = pytest.importorskip("torch", reason="Mooncake NPU registration requires PyTorch")
    pytest.importorskip("torch_npu", reason="Mooncake NPU registration requires torch_npu")
    pytest.importorskip("mooncake.engine", reason="Mooncake TransferEngine is not installed")
    shared_segment = pytest.importorskip(
        "mooncake.shared_segment",
        reason="Mooncake shared_segment is not installed",
    )
    if not shared_segment.shared_segment_supported(mmap=True, host_register=True):
        pytest.skip("Mooncake mmap + HostRegister shared segments are unavailable")
    visible_device_count = torch.npu.device_count()
    assert visible_device_count > max(device_ids), (
        f"This test requires logical dies {device_ids} to be visible, but only "
        f"{visible_device_count} dies are visible. Run with "
        "ASCEND_RT_VISIBLE_DEVICES=0,1,2."
    )

    from vllm.utils.network_utils import get_ip

    host = get_ip()
    owner_device_id, peer_device_id = device_ids
    unique_name = f"e2e_mooncake_two_dies_{owner_device_id}_{peer_device_id}_{os.getpid()}"
    rendezvous_path = str(Path(tempfile.gettempdir()) / f"{unique_name}_{sum(region_sizes)}.rendezvous")
    context = mp.get_context("spawn")
    owner_parent, owner_child = context.Pipe()
    peer_parent, peer_child = context.Pipe()
    owner = context.Process(
        target=_rank_main,
        args=(
            OWNER_RANK,
            owner_device_id,
            host,
            unique_name,
            rendezvous_path,
            region_sizes,
            owner_child,
        ),
    )
    peer = context.Process(
        target=_rank_main,
        args=(
            1,
            peer_device_id,
            host,
            unique_name,
            rendezvous_path,
            region_sizes,
            peer_child,
        ),
    )

    try:
        owner.start()
        peer.start()
        owner_child.close()
        peer_child.close()
        _receive_event(owner_parent, "process_group_ready")
        _receive_event(peer_parent, "process_group_ready")

        # The owner enters the collective allocation first. Only then may the
        # peer map the same owner-backed physical segments.
        owner_parent.send("allocate")
        _receive_event(owner_parent, "owner_allocation_started")
        peer_parent.send("allocate")
        owner_allocation = _receive_event(owner_parent, "allocation_complete")
        peer_allocation = _receive_event(peer_parent, "allocation_complete")
        assert owner_allocation["offsets"] == peer_allocation["offsets"]

        peer_parent.send("verify_shared_memory")
        _receive_event(peer_parent, "shared_memory_verified")

        # TE construction and memory registration are intentionally ordered.
        owner_parent.send("register")
        _receive_event(owner_parent, "registration_complete")
        peer_parent.send("register")
        _receive_event(peer_parent, "registration_complete")
    finally:
        # Unmap the peer before the owner releases the physical allocation.
        for connection, process in (
            (peer_parent, peer),
            (owner_parent, owner),
        ):
            with contextlib.suppress(BrokenPipeError, EOFError, OSError):
                connection.send("cleanup")
            process.join(timeout=CHILD_EXIT_TIMEOUT_SECONDS)
            if process.is_alive():
                process.terminate()
                process.join(timeout=CHILD_EXIT_TIMEOUT_SECONDS)
        for connection in (owner_parent, peer_parent, owner_child, peer_child):
            connection.close()
        Path(rendezvous_path).unlink(missing_ok=True)

    assert owner.exitcode == 0
    assert peer.exitcode == 0


@pytest.mark.parametrize("region_sizes", REGISTER_REGION_SIZES)
def test_same_chip_dies_share_and_register_mooncake_segments_sequentially(
    region_sizes: tuple[int, ...],
) -> None:
    """Exercise die 0 + die 1, the two dies on the same physical chip."""
    _run_two_dies_shared_segment_registration(
        region_sizes,
        SAME_CHIP_DEVICE_IDS,
    )


@pytest.mark.parametrize("region_sizes", REGISTER_REGION_SIZES)
def test_cross_chip_dies_share_and_register_mooncake_segments_sequentially(
    region_sizes: tuple[int, ...],
) -> None:
    """Exercise die 0 + die 2, which reside on different physical chips."""
    _run_two_dies_shared_segment_registration(
        region_sizes,
        CROSS_CHIP_DEVICE_IDS,
    )
