# SPDX-License-Identifier: Apache-2.0
"""Exercise large Mooncake Host-memory registration on one A3 die.

Run this test in a process that can see exactly one logical NPU, for example::

    ASCEND_RT_VISIBLE_DEVICES=0 pytest -sv \
        tests/e2e/nightly/single_node/test_mooncake_shared_segment_registration.py

The three parameter cases run sequentially in one pytest process and reuse one
Mooncake TransferEngine. Each case is unregistered before the next case
allocates its shared segments. Do not run this module with pytest-xdist: the
peak shared-segment allocation is approximately 100 GiB plus 4 MiB of alignment
overhead.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import pytest

GIB = 1024**3
MIB = 1024**2
KIB = 1024
SHARED_SEGMENT_ALIGNMENT = 2 * MIB
TEST_DEVICE_ID = 0
TEST_REGISTER_LOCATION = f"npu:{TEST_DEVICE_ID}"
MAX_REGISTER_REGION_SIZE = 64 * GIB - 2 * MIB
LARGE_REGISTER_TOTAL_SIZE = 100 * GIB
PMD_MAPPED_FIELDS = ("FilePmdMapped", "ShmemPmdMapped")
SMAPS_ROLLUP_PATH = "/proc/self/smaps_rollup"
MTHP_SIZE_BYTES = 2 * MIB
MTHP_SYSFS_PATH = "/sys/kernel/mm/transparent_hugepage/hugepages-2048kB"
MTHP_SHMEM_STAT_FIELDS = ("shmem_alloc", "shmem_fallback", "shmem_fallback_charge")

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


def _read_pmd_mapped_bytes() -> dict[str, int]:
    """Read this process's file- and shmem-backed PMD mappings."""
    mapped_bytes: dict[str, int] = {}
    with open(SMAPS_ROLLUP_PATH, encoding="utf-8") as smaps_rollup:
        for line in smaps_rollup:
            field, separator, remainder = line.partition(":")
            if not separator or field not in PMD_MAPPED_FIELDS:
                continue
            value, unit = remainder.split()
            if unit != "kB":
                raise RuntimeError(f"Unexpected {field} unit in {SMAPS_ROLLUP_PATH}: {unit}")
            mapped_bytes[field] = int(value) * KIB

    missing_fields = set(PMD_MAPPED_FIELDS) - mapped_bytes.keys()
    if missing_fields:
        raise RuntimeError(f"Missing PMD mapping counters in {SMAPS_ROLLUP_PATH}: {sorted(missing_fields)}")
    return mapped_bytes


def _read_mthp_shmem_stats() -> tuple[str, dict[str, int]] | None:
    """Read the system-wide 2 MiB shmem mTHP policy and counters."""
    policy_path = os.path.join(MTHP_SYSFS_PATH, "shmem_enabled")
    try:
        with open(policy_path, encoding="utf-8") as policy_file:
            policy = policy_file.read().strip()
    except FileNotFoundError:
        return None

    stats = {}
    for field in MTHP_SHMEM_STAT_FIELDS:
        stat_path = os.path.join(MTHP_SYSFS_PATH, "stats", field)
        try:
            with open(stat_path, encoding="utf-8") as stat_file:
                stats[field] = int(stat_file.read())
        except FileNotFoundError:
            continue
    return policy, stats


def _log_mthp_shmem_delta(
    before: tuple[str, dict[str, int]] | None,
    after: tuple[str, dict[str, int]] | None,
) -> None:
    """Log system-wide 2 MiB shmem mTHP allocation activity."""
    if before is None or after is None:
        print(f"Mooncake 2 MiB shmem mTHP stats unavailable at {MTHP_SYSFS_PATH}")
        return

    policy, after_stats = after
    _, before_stats = before
    stat_deltas = {
        field: after_stats[field] - before_stats[field]
        for field in MTHP_SHMEM_STAT_FIELDS
        if field in before_stats and field in after_stats
    }
    allocated_pages = stat_deltas.get("shmem_alloc")
    allocated_gib = None if allocated_pages is None else allocated_pages * MTHP_SIZE_BYTES / GIB
    delta_log = ", ".join(f"{field}_delta={stat_deltas.get(field, 'unavailable')}" for field in MTHP_SHMEM_STAT_FIELDS)
    allocated_log = "unavailable" if allocated_gib is None else f"{allocated_gib:.3f} GiB"
    print(
        "Mooncake system-wide 2 MiB shmem mTHP activity after allocation: "
        f"shmem_enabled={policy!r}, {delta_log}, allocated={allocated_log}"
    )


@pytest.fixture(scope="module")
def single_die_transfer_engine() -> Iterator[Any]:
    """Create one TE after verifying that the process is bound to one die."""
    xdist_worker_count = os.getenv("PYTEST_XDIST_WORKER_COUNT")
    if xdist_worker_count not in (None, "1"):
        pytest.fail("Run this large-memory test sequentially without pytest-xdist (omit -n)")

    torch = pytest.importorskip("torch", reason="Mooncake NPU registration requires PyTorch")
    pytest.importorskip("torch_npu", reason="Mooncake NPU registration requires torch_npu")
    pytest.importorskip("mooncake.engine", reason="Mooncake TransferEngine is not installed")
    pytest.importorskip("mooncake.shared_segment", reason="Mooncake shared_segment is not installed")
    from vllm.utils.network_utils import get_ip

    from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import (
        GlobalTE,
    )

    visible_npus = torch.npu.device_count()
    if visible_npus == 0:
        pytest.skip("Mooncake shared-segment registration requires an NPU")
    assert visible_npus == 1, (
        "This test must bind its single TransferEngine to exactly one A3 die; "
        "run pytest with ASCEND_RT_VISIBLE_DEVICES=<die-id>. "
        f"The process currently sees {visible_npus} NPUs."
    )

    torch.npu.set_device(TEST_DEVICE_ID)
    os.environ["HCCL_INTRA_ROCE_ENABLE"] = "1"
    transfer_engine = GlobalTE()
    # MooncakeConnectorV1 uses device_name=None for a single pipeline stage.
    # That initializes the Ascend transport against the current logical die.
    transfer_engine.get_transfer_engine(get_ip(), device_name=None)
    try:
        yield transfer_engine
    finally:
        if transfer_engine.is_register_buffer:
            transfer_engine.unregister_buffer()


@pytest.mark.parametrize("region_sizes", REGISTER_REGION_SIZES)
def test_register_large_mooncake_shared_segment(
    region_sizes: tuple[int, ...],
    single_die_transfer_engine: Any,
) -> None:
    """Allocate and register large DRAM segments through production helpers."""
    from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.mooncake_host_pool import (
        HostPoolTopology,
        allocate_mooncake_host_region,
    )

    regions = []
    total_size_bytes = sum(region_sizes)
    pmd_mapped_before = _read_pmd_mapped_bytes()
    mthp_shmem_before = _read_mthp_shmem_stats()
    try:
        for region_index, size_bytes in enumerate(region_sizes):
            region = allocate_mooncake_host_region(
                size_bytes=size_bytes,
                alignment=SHARED_SEGMENT_ALIGNMENT,
                topology=HostPoolTopology(
                    tp_rank=0,
                    tp_size=1,
                    device_id=TEST_DEVICE_ID,
                ),
                name=f"e2e_mooncake_te_registration_{os.getpid()}_{total_size_bytes}_{region_index}",
            )
            regions.append(region)

            assert region.tensor.device.type == "npu"
            assert region.tensor.is_contiguous()
            assert region.tensor.numel() * region.tensor.element_size() == size_bytes
            assert region.tensor.data_ptr() % SHARED_SEGMENT_ALIGNMENT == 0

        pmd_mapped_after = _read_pmd_mapped_bytes()
        allocated_pmd_bytes = {
            field: pmd_mapped_after[field] - pmd_mapped_before[field] for field in PMD_MAPPED_FIELDS
        }
        total_allocated_pmd_bytes = sum(allocated_pmd_bytes.values())
        print(
            "Mooncake shared-segment PMD allocation after allocation: "
            f"requested={total_size_bytes / GIB:.3f} GiB, "
            f"FilePmdMapped={allocated_pmd_bytes['FilePmdMapped'] / GIB:.3f} GiB, "
            f"ShmemPmdMapped={allocated_pmd_bytes['ShmemPmdMapped'] / GIB:.3f} GiB, "
            f"total={total_allocated_pmd_bytes / GIB:.3f} GiB"
        )
        _log_mthp_shmem_delta(mthp_shmem_before, _read_mthp_shmem_stats())

        # DSA Host pools use an explicit npu:<device_id> location so Mooncake
        # registers the Host VA against the die that consumes the shared pool.
        single_die_transfer_engine.register_buffer(
            [region.tensor.data_ptr() for region in regions],
            list(region_sizes),
            [TEST_REGISTER_LOCATION] * len(regions),
        )
        assert single_die_transfer_engine.is_register_buffer
    finally:
        # Registration must not outlive the shared-segment backing allocation.
        if single_die_transfer_engine.is_register_buffer:
            single_die_transfer_engine.unregister_buffer()
        for region in reversed(regions):
            region.release()
