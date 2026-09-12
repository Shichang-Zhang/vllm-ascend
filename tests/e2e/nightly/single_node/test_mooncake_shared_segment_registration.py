# SPDX-License-Identifier: Apache-2.0
"""Exercise large Mooncake Host-memory registration on one A3 die.

Run this test in a process that can see exactly one logical NPU, for example::

    ASCEND_RT_VISIBLE_DEVICES=0 pytest -sv \
        tests/e2e/nightly/single_node/test_mooncake_shared_segment_registration.py

The three parameter cases run sequentially in one pytest process and reuse one
Mooncake TransferEngine. Each shared segment is unregistered before the next
case allocates its segment. Do not run this module with pytest-xdist: the peak
shared-segment allocation is approximately 100 GiB plus 2 MiB of alignment
overhead.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import pytest

GIB = 1024**3
MIB = 1024**2
SHARED_SEGMENT_ALIGNMENT = 2 * MIB
TEST_DEVICE_ID = 0
TEST_REGISTER_LOCATION = f"npu:{TEST_DEVICE_ID}"

REGISTER_SIZES = (
    pytest.param(32 * GIB, id="32GiB"),
    pytest.param(64 * GIB - 2 * MIB, id="64GiB-minus-2MiB"),
    pytest.param(100 * GIB, id="100GiB"),
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


@pytest.mark.parametrize("size_bytes", REGISTER_SIZES)
def test_register_large_mooncake_shared_segment(
    size_bytes: int,
    single_die_transfer_engine: Any,
) -> None:
    """Allocate and register one large DRAM segment through production helpers."""
    from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.mooncake_host_pool import (
        HostPoolTopology,
        allocate_mooncake_host_region,
    )

    region = None
    try:
        region = allocate_mooncake_host_region(
            size_bytes=size_bytes,
            alignment=SHARED_SEGMENT_ALIGNMENT,
            topology=HostPoolTopology(
                tp_rank=0,
                tp_size=1,
                device_id=TEST_DEVICE_ID,
            ),
            name=f"e2e_mooncake_te_registration_{os.getpid()}_{size_bytes}",
        )

        assert region.tensor.device.type == "npu"
        assert region.tensor.is_contiguous()
        assert region.tensor.numel() * region.tensor.element_size() == size_bytes
        assert region.tensor.data_ptr() % SHARED_SEGMENT_ALIGNMENT == 0

        # DSA Host pools use an explicit npu:<device_id> location so Mooncake
        # registers the Host VA against the die that consumes the shared pool.
        single_die_transfer_engine.register_buffer(
            [region.tensor.data_ptr()],
            [size_bytes],
            [TEST_REGISTER_LOCATION],
        )
        assert single_die_transfer_engine.is_register_buffer
    finally:
        # Registration must not outlive the shared-segment backing allocation.
        if single_die_transfer_engine.is_register_buffer:
            single_die_transfer_engine.unregister_buffer()
        if region is not None:
            region.release()
