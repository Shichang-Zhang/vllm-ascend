# SPDX-License-Identifier: Apache-2.0
"""Shared setup for sequential, single-die swapped-memory registration tests."""

from __future__ import annotations

import gc
import os
from collections.abc import Iterator
from typing import Any

import pytest

GIB = 1024**3
MIB = 1024**2
ALIGNMENT = 2 * MIB
MAX_REGION_SIZE = 64 * GIB - ALIGNMENT
DEVICE_ID = 0
REGISTER_LOCATION = f"npu:{DEVICE_ID}"


@pytest.fixture(scope="module")
def swapped_memory_transfer_engine() -> Iterator[Any]:
    """Bind one production TransferEngine to one visible NPU."""
    if os.getenv("PYTEST_XDIST_WORKER_COUNT") not in (None, "1"):
        pytest.fail("Run these large-memory cases sequentially without pytest-xdist (omit -n)")

    torch = pytest.importorskip("torch")
    torch_npu = pytest.importorskip("torch_npu")
    pytest.importorskip("mooncake.engine")
    if not callable(getattr(torch_npu, "empty_with_swapped_memory", None)):
        pytest.skip("This torch_npu build does not provide empty_with_swapped_memory")
    if torch.npu.device_count() == 0:
        pytest.skip("Swapped-memory registration requires an NPU")
    assert torch.npu.device_count() == 1, "Bind exactly one die with ASCEND_RT_VISIBLE_DEVICES=<die-id>"

    from vllm.utils.network_utils import get_ip

    from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import GlobalTE

    torch.npu.set_device(DEVICE_ID)
    os.environ["HCCL_INTRA_ROCE_ENABLE"] = "1"
    engine = GlobalTE()
    engine.get_transfer_engine(get_ip(), device_name=None)
    try:
        yield engine
    finally:
        if engine.is_register_buffer:
            engine.unregister_buffer()


def register_swapped_regions(engine: Any, region_sizes: tuple[int, ...]) -> None:
    """Allocate independent aligned Host tensors and register their exact sizes.

    Allocation follows _allocate_swapped_host_tensor in GitCode branch
    mte_fuised_rebase_0713_mooncake_test-0817-both-pd, model_runner_v1.py:4232.
    Each region uses one empty_with_swapped_memory call including alignment
    padding; there is no chunking of the single-region cases.
    """
    import torch
    import torch_npu

    regions = []
    try:
        for size_bytes in region_sizes:
            raw = torch_npu.empty_with_swapped_memory((size_bytes + ALIGNMENT,), dtype=torch.uint8, device="npu")
            offset = (-raw.data_ptr()) % ALIGNMENT
            regions.append(raw[offset : offset + size_bytes])
            # The view retains the backing storage until after unregistration.
            del raw

        for region, size_bytes in zip(regions, region_sizes):
            assert region.device.type == "npu"
            assert region.is_contiguous()
            assert region.numel() * region.element_size() == size_bytes
            assert region.data_ptr() % ALIGNMENT == 0
        del region

        assert not engine.is_register_buffer
        print(f"Registering swapped Host regions: sizes={region_sizes}, location={REGISTER_LOCATION}")
        engine.register_buffer(
            [region.data_ptr() for region in regions],
            list(region_sizes),
            [REGISTER_LOCATION] * len(regions),
        )
        assert engine.is_register_buffer
    finally:
        # A registration failure may leave regions tracked after failed rollback.
        # Never release their storage before unregistration succeeds.
        if engine.is_register_buffer:
            engine.unregister_buffer()
        regions.clear()
        gc.collect()
        torch.npu.empty_cache()
    assert not engine.is_register_buffer
