# SPDX-License-Identifier: Apache-2.0
"""Register split swapped Host regions, including three independent 32 GiB regions.

Run from the repository root on an Ascend host with Mooncake Ascend transport
and a torch_npu build providing empty_with_swapped_memory. Use exactly one
visible NPU and run sequentially without pytest-xdist (omit -n). Allow at least
100 GiB of available Host memory plus allocation overhead and 4 MiB alignment
padding (6 MiB for the three-region case). The two-region cases use 64 GiB
minus 2 MiB for the first region and the remainder for the second.
Each configuration runs with both uint8 and bfloat16. The reference allocator
defaults to bfloat16; uint8 also exercises byte-sized allocations.

Run all 18 single-region and split-region cases::

    ASCEND_RT_VISIBLE_DEVICES=0 pytest -sv \
        tests/e2e/nightly/single_node/test_mooncake_swapped_memory_registration.py \
        tests/e2e/nightly/single_node/test_mooncake_swapped_memory_split_registration.py

Run only the split-region cases::

    ASCEND_RT_VISIBLE_DEVICES=0 pytest -sv \
        tests/e2e/nightly/single_node/test_mooncake_swapped_memory_split_registration.py

Run one case, for example the 70 GiB split allocation::

    ASCEND_RT_VISIBLE_DEVICES=0 pytest -sv \
        tests/e2e/nightly/single_node/test_mooncake_swapped_memory_split_registration.py \
        -k '70GiB-split and uint8'

Run the three-region case with both dtypes::

    ASCEND_RT_VISIBLE_DEVICES=0 pytest -sv \
        tests/e2e/nightly/single_node/test_mooncake_swapped_memory_split_registration.py \
        -k three
"""

from typing import Any

import pytest

from tests.e2e.mooncake_swapped_memory_registration import (
    GIB,
    MAX_REGION_SIZE,
    register_swapped_regions,
    swapped_memory_engine_fixture,  # noqa: F401
)


@pytest.mark.parametrize("dtype_name", ("uint8", "bfloat16"))
@pytest.mark.parametrize(
    "total_size",
    (
        pytest.param(70 * GIB, id="70GiB-split"),
        pytest.param(80 * GIB, id="80GiB-split"),
        pytest.param(100 * GIB, id="100GiB-split"),
    ),
)
def test_register_split_swapped_host_regions(
    total_size: int, dtype_name: str, swapped_memory_transfer_engine: Any
) -> None:
    register_swapped_regions(
        swapped_memory_transfer_engine, (MAX_REGION_SIZE, total_size - MAX_REGION_SIZE), dtype_name
    )


@pytest.mark.parametrize("dtype_name", ("uint8", "bfloat16"))
def test_register_three_swapped_host_regions(dtype_name: str, swapped_memory_transfer_engine: Any) -> None:
    """Allocate three 32 GiB segments before registering them in one call."""
    register_swapped_regions(swapped_memory_transfer_engine, (32 * GIB,) * 3, dtype_name)
