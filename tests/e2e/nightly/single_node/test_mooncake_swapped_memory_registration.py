# SPDX-License-Identifier: Apache-2.0
"""Register single swapped Host regions of 32, 64-minus-2MiB, 70, 80, 100 GiB.

Run from the repository root on an Ascend host with Mooncake Ascend transport
and a torch_npu build providing empty_with_swapped_memory. Use exactly one
visible NPU and run sequentially without pytest-xdist (omit -n). Allow at least
100 GiB of available Host memory plus allocation overhead (up to 6 MiB of
alignment padding when running the three-region case).

Each configuration runs with both uint8 and bfloat16. The reference allocator
defaults to bfloat16; uint8 also exercises byte-sized allocations.

Run all 18 single-region and split-region cases::

    ASCEND_RT_VISIBLE_DEVICES=0 pytest -sv \
        tests/e2e/nightly/single_node/test_mooncake_swapped_memory_registration.py \
        tests/e2e/nightly/single_node/test_mooncake_swapped_memory_split_registration.py

Run only the single-region cases::

    ASCEND_RT_VISIBLE_DEVICES=0 pytest -sv \
        tests/e2e/nightly/single_node/test_mooncake_swapped_memory_registration.py

Run one case, for example the 70 GiB single-region allocation::

    ASCEND_RT_VISIBLE_DEVICES=0 pytest -sv \
        tests/e2e/nightly/single_node/test_mooncake_swapped_memory_registration.py \
        -k '70GiB and uint8'
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
    "size_bytes",
    (
        pytest.param(32 * GIB, id="32GiB"),
        pytest.param(MAX_REGION_SIZE, id="64GiB-minus-2MiB"),
        pytest.param(70 * GIB, id="70GiB"),
        pytest.param(80 * GIB, id="80GiB"),
        pytest.param(100 * GIB, id="100GiB"),
    ),
)
def test_register_single_swapped_host_region(
    size_bytes: int, dtype_name: str, swapped_memory_transfer_engine: Any
) -> None:
    register_swapped_regions(swapped_memory_transfer_engine, (size_bytes,), dtype_name)
