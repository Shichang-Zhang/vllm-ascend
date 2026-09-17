# SPDX-License-Identifier: Apache-2.0
"""Register 70/80/100 GiB as two separately allocated swapped Host regions.

Run from the repository root on an Ascend host with Mooncake Ascend transport
and a torch_npu build providing empty_with_swapped_memory. Use exactly one
visible NPU and run sequentially without pytest-xdist (omit -n). Allow at least
100 GiB of available Host memory plus allocation overhead and 4 MiB alignment
padding. The first region is 64 GiB minus 2 MiB; the second is the remainder.

Run all eight single-region and split-region cases::

    ASCEND_RT_VISIBLE_DEVICES=0 pytest -sv \
        tests/e2e/nightly/single_node/test_mooncake_swapped_memory_registration.py \
        tests/e2e/nightly/single_node/test_mooncake_swapped_memory_split_registration.py

Run only the split-region cases::

    ASCEND_RT_VISIBLE_DEVICES=0 pytest -sv \
        tests/e2e/nightly/single_node/test_mooncake_swapped_memory_split_registration.py

Run one case, for example the 70 GiB split allocation::

    ASCEND_RT_VISIBLE_DEVICES=0 pytest -sv \
        'tests/e2e/nightly/single_node/test_mooncake_swapped_memory_split_registration.py::test_register_split_swapped_host_regions[70GiB-split]'
"""

from typing import Any

import pytest

from tests.e2e.mooncake_swapped_memory_registration import (
    GIB,
    MAX_REGION_SIZE,
    register_swapped_regions,
    swapped_memory_transfer_engine,  # noqa: F401
)


@pytest.mark.parametrize(
    "total_size",
    (
        pytest.param(70 * GIB, id="70GiB-split"),
        pytest.param(80 * GIB, id="80GiB-split"),
        pytest.param(100 * GIB, id="100GiB-split"),
    ),
)
def test_register_split_swapped_host_regions(total_size: int, swapped_memory_transfer_engine: Any) -> None:  # noqa: F811
    register_swapped_regions(swapped_memory_transfer_engine, (MAX_REGION_SIZE, total_size - MAX_REGION_SIZE))
