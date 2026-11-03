# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.ut.kv_offload.test_delayed_d2h_executor import FakeRuntime, cpu_pinned_allocation  # noqa: F401
from tests.ut.kv_offload.test_fused_copy_sfa_config import make_config
from tests.ut.kv_offload.test_lim_group_onboard import fixture_groups
from vllm_ascend.ascend_config import SparseKVOffloadConfig
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.delayed_d2h import DelayedD2HPlanner
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.delayed_d2h_executor import (
    D2HCopyComponent,
    MooncakeDelayedD2HExecutor,
)


@pytest.mark.parametrize(
    "name",
    [
        "a_overlap",
        "b_copy_sfa",
        "c_delayed",
        "d_shared_inline",
        "e_combined_inline",
        "f_shared_side_stream",
        "g_combined_side_stream",
    ],
)
def test_shipped_additional_configs_pass_real_mtp3_serving_validation(name):
    root = Path(__file__).resolve().parents[3]
    config = json.loads(
        (root / "examples/disaggregated_prefill_v1/mooncake_dsa_migration" / f"{name}.json").read_text()
    )
    parsed = SparseKVOffloadConfig(make_config(drafts=3, max_model_len=1048576), config["sparse_kv_offload_config"])
    assert parsed.enabled and parsed.host_backend == "mooncake"


def test_delayed_commit_then_side_onboard_copies_new_host_contents_without_future_writes():
    onboarder, manager, owners, md, _, _ = fixture_groups()
    components = []
    for i in range(3):
        for suffix, resident, host in (
            ("kv", manager.topk_buffers_k[i], manager.k_caches_cpu[i]),
            ("rope", manager.topk_buffers_v[i], manager.v_caches_cpu[i]),
        ):
            host.fill_(-99)
            resident.fill_(-7)
            resident[2, 4:8].copy_(torch.arange(41, 45, dtype=torch.bfloat16).view(4, 1, 1).expand_as(resident[2, 4:8]))
            components.append(D2HCopyComponent(f"member{i}.{suffix}", resident, host))
    planner = DelayedD2HPlanner(capacity=3, block_size=4, hot_tokens=8, host_num_blocks=6)
    slots, generations = np.array([2]), np.array([1])
    planner.bind(slots, generations, np.array([4]))
    runtime = FakeRuntime()
    executor = MooncakeDelayedD2HExecutor(
        planner, components, runtime=runtime, tp_group=None, tp_rank=0, tp_size=1, copy_blocks=runtime.copy_blocks
    )
    planner.plan(
        computed_tokens=np.array([8]),
        pool_slots=slots,
        generations=generations,
        max_write_tokens=np.array([1]),
        block_table=np.array([[1, 3, 0]]),
    )
    executor.record_source_ready()
    executor.launch()
    assert planner.prefixes(slots, generations).tolist() == [4]
    for component in components:
        assert torch.all(component.host == -99)
        component.resident[2, 8].fill_(-33)  # next forward's uncommitted tail
    executor.retire()
    assert planner.prefixes(slots, generations).tolist() == [8]
    token = object()
    current = SimpleNamespace(
        copy_sfa_pool_entries=md.copy_sfa_pool_entries[:1], sfa_cache_group_tables=md.sfa_cache_group_tables
    )
    for owner in owners:
        owner.lim_miss_src[0] = torch.tensor([5, 7], dtype=torch.int32)
        onboarder.launch(owner, current, token, capturing=False)
    for i, name in enumerate(("member0", "member1", "member2")):
        onboarder.wait(owners[0 if i < 2 else 1], name, token)
        assert torch.all(manager.topk_buffers_k[i][2, 0] == 42)
        assert torch.all(manager.topk_buffers_k[i][2, 1] == 44)
        assert torch.all(manager.k_caches_cpu[i][0] == -99)
        assert torch.all(manager.topk_buffers_k[i][2, 8] == -33)
    onboarder.close()
    executor.close()
