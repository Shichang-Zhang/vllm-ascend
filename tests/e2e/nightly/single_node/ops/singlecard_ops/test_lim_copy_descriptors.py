# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU GM copy/graph test; Host VMM transport is covered by the PD serving run."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.lim_group_onboard import LimGroupOnboarder
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sfa_layer_residency import SfaCacheLayer, SfaCacheLayout
from vllm_ascend.utils import enable_custom_op


def _host_pattern(blocks, block, width, *, salt, device):
    # All values are exactly representable in BF16. The first three channels
    # identify the token and layer/KV component even after dtype conversion.
    token = torch.arange(blocks * block, dtype=torch.int64).reshape(-1, 1)
    channel = torch.arange(width, dtype=torch.int64).reshape(1, -1)
    values = (token * 17 + channel * 13 + salt * 7) % 251
    values[:, 0] = token[:, 0] % 128
    values[:, 1] = token[:, 0] // 128
    values[:, 2] = salt
    return values.reshape(blocks, block, 1, width).to(device=device, dtype=torch.bfloat16)


@pytest.mark.parametrize("graph_mode", [False, True])
def test_lim_copy_dynamic_plan_geometry_and_destinations(graph_mode):
    enable_custom_op()
    device = "npu:0"
    block, hot, rows, host_blocks, stride = 128, 256, 3, 7, 8
    row_tokens, kv_dim, rope_dim = hot + 2 * block, 512, 64
    host_kv = _host_pattern(host_blocks, block, kv_dim, salt=1, device=device)
    host_rope = _host_pattern(host_blocks, block, rope_dim, salt=2, device=device)
    kv = torch.full((rows, row_tokens, 1, kv_dim), -7, device=device, dtype=torch.bfloat16)
    rope = torch.full((rows, row_tokens, 1, rope_dim), -7, device=device, dtype=torch.bfloat16)
    source = torch.full((rows, stride), -1, device=device, dtype=torch.int32)
    destination = torch.zeros_like(source)
    counts = torch.zeros(rows, device=device, dtype=torch.int32)
    table = torch.tensor([[4, 1, 6, 0], [3, 2, 5, 1], [6, 5, 4, 3]], device=device, dtype=torch.int32)
    pools = torch.tensor([2, 0, -1], device=device, dtype=torch.int32)
    host = torch.tensor([host_kv.data_ptr(), host_rope.data_ptr()], device=device, dtype=torch.int64)
    params = torch.tensor(
        [block, hot, stride, kv_dim, rope_dim, host_blocks, rows, rows], device=device, dtype=torch.int64
    )
    geometry = torch.tensor([rows, table.shape[1]], device=device, dtype=torch.int64)

    def copy():
        torch.ops._C_ascend.npu_lim_copy_descriptors(
            source, destination, counts, table, pools, host, kv, rope, params, geometry
        )

    copy()
    torch.npu.synchronize()
    graph = None
    if graph_mode:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            copy()
    for active in (3, 1, 2):
        kv.fill_(-7)
        rope.fill_(-7)
        source[:, :2].copy_(torch.tensor([block + active, 3 * block + 1], device=device, dtype=torch.int32))
        destination[:, :2].copy_(torch.tensor([active, hot - active], device=device, dtype=torch.int32))
        counts.fill_(2)
        geometry[0].fill_(active)
        if graph is not None:
            graph.replay()
        else:
            copy()
        torch.npu.synchronize()
        expected_kv, expected_rope = torch.full_like(kv.cpu(), -7), torch.full_like(rope.cpu(), -7)
        table_cpu, pools_cpu = table.cpu(), pools.cpu()
        for row in range(active):
            pool = int(pools_cpu[row])
            if pool < 0:
                continue
            for logical, slot in ((block + active, active), (3 * block + 1, hot - active)):
                token = int(table_cpu[row, logical // block]) * block + logical % block
                expected_kv[pool, slot, 0] = host_kv.reshape(-1, kv_dim)[token].cpu()
                expected_rope[pool, slot, 0] = host_rope.reshape(-1, rope_dim)[token].cpu()
        torch.testing.assert_close(kv.cpu(), expected_kv, rtol=0, atol=0)
        torch.testing.assert_close(rope.cpu(), expected_rope, rtol=0, atol=0)


def test_real_group_stream_fork_join_and_capture_size_switching():
    enable_custom_op()
    device, block, hot, rows = "npu:0", 128, 256, 3
    owners = [
        SimpleNamespace(
            layer_name=f"owner{i}",
            skip_topk=False,
            lim_full_resident=True,
            lim_miss_src=torch.zeros((rows, 8), device=device, dtype=torch.int32),
            lim_miss_dst=torch.zeros((rows, 8), device=device, dtype=torch.int32),
            lim_misses=torch.ones(rows, device=device, dtype=torch.int32),
            lim_topk_misses=torch.ones(rows, device=device, dtype=torch.int32),
        )
        for i in range(2)
    ]
    names = ("member0", "member1", "member2")
    hosts = [_host_pattern(7, block, 512, salt=2 * i + 1, device=device) for i in range(3)]
    ropes = [_host_pattern(7, block, 64, salt=2 * i + 2, device=device) for i in range(3)]
    manager = SimpleNamespace(
        residency_plan=SimpleNamespace(draft_names=(), groups=(("owner0", names[:2]), ("owner1", names[2:]))),
        sfa_cache_layout=SfaCacheLayout(tuple(SfaCacheLayer(name, 2, True, False) for name in names), 2, block),
        topk_buffers_k=[
            torch.full((rows, hot + 2 * block, 1, 512), -7, device=device, dtype=torch.bfloat16) for _ in names
        ],
        topk_buffers_v=[
            torch.full((rows, hot + 2 * block, 1, 64), -7, device=device, dtype=torch.bfloat16) for _ in names
        ],
        k_caches_cpu=hosts,
        v_caches_cpu=ropes,
        copy_sfa_host_bases=[
            torch.tensor([kv.data_ptr(), rope.data_ptr()], device=device, dtype=torch.int64)
            for kv, rope in zip(hosts, ropes, strict=True)
        ],
        _get_offload_layer_id=names.index,
        block_size=block,
        topk_buffer_size=hot,
        ensure_copy_sfa_host_visibility=lambda **kwargs: None,
    )
    onboarder = LimGroupOnboarder(
        manager, {owner.layer_name: owner for owner in owners}, torch.ops._C_ascend.npu_lim_copy_descriptors.default
    )
    pools = torch.tensor([2, 0, -1], device=device, dtype=torch.int32)
    host_table = torch.tensor([[1, 3, 0], [2, 5, 1], [0, 1, 2]], device=device, dtype=torch.int32)
    tables = (torch.full_like(host_table, 6), torch.full_like(host_table, 4), host_table)
    observed = [torch.empty_like(kv) for kv in manager.topk_buffers_k]
    observed_rope = [torch.empty_like(rope) for rope in manager.topk_buffers_v]

    def run(count):
        token = object()
        md = SimpleNamespace(copy_sfa_pool_entries=pools[:count], sfa_cache_group_tables=tables)
        one = SimpleNamespace(copy_sfa_pool_entries=pools[:1], sfa_cache_group_tables=tables)
        onboarder.launch(owners[0], md, token, capturing=True)
        onboarder.launch(owners[1], one, token, capturing=True)
        for index, name in enumerate(names):
            onboarder.wait(owners[0 if index < 2 else 1], name, token)
            observed[index].copy_(manager.topk_buffers_k[index])
            observed_rope[index].copy_(manager.topk_buffers_v[index])

    graphs = {}
    run(rows)
    torch.npu.synchronize()
    for count in (1, 3):
        graphs[count] = torch.npu.NPUGraph()
        with torch.npu.graph(graphs[count]):
            run(count)
    for count, physical in ((3, 2), (1, 4), (3, 1), (1, 3)):
        host_table[0, 0].fill_(physical)
        for owner in owners:
            owner.lim_miss_src[:, 0].fill_(count)
        for kv in manager.topk_buffers_k:
            kv.fill_(-7)
        for rope in manager.topk_buffers_v:
            rope.fill_(-7)
        graphs[count].replay()
        torch.npu.synchronize()
        for results, sources in ((observed, hosts), (observed_rope, ropes)):
            for index, result in enumerate(results):
                expected = torch.full_like(result.cpu(), -7)
                expected[2, 0] = sources[index][physical, count].cpu()
                if count == 3 and index < 2:
                    expected[0, 0] = sources[index][2, count].cpu()
                torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
    onboarder.close()
