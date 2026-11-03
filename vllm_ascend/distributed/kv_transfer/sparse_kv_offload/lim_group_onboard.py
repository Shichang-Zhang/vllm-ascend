# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Captured target-owner LIM -> member miss copy -> member attention edges.

Only target owners with unconditional LIM execution participate. Draft owners
and their reuse forwards retain inline Copy-SFA. Every graph contains its own
plan record, side-stream copies and member waits; Python launch tokens check
eager/capture execution, and are never the replay synchronization mechanism.
"""

from dataclasses import dataclass
from typing import Any

import torch
import torch_npu
from vllm.logger import logger

from vllm_ascend.utils import AscendDeviceType, enable_custom_op, get_ascend_device_type

LIM_COPY_ARGUMENT_NAMES = (
    "miss_src_ids",
    "miss_dst_slots",
    "miss_counts",
    "source_block_table",
    "pool_entries",
    "host_ptrs",
    "hbm_kv_cache",
    "hbm_k_rope",
    "params",
    "active_geometry",
)
LIM_KV_DIM_MAX = 512
LIM_ROPE_DIM_MAX = 64


def resolve_lim_copy_operator():
    # The Torch schema is registered on other SoCs too, but this ACLNN
    # implementation is only included in the ascend910_93 build list.
    if get_ascend_device_type() is not AscendDeviceType.A3:
        logger.warning("LIM side-stream copy is built for A3; using inline miss copies on this SoC")
        return None
    # Cache initialization is after device selection; resolve the lazy
    # extension here instead of depending on an earlier profile invocation.
    enable_custom_op()
    operation = getattr(torch.ops._C_ascend, "npu_lim_copy_descriptors", None)
    if operation is None:
        logger.warning("npu_lim_copy_descriptors is unavailable; using inline miss copies")
        return None
    operation = operation.default
    if tuple(argument.name for argument in operation._schema.arguments) != LIM_COPY_ARGUMENT_NAMES:
        raise RuntimeError("LIM copy operator ABI mismatch; rebuild the extension and A3 custom operators together")
    return operation


@dataclass(frozen=True)
class LimMemberCopy:
    name: str
    host_ptrs: torch.Tensor
    kv: torch.Tensor
    rope: torch.Tensor
    params: torch.Tensor
    done: Any


@dataclass
class LimGroupCopy:
    owner: Any
    host_group_id: int
    members: tuple[LimMemberCopy, ...]
    geometry: torch.Tensor
    plan_ready: Any
    zero_topk_counts: torch.Tensor
    zero_miss_counts: torch.Tensor
    launch_token: object | None = None


class LimGroupOnboarder:
    def __init__(self, manager, implementations, operation, *, runtime=None):
        self.manager = manager
        self.operation = operation
        self.runtime = torch_npu.npu if runtime is None else runtime
        self.groups: dict[str, LimGroupCopy] = {}
        self.member_owners: dict[str, str] = {}
        self.closed = False
        plan, layout = manager.residency_plan, manager.sfa_cache_layout
        if plan is None or layout is None:
            raise ValueError("LIM group onboarding requires a validated mixed cache layout")
        layers = {layer.name: layer for layer in layout.layers}
        drafts = set(plan.draft_names)
        device = manager.topk_buffers_k[0].device
        self.stream = self.runtime.Stream(device)
        for owner_name, member_names in plan.groups:
            if not member_names or owner_name in drafts:
                continue
            owner = implementations[owner_name]
            if owner.skip_topk or not owner.lim_full_resident or any(name in drafts for name in member_names):
                raise ValueError("side-stream groups require an unconditional resident target owner and target members")
            plan_rows, miss_stride = owner.lim_miss_src.shape
            if owner.lim_miss_dst.shape != owner.lim_miss_src.shape or owner.lim_misses.shape != (plan_rows,):
                raise ValueError("inconsistent owner LIM plan capacity")
            if any(
                t.dtype != torch.int32 or not t.is_contiguous() or t.device != device
                for t in (owner.lim_miss_src, owner.lim_miss_dst, owner.lim_misses)
            ):
                raise ValueError("LIM plan tensors must be contiguous int32 on the cache device")
            members = []
            for name in member_names:
                layer = layers[name]
                if not layer.store_on_host or layer.group_id != layout.host_group_id:
                    raise ValueError("LIM member must use the validated Host cache group")
                index = manager._get_offload_layer_id(name)
                kv, rope = manager.topk_buffers_k[index], manager.topk_buffers_v[index]
                host_kv, host_rope = manager.k_caches_cpu[index], manager.v_caches_cpu[index]
                block, hot = manager.block_size, manager.topk_buffer_size
                row_tokens = hot + 2 * block
                kv_dim, rope_dim = kv.shape[-1], rope.shape[-1]
                if not (0 < kv_dim <= LIM_KV_DIM_MAX and 0 < rope_dim <= LIM_ROPE_DIM_MAX):
                    raise ValueError("LIM copy head dimensions exceed the kernel UB capacity")
                if (
                    kv.ndim != 4
                    or rope.ndim != 4
                    or kv.shape[:3] != rope.shape[:3]
                    or kv.shape[1:3] != (row_tokens, 1)
                    or kv.shape[0] < plan_rows
                    or any(t.dtype != torch.bfloat16 or not t.is_contiguous() for t in (kv, rope, host_kv, host_rope))
                    or kv.device != device
                    or rope.device != device
                ):
                    raise ValueError("LIM copy requires private contiguous BF16 hot rows matching the actual geometry")
                host_blocks = host_kv.numel() // (block * kv_dim)
                if (
                    host_blocks <= 0
                    or host_kv.numel() != host_blocks * block * kv_dim
                    or host_rope.numel() != host_blocks * block * rope_dim
                ):
                    raise ValueError("LIM copy Host KV capacities disagree")
                params = torch.tensor(
                    [block, hot, miss_stride, kv_dim, rope_dim, host_blocks, kv.shape[0], plan_rows],
                    dtype=torch.int64,
                    device=device,
                )
                members.append(
                    LimMemberCopy(
                        name, manager.copy_sfa_host_bases[index].view(2), kv, rope, params, self.runtime.Event()
                    )
                )
                self.member_owners[name] = owner_name
            self.groups[owner_name] = LimGroupCopy(
                owner,
                layout.host_group_id,
                tuple(members),
                torch.zeros(2, dtype=torch.int64, device=device),
                self.runtime.Event(),
                torch.zeros_like(owner.lim_topk_misses),
                torch.zeros_like(owner.lim_misses),
            )
        # Fail initialization on an ABI/build/kernel error. Only absence and
        # unsupported SoC use the explicit inline fallback above. Zero active
        # rows touch no Host memory and compile the operator before capture.
        dummy_table = torch.zeros((1, 1), dtype=torch.int32, device=device)
        dummy_pool = torch.full((1,), -1, dtype=torch.int32, device=device)
        for group in self.groups.values():
            for member in group.members:
                operation(
                    group.owner.lim_miss_src,
                    group.owner.lim_miss_dst,
                    group.owner.lim_misses,
                    dummy_table,
                    dummy_pool,
                    member.host_ptrs,
                    member.kv,
                    member.rope,
                    member.params,
                    group.geometry,
                )
        self.runtime.current_stream().synchronize()

    def launch(self, owner, metadata, token: object, *, capturing: bool) -> None:
        group = self.groups.get(owner.layer_name)
        if group is None:
            return
        if self.closed or owner.skip_topk or token is None:
            raise RuntimeError("LIM target group has no unconditional current-forward plan")
        count = metadata.copy_sfa_pool_entries.numel()
        tables = metadata.sfa_cache_group_tables
        if tables is None or group.host_group_id >= len(tables):
            raise RuntimeError("LIM side copy requires the actual member Host-group table")
        table = tables[group.host_group_id][:count]
        pools = metadata.copy_sfa_pool_entries
        if (
            not 0 < count <= owner.lim_miss_src.shape[0]
            or table.shape[0] != count
            or table.shape[1] <= 0
            or table.dtype != torch.int32
            or pools.dtype != torch.int32
            or table.device != group.geometry.device
            or pools.device != group.geometry.device
            or not table.is_contiguous()
            or not pools.is_contiguous()
        ):
            raise ValueError("invalid current LIM copy geometry")
        # Main-stream writes, visible through the captured plan-ready edge.
        # Each group owns its geometry, so a later owner cannot overwrite it.
        self.manager.ensure_copy_sfa_host_visibility(capturing=capturing)
        group.geometry[0].fill_(count)
        group.geometry[1].fill_(table.shape[1])
        group.plan_ready.record(self.runtime.current_stream())
        with self.runtime.stream(self.stream):
            self.stream.wait_event(group.plan_ready)
            for member in group.members:
                self.operation(
                    owner.lim_miss_src,
                    owner.lim_miss_dst,
                    owner.lim_misses,
                    table,
                    pools,
                    member.host_ptrs,
                    member.kv,
                    member.rope,
                    member.params,
                    group.geometry,
                )
                member.done.record(self.stream)
        group.launch_token = token

    def wait(self, owner, member_name: str, token: object):
        if self.closed:
            raise RuntimeError("LIM group onboarding is closed")
        group = self.groups[self.member_owners[member_name]]
        if group.owner is not owner or token is None or group.launch_token is not token:
            raise RuntimeError("LIM member has no copy launched by its owner in this forward")
        member = next(member for member in group.members if member.name == member_name)
        self.runtime.current_stream().wait_event(member.done)
        return group.zero_topk_counts, group.zero_miss_counts

    def close(self) -> None:
        if not self.closed:
            self.stream.synchronize()
            self.closed = True
