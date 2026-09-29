# SPDX-License-Identifier: Apache-2.0
"""Compare separate and combined plan broadcasts on one node with multiple NPUs.

Example (requires the vLLM Ascend runtime and two NPUs)::

    python benchmarks/scripts/benchmark_sparse_kv_plan_broadcast.py \
        --world-size 2 --rows 32 --topk 2048 --iterations 200

Reports per-rank mean wall time per broadcast sequence, including submission
and final device synchronization. This eager communication microbenchmark
excludes planner execution, swapped-memory publication, and graph replay.
Both case orders are measured; it does not impose a hardware-specific threshold.
"""

import argparse
import json
import tempfile
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu

from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import SparseKVOffloadManager


def run_rank(rank: int, args: argparse.Namespace, rendezvous: str) -> None:
    torch_npu.npu.set_device(rank)
    dist.init_process_group(
        "hccl", rank=rank, world_size=args.world_size, init_method=rendezvous, timeout=timedelta(seconds=120)
    )
    try:
        manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
        manager.topk = args.topk
        manager.fused_plan_metadata_npu = torch.zeros(args.rows + 1, dtype=torch.int32, device=f"npu:{rank}")
        manager._allocate_fused_plan_staging(args.rows, torch.device(f"npu:{rank}"))
        metadata = manager.fused_plan_metadata_npu
        plan = manager.fused_overlap_membership_plan_device_staging
        payload = manager.fused_plan_broadcast_buffer
        separate_metadata = torch.zeros_like(metadata)
        separate_plan = torch.zeros_like(plan)
        # Exercise int32 slot values that cannot survive an int16 numeric cast.
        expected_metadata = torch.arange(args.rows + 1, dtype=torch.int32) + torch.iinfo(torch.int16).max
        expected_metadata[0] = 0
        expected_plan = torch.arange(plan.numel(), dtype=torch.int64).to(torch.int16).view(plan.shape)

        def broadcast(combined: bool) -> None:
            if combined:
                dist.broadcast(payload, src=0)
            else:
                dist.broadcast(separate_metadata, src=0)
                dist.broadcast(separate_plan, src=0)

        measurements = []
        for combined in (False, True, True, False):
            case_metadata = metadata if combined else separate_metadata
            case_plan = plan if combined else separate_plan
            payload.zero_()
            separate_metadata.zero_()
            separate_plan.zero_()
            if rank == 0:
                case_metadata.copy_(expected_metadata)
                case_plan.copy_(expected_plan)
            torch_npu.npu.synchronize()
            dist.barrier()
            broadcast(combined)
            torch_npu.npu.synchronize()
            torch.testing.assert_close(case_metadata.cpu(), expected_metadata)
            torch.testing.assert_close(case_plan.cpu(), expected_plan)
            for _ in range(args.warmup):
                broadcast(combined)
            dist.barrier()
            torch_npu.npu.synchronize()
            start = time.perf_counter()
            for _ in range(args.iterations):
                broadcast(combined)
            torch_npu.npu.synchronize()
            measurements.append(
                {
                    "combined": combined,
                    "mean_seconds": (time.perf_counter() - start) / args.iterations,
                }
            )
        print(json.dumps({"rank": rank, "config": vars(args), "measurements": measurements}), flush=True)
    finally:
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()
    if args.world_size < 2 or min(args.rows, args.topk, args.iterations) <= 0 or args.warmup < 0:
        parser.error("world-size must be at least 2; rows, topk, iterations positive; warmup nonnegative")
    if torch_npu.npu.device_count() < args.world_size:
        parser.error("not enough local NPUs")
    with tempfile.TemporaryDirectory(prefix="sparse-plan-broadcast-") as directory:
        rendezvous = (Path(directory) / "store").as_uri()
        mp.spawn(run_rank, args=(args, rendezvous), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
