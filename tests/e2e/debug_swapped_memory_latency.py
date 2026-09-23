"""Temporary diagnostic, run with torchrun; see docs below for interpretation.

This measures serialized iteration wall time (including launch/sync overhead),
not pure kernel time. Native NPU traces separate execution and waiting.
"""

import argparse
import json
import os
import resource
import socket
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu


MEMBERSHIP_COLUMNS = 16400
CONTROL_OFFSET = 16384
CONTROL_COLUMNS = 8


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--kv-width", type=int, default=576)
    parser.add_argument("--kv-slots", type=int, default=4096)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("swapped_memory_checks"))
    args = parser.parse_args()
    if min(args.rows, args.topk, args.tp_size, args.iterations, args.kv_width, args.kv_slots) <= 0:
        parser.error("sizes and iterations must be positive")
    if args.warmup < 0 or args.rows > args.kv_slots or args.topk > CONTROL_OFFSET:
        parser.error("require warmup >= 0, rows <= kv-slots, topk <= 16384")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch_npu.npu.set_device(local_rank)
    dist.init_process_group("hccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world % args.tp_size:
        raise ValueError("world size must be divisible by tp-size")
    group = None
    for begin in range(0, world, args.tp_size):
        candidate = dist.new_group(list(range(begin, begin + args.tp_size)), backend="hccl")
        if begin <= rank < begin + args.tp_size:
            group = candidate
    tp_rank = rank % args.tp_size
    tp_src = rank - tp_rank
    device = torch.device(f"npu:{local_rank}")
    args.output.mkdir(parents=True, exist_ok=True)
    plan_width = args.topk + CONTROL_COLUMNS
    plan_start = CONTROL_OFFSET - args.topk
    plan = torch.full((args.rows, plan_width), 7, dtype=torch.int16, device=device)
    kv_source = torch.ones((args.rows, args.kv_width), dtype=torch.bfloat16, device=device)
    kv_indices = torch.arange(args.rows, dtype=torch.int64, device=device)
    # This is a controlled competing Host write, not the actual Mooncake VMM pool.
    kv_host = torch_npu.empty_with_swapped_memory(
        (args.kv_slots, args.kv_width), dtype=torch.bfloat16, device=device
    )
    kv_host.zero_()
    main_stream = torch_npu.npu.Stream()
    writer_stream = torch_npu.npu.Stream()
    ready = torch_npu.npu.Event()
    torch_npu.npu.synchronize()
    results = []

    # Repeat in reverse order to expose ordering/temperature bias.
    cases = [(kind, writer, broadcast) for kind in ("hbm", "swapped")
             for writer in (False, True) for broadcast in (False, True)]
    for pass_id, ordered_cases in enumerate((cases, list(reversed(cases)))):
        for kind, writer, broadcast in ordered_cases:
            allocate = torch.empty if kind == "hbm" else torch_npu.empty_with_swapped_memory
            membership = allocate((args.rows, MEMBERSHIP_COLUMNS), dtype=torch.int16, device=device)
            plan.fill_(rank + 1)
            destination = membership[:, plan_start:plan_start + plan_width]
            torch_npu.npu.synchronize()
            faults_before = resource.getrusage(resource.RUSAGE_SELF)
            start = time.perf_counter_ns()
            membership.fill_(-1)
            torch_npu.npu.synchronize()
            first_touch_us = (time.perf_counter_ns() - start) / 1000
            faults_after = resource.getrusage(resource.RUSAGE_SELF)
            start = time.perf_counter_ns()
            membership.fill_(-1)
            torch_npu.npu.synchronize()
            second_touch_us = (time.perf_counter_ns() - start) / 1000

            def workload():
                ready.record(main_stream)
                if writer and tp_rank == 0:
                    with torch_npu.npu.stream(writer_stream):
                        writer_stream.wait_event(ready)
                        kv_host.index_copy_(0, kv_indices, kv_source)
                if broadcast:
                    dist.broadcast(plan, src=tp_src, group=group)
                destination.copy_(plan, non_blocking=True)
                if writer and tp_rank == 0:
                    main_stream.wait_stream(writer_stream)

            # Warm operators/collectives before capture, as in the service.
            with torch_npu.npu.stream(main_stream):
                for _ in range(args.warmup):
                    workload()
            torch_npu.npu.synchronize()
            graph = None
            if args.graph:
                graph = torch_npu.npu.NPUGraph()
                with torch_npu.npu.graph(graph, stream=main_stream):
                    workload()
                torch_npu.npu.synchronize()
                for _ in range(args.warmup):
                    graph.replay()
                torch_npu.npu.synchronize()

            label = f"pass{pass_id}_{kind}_writer{int(writer)}_broadcast{int(broadcast)}"
            profiler = nullcontext()
            if args.profile:
                profiler = torch_npu.profiler.profile(
                    activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                    experimental_config=torch_npu.profiler._ExperimentalConfig(
                        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                    ),
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                        str(args.output / label), worker_name=f"rank{rank}"
                    ),
                )
            dist.barrier()  # Align case start only; no per-iteration rank barrier.
            samples = []
            with profiler:
                measured_faults_before = resource.getrusage(resource.RUSAGE_SELF)
                with torch_npu.npu.stream(main_stream):
                    for _ in range(args.iterations):
                        start = time.perf_counter_ns()
                        if graph is None:
                            workload()
                        else:
                            graph.replay()
                        main_stream.synchronize()
                        samples.append((time.perf_counter_ns() - start) / 1000)
                measured_faults_after = resource.getrusage(resource.RUSAGE_SELF)
            # Functional check after timing; read back only through an ordinary NPU tensor.
            observed = torch.empty_like(plan)
            observed.copy_(destination)
            expected = torch.full(
                (args.rows, plan_width), tp_src + 1 if broadcast else rank + 1, dtype=torch.int16
            )
            torch.testing.assert_close(observed.cpu(), expected)
            torch.testing.assert_close(plan.cpu(), expected)
            if writer and tp_rank == 0:
                observed_kv = torch.empty_like(kv_source)
                observed_kv.copy_(kv_host[:args.rows])
                torch.testing.assert_close(observed_kv.cpu(), kv_source.cpu())
            ordered = sorted(samples)
            result = {
                "case": label,
                "first_touch_us": first_touch_us,
                "second_touch_us": second_touch_us,
                "first_touch_minor_faults": faults_after.ru_minflt - faults_before.ru_minflt,
                "first_touch_major_faults": faults_after.ru_majflt - faults_before.ru_majflt,
                "p50_us": ordered[int((len(ordered) - 1) * 0.50)],
                "p95_us": ordered[int((len(ordered) - 1) * 0.95)],
                "p99_us": ordered[int((len(ordered) - 1) * 0.99)],
                "max_us": max(samples),
                "minor_faults": measured_faults_after.ru_minflt - measured_faults_before.ru_minflt,
                "major_faults": measured_faults_after.ru_majflt - measured_faults_before.ru_majflt,
                "samples_us": samples,
            }
            results.append(result)
            # Write after each case so a later unsupported graph/op leaves useful evidence.
            document = {"rank": rank, "tp_rank": tp_rank, "dp_rank": rank // args.tp_size,
                        "hostname": socket.gethostname(), "torch": torch.__version__,
                        "torch_npu": torch_npu.__version__, "cann": getattr(torch.version, "cann", None),
                        "config": {**vars(args), "output": str(args.output)}, "results": results}
            (args.output / f"rank{rank}.json").write_text(json.dumps(document, indent=2))
            # Graph must die before the captured tensors are replaced in the next case.
            del graph
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
