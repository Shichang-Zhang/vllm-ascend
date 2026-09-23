# Temporary swapped-memory latency checks

These diagnostics target the external LRU planner and Host-backed membership
storage. They add no per-layer synchronization to the service. Remove the debug
changes after diagnosis.

## Actual planner callback timing

Use the existing service configuration:

```bash
--profiler-config '{"profiler":"torch","torch_profiler_dir":"./vllm_profile","torch_profiler_with_stack":true}'
```

Start and stop profiling through the existing `/start_profile` and
`/stop_profile` endpoints. Each DP replica's TP0 exports an additional
`*_lru_callback.json` file under the profiling directory. Other TP ranks do not
compute plans and therefore do not emit a callback file.

The callback records CPU timestamps around `lru_resident_compact_impl` during
actual execution, including graph replay. It does not time the Python enqueue
call. The buffer is allocated before capture and enabled when profiling starts;
no MSTX markers need to have been captured at server startup. The profiler's MSTX
setting stays unchanged.

Each captured callback retains its first 256 executions per profiling window.
`calls` includes executions beyond capacity; `dropped` reports omitted samples.
Separate graph variants have separate buffers. The file contains graph ID,
layer ID, request count, TopK, begin/end timestamps, and p50/p99/max duration in
microseconds. Array position is a callback-local invocation number, **not a
cross-DP step ID**. Layers that reuse a plan have no new planner execution.

The trace applies to graphs already captured when profiling starts. Eager
planner calls and graphs captured later in the window are not included. Use an
established graph workload for this check.

There is a device synchronization at trace start and after profiler stop to
safely reset/read the buffers. These are profile-boundary operations, outside
the forward path. Do not call the trace reset/export extension methods while
callbacks are running. The recorded window can include work drained at profile
stop; discard boundary samples when comparing steady state. The disabled
callback path still pays a small atomic-flag check.

`start_clock_anchor` and `stop_clock_anchor` bracket a C++ steady-clock sample
with Unix-clock timestamps. These allow approximate correlation with host
traces, with uncertainty from the bracket width and clock drift. Do not directly
compare steady-clock timestamps from different hosts, or mix raw NPU clock
values with CPU timestamps. Use profiler time alignment for device timelines.
CPU minor/major fault counts cover the whole process/window (including profiler
work); they do not identify NPU translation faults or attribute faults to a
particular allocation.

Use native NPU timeline tasks for D2H, H2D, HCCL broadcasts, membership copy,
KV writeback, and fused attention. This patch does not add device phase markers.
Correlate by stream, graph/layer order, and the corresponding EP execution.
A long broadcast may be waiting for a late participant. A large gap between
D2H completion and callback entry is different from a long callback duration.

## Controlled memory experiment

Run outside the serving process, using the same torch_npu/CANN/driver build.
For a single device:

```bash
torchrun --standalone --nproc-per-node=1 tests/e2e/debug_swapped_memory_latency.py \
  --tp-size 1 --rows 32 --topk 2048 --output ./swapped_checks_eager
```

For two emulated DP replicas, each with TP=4:

```bash
torchrun --standalone --nproc-per-node=8 tests/e2e/debug_swapped_memory_latency.py \
  --tp-size 4 --rows 32 --topk 2048 --graph --output ./swapped_checks_graph
```

Add `--profile` in a separate run to collect native per-case NPU traces. Profiling
itself can change the latency distribution, so keep the unprofiled result as a
baseline. Use `--warmup`, `--iterations`, `--kv-width`, and `--kv-slots` to match
representative workloads. Repeat with representative token counts via `--rows`.

Each rank writes `rank<N>.json` with versions, configuration, raw iteration
samples, percentiles, and CPU fault counters. The script compares:

- Ordinary HBM and swapped membership, with the same strided plan-copy geometry.
- Membership copy alone and a preceding TP-local plan broadcast.
- With and without a competing Host write on a second stream on each TP0.
- Initial and repeated full-allocation fills, followed by warmed measurements.
- Both forward and reversed case order to expose ordering bias.

The main measurement is **serialized iteration wall time**, including launch
and synchronization overhead and, when enabled, waiting for the competing
write. It is not the isolated membership-copy kernel duration. Use the native
trace to separate those phases. There is a barrier at case start, not at each
iteration. Concurrent broadcasts intentionally allow participant waiting.

The competing write uses another swapped allocation; it is not the production
Mooncake VMM pool. The test covers copy/write contention, not the full fused
attention access pattern or CPU LRU algorithm. It checks copied data after
timing. If graph capture or an operation is unsupported by the installed build,
the script fails rather than silently falling back to eager execution.

Interpretation:

- A slow first fill alone can include lazy runtime initialization; it is not
  proof of paging. Compare the repeated fill and warmed runs.
- Increased tails only with concurrent writes suggest shared-resource contention.
- Increased CPU fault counts alone do not prove faults in swapped memory.
- Stable copies but slow real planner callbacks point toward CPU scheduling,
  OpenMP contention, or LRU workload differences.
- Stable planner/copies but delayed attention or final writeback waits require
  inspecting the actual VMM Host-KV and fused-attention paths.
