# Serialized SFA phase timing

This temporary diagnostic uses device completion waits, not TP-group barriers.
Enable it on **every Decode worker**, before server startup:

```bash
export VLLM_ASCEND_DEBUG_SFA_SYNC_TIMING=1
```

It defaults to disabled. No profiler endpoints or trace viewer are required.
Collect worker logs containing `SFA_SYNC_TIMING`. Each line is JSON after that
prefix, with process/rank identity, phase, call count, sample count, and mean/max
microseconds. The first observation is logged immediately; subsequent reports
cover non-overlapping windows of 32 observations per phase and row count.
Discard initialization/warmup observations when comparing steady state.
Incomplete final windows are not exported.

## First run: keep the production graph and MTP settings

For FULL_DECODE_ONLY target decoding with eager MTP, the output includes:

- `graph_replay`: completed FULL graph execution, including its internal
  transfers, CPU callbacks, collectives, and peer waits. This is the model
  graph interval, not the entire scheduler iteration or TPOT.
- Eager SFA phases: `metadata_prepare`, `topk_d2h`, `metadata_d2h`,
  `cpu_planner`, `plan_h2d`, `metadata_broadcast`, `status_d2h`,
  `plan_broadcast`, `membership_copy`, `current_kv_injection`,
  and `fused_attention`. Layer names and row counts identify the work.
- Eager TP0 KV writeback: `nonzero` and `index_copy`, aggregated across layers
  with the same row count. Empty/invalid batches have no `index_copy` sample.

These Python phase probes are skipped during graph capture and do not execute
inside graph replay. Thus steady-state target layers appear in `graph_replay`,
while eager draft layers have individual phase measurements. Warmup and
uncaptured target execution can also produce individual phase measurements.

## Second run: target eager for phase isolation

For detailed target-layer measurements, repeat the diagnostic with target
`--enforce-eager`, keeping MTP and all other settings unchanged. Do not interpret
this as the target graph's phase timing: eager dispatch, callbacks, and stream
overlap differ. Compare both branches under the same diagnostic settings.
Keep the existing callback trace/native profiler for measurements *inside*
the real captured target graph.

## Interpretation

Each probe performs `device.synchronize()`, runs the phase, then performs
`device.synchronize()` again. There is no synchronization call during capture.

| Field | Meaning |
| --- | --- |
| `prior_work_us` | Time draining previously queued device work before this phase |
| `host_call_us` | Host elapsed time in the operation, including any blocking API waits |
| `completion_wait_us` | Remaining device completion wait after the operation returns |
| `phase_us` | `host_call_us + completion_wait_us` |

A slow TopK copy with a large `prior_work_us` and small `phase_us` was mainly
waiting for earlier work. A slow `cpu_planner` with a small prior drain points
toward CPU planning or CPU scheduling. Compare `membership_copy` on TP0 and TP1
to separate the local publication cost from waiting for the producer.

Broadcast timings **still include peer arrival delays**. TP1 can arrive while
TP0 is copying or planning. Do not add TP1's broadcast duration to the TP0 work
it waits for. An extra TP barrier would move that waiting into the barrier; it
would not prove that communication was the original bottleneck. These probes
retain the existing collective order and do not add cross-rank collectives.

Every device synchronization also includes its own runtime overhead, and logging
can perturb the following work or peer arrival. The probes serialize execution
and drain unrelated streams on the same device. They are attribution experiments,
not pure kernel timings or normal-service performance measurements. In particular,
do not compare their TPOT with an uninstrumented baseline. With MTP, compare
iteration duration and accepted/emitted tokens as well as TPOT.

Unset the variable and restart all workers to restore normal execution. Validate
on the deployment's CANN/torch_npu stack; CPU tests cannot validate HCCL or graph
runtime behavior.
