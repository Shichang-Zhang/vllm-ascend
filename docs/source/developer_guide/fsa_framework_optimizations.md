# FSA framework optimizations

These configurable changes reduce framework work around fused sparse attention.
They preserve the external LRU planner, cross-layer plan reuse, MemFabric
`sparse_copy`, request metadata construction and the FSA C++ kernel.

## Configuration

All variables below are non-sensitive boolean settings accepting `0` or `1`.
Set them before worker initialization and graph capture.

| Variable | Default | Behavior |
| --- | --- | --- |
| `VLLM_ASCEND_FSA_ASYNC_PLAN` | `1` | Enqueue dynamic decode D2H and the native planner callback on the planner stream. Graph capture already uses this stream. |
| `VLLM_ASCEND_FSA_VALIDATE_DEVICE_METADATA` | `0` | Validate device values in non-captured attention metadata. `0` skips only these optional value checks. |
| `VLLM_ASCEND_FSA_REUSE_WRITEBACK_LAYOUT` | `1` | Reuse slot validity and destination offsets across layers in one graph capture. |
| `VLLM_ASCEND_FSA_FUSED_WRITEBACK_DESCRIPTORS` | `1` | Generate runtime current-KV copy descriptors with a Triton kernel. Supersedes layout reuse when initialized. |
| `VLLM_ASCEND_FSA_PAIRED_CURRENT_SCATTER` | `1` | Update selection K and Rope with one Triton kernel, with native scatter fallback. |

The historically qualified performance configuration is:

```bash
export VLLM_ASCEND_FSA_ASYNC_PLAN=1
export VLLM_ASCEND_FSA_VALIDATE_DEVICE_METADATA=0
export VLLM_ASCEND_FSA_REUSE_WRITEBACK_LAYOUT=0
export VLLM_ASCEND_FSA_FUSED_WRITEBACK_DESCRIPTORS=1
export VLLM_ASCEND_FSA_PAIRED_CURRENT_SCATTER=1
```

The patch enables async planning, layout reuse, descriptor fusion and paired
scatter by default, and disables optional device-value checks. Explicit process
environment values still override these defaults. Descriptor fusion takes
precedence over layout reuse once initialized; the two do not execute together.
The historical configuration above used layout reuse off because descriptor
fusion supersedes it. These defaults do not select the fused model path itself.

Disabling device-value validation assumes valid
request mappings, cumulative query lengths and positive per-token KV lengths.
Shape checks, MTP flattening, invalidation and the synchronous planner failure
status check remain. Native asynchronous callback failures are not given the
same Python exception propagation guarantee as the synchronous planner.

## Ordering and compatibility

Dynamic planner inputs wait for their producing compute stream. D2H inputs,
the native CPU callback, slot publication and TP broadcast are ordered on the
planner stream. Source tensors record that stream for allocator lifetime.
Before selection consumption, compute waits for the planner. Captured
current-KV writeback also completes before consumption and is published across
TP ranks, including layers that reuse an owner's plan. The final graph copy
stream join is retained. CPU copies into separate mapped plan storage wait for
the asynchronous producer.

Writeback descriptors retain the MemFabric copy and its result check. They
support BF16 decode KV with positive rows and token capacity below `2**31`;
other cases use the original descriptor operations. Paired scatter requires
contiguous BF16 KV, int32 slots, matching dimensions and devices, and valid
distinct planner-owned physical slots. Unsupported layouts use native scatter.
Both kernels warm up during cache registration before graph capture.

Layout reuse is scoped to a capture token and additionally keyed by slot tensor
identity, capacity, token sizes, device and stream. Each layer keeps its own
source and destination bases. Replaying the graph recomputes offsets from current
slot values. A new capture cannot reuse old captured tensors.

This port preserves the new baseline's memory-budget checks, split metadata
validator, super-kernel support and LoRA-specific graph parameter storage.
Rejected packed planner input and vector descriptor variants, and experimental
dynamic writeback scheduling, are not restored.

## Evidence and limits

Source snapshots were recovered from the 2026-09-10 qualification in
`10.246.63.33`, container `fuse_offload_v2`, under
`/home/j00628475/d2h_async_20260908`.

- Manager snapshot SHA256: `3599a477595ef1f34990ab340dc9492bdbb0aa3a5ef188712d7cc19d0806fc62`.
- Descriptor helper SHA256: `6ba6bafb13f7f60236c4536803aafbec5d227baa9fdce3f70da521a95b374445`.
- Paired scatter helper SHA256: `78b8f648dee6923a9c729fd80c1630bf408925a94d2e86a073d8ae3a96b5a894`.

Historical mixed UT used 78 Graph attention layers plus dynamic MTP, one request
group, TopK 2048, selection capacity 4096 and synthetic full-KV backing derived
from approximately 6K dumps. Descriptor fusion improved nine cases by
2.99%-7.40%; paired scatter subsequently improved nine cases by 5.81%-9.05%.
The matched rows6 trace cycle was 26.5960 ms synchronous, 18.5083 ms with async
planner/descriptors/checks disabled, and 17.2980 ms with paired scatter.
The cumulative 34.96% includes disabling optional metadata checks. It is not
production TPOT, and percentages from different experiments cannot be added.

Historical qualification covered changing inputs, pure-SFA comparisons, helper
fallbacks, mixed MTP and TP2 delayed-writeback checks. The test runtime used
torch_npu post2; the target repository requires post4. New-baseline full-engine,
multi-request scheduler and error-propagation validation remain separate gates.

Focused repository tests:

```bash
pytest tests/ut/kv_offload/test_sparse_kv_offload_framework.py \
  tests/ut/kv_offload/test_sparse_kv_offload_external_plan.py \
  tests/ut/attention/test_sfa_kv_offload.py
```

## Replacement-checkout validation (2026-09-10)

The port targets `releases/v0.26.0rc` at
`47fc59895cd3dceff38255dea73cdb2c9ddc6413` plus the uncommitted framework changes.
The isolated remote run used copies of the ported manager, attention methods,
Graph wrapper and helpers, and freshly compiled this checkout's native C++
planner. Existing FSA OPP binaries and post2 runtime were reused. The wrapper
fixture sets super-kernel optimization off; full-engine imports and post4
deployment are not covered by this method-extraction harness.

Nine cases (main rows 1/4/6, fixed/reshuffle/trace) passed eight precision cycles
per version with bitwise equality against the qualified snapshot. Mean round
P50 cycle differences ranged from -1.06% to +0.72%. TP2 Graph and dynamic tests
with 128 matmuls delaying TP0 writeback passed eight steps per rank, zero error.
Local method contracts passed 49 cases; scoped Ruff, syntax and diff checks
passed. Standard local pytest collection requires the unavailable `vllm`
dependency; these results do not represent a full repository pytest run.

Same-run rows6 trace ablation, 78 Graph layers plus five dynamic drafts, two
alternating rounds of 20 samples (mean round P50):

| Configuration | Component cycle (ms) | Reduction from preceding row |
| --- | ---: | ---: |
| Synchronous planner, device checks on | 26.8303 | - |
| Async planner, optional device checks off | 20.1124 | 25.04% |
| Plus capture-scoped layout reuse | 19.3633 | 3.72% |
| Descriptor fusion replacing layout reuse | 18.4539 | 4.70% |
| Plus paired scatter | 17.2172 | 6.70% |

The cumulative component-cycle reduction is 35.83%, with all five variants
passing paired precision checks. Async scheduling and optional validation
removal are combined in the first increment. This is not TPOT or evidence of
multi-request scheduler stability. Raw outputs and source hashes are retained
under `/home/j00628475/fsa_framework_restore_20260910` on host 10.246.63.33.
