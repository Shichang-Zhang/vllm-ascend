# Mooncake delayed D2H validation

This adapts [PR #20289](https://github.com/vllm-project/vllm-ascend/pull/20289)
to V1 Runner with Prefill DCP and MooncakeConnectorV1 Decode DSA offload.
Delayed D2H is disabled by default. Enable `delayed_d2h` with
`fused_op_type=fused_copy_sfa`, `host_backend=mooncake`,
`keep_device_kv_cache=false`, and connector extra config `dsa_pd_offload=true`.
Decode requires PCP1/DCP1; MTP3 requires a hot budget of at least 8192.

Compare the same model and workload with `a_overlap.json` (existing overlap),
`b_copy_sfa.json` (Copy-SFA baseline), and `c_delayed.json` (delayed writes).
Pass each file as the complete `--additional-config` value. Keep the Prefill
MultiConnector setup and the Decode MooncakeConnectorV1 setup unchanged.
Separate the baseline switch and its larger HBM budget from delayed-D2H gains.

CPU regression tests:

```bash
VLLM_VERSION=0.26.0 .venv/bin/python -m pytest -q \
  $(cat examples/disaggregated_prefill_v1/mooncake_dsa_migration/local_tests.txt)
```

On NPU hardware, check eager and graph execution, MTP acceptance, long-context
accuracy, short/dense-to-ring transitions, APC reuse, request cancellation,
shutdown, and TP completion. Measure latency, throughput, HBM/Host memory,
and the per-step completion collective. Record all software and model revisions.
CPU tests do not establish NPU correctness or performance. Memory-pressure
preemption/resume is unsupported while a delayed-D2H lifetime owns blocks;
size the workload accordingly. Roll back by disabling `delayed_d2h`.
