# LIM miss descriptor copy

Source migrated from vllm-ascend PR #20478, pinned at
`f382634f508563e6c403c2d7381f09598f461f45`.

The operator copies BF16 main KV rows using a LIM miss plan and the member's
actual Host-group block table. This build enables the kernel on `ascend910_93`.
Host pointers must be local NPU-addressable Mooncake VMM mappings; ordinary
CPU tensor pointers are insufficient.

The local Torch ABI replaces raw destination pointers with mutable
`hbm_kv_cache` and `hbm_k_rope` tensor arguments, passed directly through ACLNN
to the kernel. Graph optimization retains the copy and any replacement of
the destination tensors also changes the actual copy destinations.
`params` is an immutable int64 tensor with eight entries:
`[block_size, hot_tokens, miss_stride, kv_dim, rope_dim, host_blocks, pool_rows, plan_rows]`.
The caller validates these against the actual allocations before capture.
`active_geometry` contains `[active_rows, live_block_table_width]`; geometry,
counts, table and pool entries must become visible before the copy stream runs.

The kernel bounds physical Host blocks and pool rows, ignores padded negative
entries, and aligns the KV/rope UB boundary to 32 bytes. The CPU test executes
the production kernel with synchronous byte-copy primitives and sanitizers;
it does not validate NPU scheduling, CANN compilation or VMM visibility.

The NPU operator test exercises eager execution and dynamic graph replay with
device GM sources. The full PD serving test must additionally verify actual
Host VMM sources, stream dependencies and copy/compute overlap.
