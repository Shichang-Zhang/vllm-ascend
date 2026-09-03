# Unsolved Problems

## Mooncake membership 发布的性能开销

Decode external planner 在每个需要重新计算 Top-K 的 owner layer 上生成
membership plan。融合算子依赖该 plan，因此 plan 必须发布完成后融合算子才能启动，
这部分延迟无法被当前融合算子掩盖。

MemFabric 的 membership 同时支持 CPU planner 写入和 NPU 算子访问，planner 可以
直接写最终 membership，不需要额外发布。Mooncake shared/swapped memory 不能由
CPU planner 直接写入，因此当前兼容方案需要：

```text
CPU pinned membership
    -> ordinary NPU staging
    -> Mooncake shared/swapped membership
```

Mooncake 因而比 MemFabric 多两次串行小拷贝。虽然单次 compact plan 只有约
4 KiB/token，但该过程会在一个 decode step 的多个 owner layer 上重复。性能风险
不仅来自累计传输量，也来自多次小拷贝的下发和同步延迟，可能直接增加 TPOT。

一个长期优化方向是让 Mooncake Shared Segment 为同一段物理页同时提供 CPU
和 NPU tensor 视图：CPU planner 直接写 CPU 视图，融合算子通过 NPU 视图读取，
从而消除 CPU staging -> NPU staging -> membership 的两次串行拷贝。该方案需要
验证 HostRegister/VMM 模式下的地址偏移、CPU/NPU 内存可见性、跨 TP 共享和 segment
生命周期，不能仅按两个 tensor 的 data_ptr 是否相同来判断别名关系。

该问题尚未解决。需要分别测量 CPU planner、两段 copy、stream wait 和端到端
TPOT，再判断是否需要双缓冲提前规划或 Shared Segment CPU/NPU 双视图来消除拷贝。

## Blockwise Indexer 传输项未合并

当前 DSA Blockwise 路径按 layer、position 和 block 展开 Mooncake transfer
entry。本次 22 个 block 的请求在每个 Decode TP 上生成了 924 个 Indexer D2D
entries，并且该开销会随 TP 数量重复。

`v0.25.1rc1` 的 MooncakeConnectorV1 使用
`group_concurrent_contiguous()` 合并源端和目的端同时字节连续的 block，并通过
`split_if_not_byte_contiguous()` 处理不能连续合并的范围。当前 DSA transfer-list
构造没有复用该优化，可能增加地址列表构造、Transfer Engine 下发和小传输调度开销，
影响 PD 传输延迟。

后续需要将连续区间合并移植到 Indexer D2D 和 Main D2RH 路径，并确认
block stride、compress ratio/token scale 以及 Host DRAM 布局下的合并正确性，
对比优化前后的 entry 数量和端到端传输耗时。

## Mooncake 图模式 index descriptor 的 CPU callback 开销

Mooncake 当前 token 写回使用固定长度 index_copy。图模式下，每次 replay
需要先将 slot_mapping 拷到 CPU，由 host callback 刷新固定长度的
src_idx/dst_idx，再将 descriptor 拷回 NPU。真正的 KV index_copy 已放到
current_kv_save_stream 与融合算子计算重叠，但 callback 及 descriptor 的
Device-to-Host/Host-to-Device 仍在主 stream，属于无法掩盖的串行延迟。

当前已将 descriptor 生成从每层一次降低为每个 decode step 一次，各层复用同一组
索引；每层只保留自身 KV 的 index_copy。实测将 host callback 本身提交到 side
stream 后，descriptor 不会在 graph replay 时刷新，因此不能直接把整条链路移到
side stream。

后续可评估在 NPU 侧生成固定 descriptor，或为无效 slot 预留 dummy/sink slot，
从而消除 CPU callback 和双向小拷贝，并通过 TPOT profiling 判断实际收益。
