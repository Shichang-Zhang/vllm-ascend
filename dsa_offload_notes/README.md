# DSA Offload 图模式精度问题 —— 分析与实验文档索引

本目录收录 **DSA（DeepSeek Sparse Attention）稀疏 KV Offload 在 ACL graph 图模式下偶发精度问题** 的完整分析链路与实验手册。
当前分支：**decode-only 服务分支**（单独拉起 decode 服务），代码 = 修复 + 探针移植。

## 一、本分支代码构成

| commit | 内容 |
| --- | --- |
| `c806ec153` | 单独拉起 Mooncake decode 卸载（配置放宽、MTP descriptor 刷新、prefill KV staging） |
| `71422442b` | **打桩**：图/eager Mooncake 路径 trace（含无开关的 Host 地址校验，见实验手册第 2 节） |
| `3fa38c72b` | 单测改用 g++（与运行时无关） |
| `31d2e6edc` | 让 Host-KV 可见性探针在**图模式**下可用（本分支图模式计数的来源） |
| `2e9f6194d` | 修复：把图模式 Host-KV 写回的 join 提到读者（fused sparse attention）之前 |

> 探针统一由 `VLLM_ASCEND_SFA_INDEX_COPY_PROBE=1` 打开（默认关闭，所有诊断短路）。
> 另外两个开关：`VLLM_ASCEND_SFA_PROBE_FAIL_FAST`（eager 比对失败是否抛异常，默认 1）、`VLLM_ASCEND_SFA_READY_BCAST`（是否执行 TP 放行广播，默认 1）。
> 日志前缀：`[SFA_INDEX_COPY_PROBE]`、`[SFA_CURRENT_HOST_READ]`、`[SFA_GRAPH_TRACE]`、`[SFA_READY_BCAST]`、`[SFA_PLAN_TRACE]`。

## 二、文档阅读顺序

| # | 文档 | 作用 |
| --- | --- | --- |
| 1 | `260916_DSA-offload方案PR定位与1P1D部署解析.md` | 方案全貌：上游 4 个 PR 的定位、1P1D 部署拓扑、依赖层次 |
| 2 | `260916_DSA精度问题定位分析.md` | 精度问题差分分析（fused+Mooncake vs fused+MemFabric），后端差分候选表 |
| 3 | `260917_短序列图模式精度问题定位.md` | 短序列图模式偶发精度问题的**原始定位**（即以本分支为对象的静态分析） |
| 4 | `260917_短序列图模式精度问题定位_复核与新增候选.md` | **复核报告**：修正原文档的结论、给出 P0 根因与新增候选、实验与探针方案 |
| 5 | `260917_decode-only分支代码审查.md` | **本分支专属**：运行态代码 vs 打桩代码、旧探针的覆盖盲区 |
| 6 | `260917_decode-only分支_代码修正说明.md` | **本分支专属**：修复 `31d2e6edc` + `2e9f6194d` 说明 |
| 7 | `260917_图模式HostKV时序问题_代码修正说明.md` | 主干分支修复（`a36b7b885`）说明，用于对照 |
| 8 | `260917_修复复核与二次修正.md` | 对两份修复代码的复核与二次修正 |
| 9 | `260917_三轮复核报告与repo2修正.md` | 三轮复核：既有分析结论 + 修复代码的最终核对 |
| 10 | `260917_裁决实验桩说明.md` | 裁决实验桩设计说明：四个实验、四层假设 |
| 11 | `踩坑总结.md` | 已观察现象与踩坑记录（TP0 D2H 后同步、广播等） |
| 12 | `260918_裁决实验验证手册_decode-only分支.md` | **实验手册（本分支）**：验证哪些问题、怎么跑、日志怎么读、结论怎么下 |

## 三、核心结论速览

* 图模式下 decode 每步的 Host-KV 写回被 fork 到 `current_kv_save_stream`，读者在 compute 流上；唯一 join 边在算子之后且仅 TP0 执行 ⇒ 同一步读-写之间缺少 happens-before。
* eager 探针（`_probe_mooncake_index_copy` 逐元素比对）只在 `not capturing` 时挂载，因此它的 PASS **不能外推**到图模式；这正是新增 `[SFA_GRAPH_TRACE]` 图模式计数的原因。
* 复用 layer-0 plan 的层**不发出任何集合通信**（`[SFA_PLAN_TRACE] ... metadata_broadcast_will_run=False`），这些层上 peer 只能依赖更早的同步点，属于最高风险层。
* 本分支的可见性探针只在 Mooncake 后端生效（MemFabric 的 Host 视图是 CPU/GVA，设备比较会直接返回）。

## 四、未入库的文档

以下两份有意未纳入本目录：

* `task.md` —— 实验环境准备步骤，含内网机器/共享盘路径，不入公开仓库；
* `问题清单.md` —— 空文件。

## 五、相关分支

| 分支 | 用途 |
| --- | --- |
| `dsa_offload_rebase_pr15642_0911_decode_fix_probe`（本分支） | decode-only 服务：修复 + 探针移植 |
| `dsa_offload_rebase_pr15642_0911_fix_probe` | 主干分支：修复 + 全量裁决探针 |
