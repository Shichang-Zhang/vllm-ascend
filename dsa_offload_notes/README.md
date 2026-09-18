# DSA Offload 图模式精度问题 —— 分析与实验文档索引

本目录收录 **DSA（DeepSeek Sparse Attention）稀疏 KV Offload 在 ACL graph 图模式下偶发精度问题** 的完整分析链路与实验手册。
当前分支：**主干分支**（标准 PD 分离服务侧），代码 = 修复 + 全量裁决探针。

## 一、本分支代码构成

| commit | 内容 |
| --- | --- |
| `a36b7b885` | 修复：把图模式 Host-KV 写回的 join 提到读者（fused sparse attention）之前 |
| `7fb742f52` | 修复（二次修正，纯代码）：`current_kv_writeback_on_side_stream` 每次调用复位；MemFabric fork 分支同样置位该标记 |
| `011a76f59` | 探针：`VLLM_ASCEND_SFA_ADJ_PROBE` 控制的全套裁决探针与日志（默认关闭，不影响生产路径） |

> 探针通过环境变量 `VLLM_ASCEND_SFA_ADJ_PROBE=1` 打开，输出在 stderr，前缀 `[SFA_ADJ_PROBE]`。
> 关闭时只多一次 relaxed 原子读，生产行为与修改前一致。

## 二、文档阅读顺序

| # | 文档 | 作用 |
| --- | --- | --- |
| 1 | `260916_DSA-offload方案PR定位与1P1D部署解析.md` | 方案全貌：上游 4 个 PR 的定位、1P1D 部署拓扑、依赖层次 |
| 2 | `260916_DSA精度问题定位分析.md` | 精度问题差分分析（fused+Mooncake vs fused+MemFabric），后端差分候选表 |
| 3 | `260917_短序列图模式精度问题定位.md` | 短序列图模式偶发精度问题的**原始定位**（decode-only 分支静态分析） |
| 4 | `260917_短序列图模式精度问题定位_复核与新增候选.md` | **复核报告**：修正原文档的结论、给出 P0 根因与新增候选、实验与探针方案 |
| 5 | `260917_图模式HostKV时序问题_代码修正说明.md` | 主干分支修复（`a36b7b885`）的逐项说明 |
| 6 | `260917_修复复核与二次修正.md` | 对两份修复代码的复核与二次修正（`7fb742f52` 的来源） |
| 7 | `260917_三轮复核报告与repo2修正.md` | 三轮复核：既有分析结论 + 修复代码的最终核对 |
| 8 | `260917_decode-only分支代码审查.md` | decode-only 分支审查：运行态代码 vs 打桩代码，旧探针的覆盖盲区 |
| 9 | `260917_decode-only分支_代码修正说明.md` | decode 分支修复（`31d2e6edc` + `2e9f6194d`）说明 |
| 10 | `260917_裁决实验桩说明.md` | 裁决实验桩（repo3）设计说明：四个实验、四层假设 |
| 11 | `踩坑总结.md` | 已观察现象与踩坑记录（TP0 D2H 后同步、广播等） |
| 12 | `260918_裁决实验验证手册_主干分支.md` | **实验手册（本分支）**：验证哪些问题、怎么跑、日志怎么读、结论怎么下 |

## 三、核心结论速览

* 图模式下 decode 每步的 Host-KV 写回被 fork 到 `current_kv_save_stream`，而读者（fused sparse attention 的 miss-onload、以及 TP ready 广播）在 compute 流上；**唯一的 join 边位置在算子之后且仅 TP0 执行**，因此同一步的读-写之间缺少 happens-before。
* "同一步读 Host"这条依赖真实存在的前提是：planner 会驱逐本步窗口 token、只有 own token 拿到正 plan，其余窗口位置只能从 Host 池读；且只有 MTP（`q_len ≥ 2`）才会出现行间引用。这一前提由 `planner_stats` 探针独立验证。
* 探针非零 ≠ 业务一定出错（单 key 错误会被 softmax 稀释），需与业务指标联合判定。

## 四、未入库的文档

以下两份有意未纳入本目录：

* `task.md` —— 实验环境准备步骤，含内网机器/共享盘路径，不入公开仓库；
* `问题清单.md` —— 空文件。

## 五、相关分支

| 分支 | 用途 |
| --- | --- |
| `dsa_offload_rebase_pr15642_0911_fix_probe`（本分支） | 主干分支：修复 + 全量裁决探针 |
| `dsa_offload_rebase_pr15642_0911_decode_fix_probe` | decode-only 服务分支：修复 + 探针移植（含打桩提交去留说明） |
