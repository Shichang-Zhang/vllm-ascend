# Prefill DCP 与 Decode DSA offload 迁移计划

将官方 PR #20289 的延迟批量 D2H 和 PR #20478 的 shared-indexer-only KV 驻留及 side-stream onboarding，适配到当前 Prefill DCP → Mooncake → Decode DSA offload 链路。保留现有 V1 Runner、AscendVMM、量化 LIM 和 full-reuse 行为。

本计划采用按功能适配回移。两个功能独立开关、独立验收，最后验证组合；不直接 cherry-pick 整个 PR，也不把静态审查结论当作硬件验证结果。

实施分工更新：用户指定以 `/home/sigh/proj/mayi/sig-ai-inference/scripts` 为部署配置参考，并确认本机无 NPU 环境，真实机器测试由用户后续执行。本次本地交付覆盖迁移代码、CPU 测试、逐步骤子代理审查及硬件验证说明。下文硬件退出条件仍是上线验收要求，但不阻塞本地后续阶段；不得将其标记为已通过。实际配置核对和进度见 [迁移进度](MIGRATION_PROGRESS_PR_20289_20478.md)。

实施交付：阶段 B/C/D/E 已完成本地实现及逐步子代理审查；硬件验收待用户执行。七组完整 Decode 配置、升级/回退顺序和真机操作见 [Mooncake 验证说明](examples/disaggregated_prefill_v1/mooncake_dsa_migration/README.md)。当前实现采用 DSA v2 分组协议，启用 shared-only 前须先升级所有 Prefill；LIM copy 算子采用本地10参数 ABI，扩展和算子库须一起重建，目前仅加入 A3 构建。MTP3 的 Copy-SFA 基线要求 hot budget 至少8192。

## 1. 基线与参考版本

| 项目 | 固定参考 |
| --- | --- |
| 当前分支 | `dsa_offload_rebase_0.26_kernel_v2` |
| 当前 HEAD | `4cac524d9d19fb71fee4940c782884ed0773782f` |
| [PR #20289](https://github.com/vllm-project/vllm-ascend/pull/20289) | `f1c22e86416b07dd18a49acc853b6c6ea4390bd8` |
| [PR #20478](https://github.com/vllm-project/vllm-ascend/pull/20478) | `f382634f508563e6c403c2d7381f09598f461f45` |

实施时固定上述输入版本。如果选择更新后的 PR revision，需要重新审查差异并更新验证范围，不自动把后续改动加入迁移。

## 2. 支持范围和实施前置条件

用户已确认 Prefill 使用 DCP、Decode 使用 DSA offload，并明确目标 connector 是 MooncakeConnectorV1。指定脚本确认 P 端 DP4/TP8/DCP8、D 端 DP16/TP2/PCP1/DCP1、MTP3、当前 fused-overlap。实际启动默认值（async/APC）及模型 revision 仍需在用户后续硬件验证中确认。迁移目标如下：

```text
Prefill：DCP > 1，沿用现有 KV 分片和 replicated-indexer 布局
   ↓ Mooncake P→D 传输
Decode：PCP = 1、DCP = 1，可使用 TP/DP
        V1 Runner + fused Copy-SFA + DSA offload
```

| 项目 | 必须记录 | 对迁移的影响 |
| --- | --- | --- |
| 模型 | 模型 revision、indexer-owner/shared-member 分布、MTP 层 | 判断 #20478 的适用性 |
| 并行配置 | P/D 两端 TP、DP、PCP、DCP、PP | 目标要求 D 端无 CP，保留现有 PP 限制 |
| Offload 模式 | `fused_op_type`、`use_fused_overlap`、host backend | 两个优化以 fused Copy-SFA 为目标 |
| Spec decode | MTP token 数、async scheduling、reuse 模式 | 决定准确长度来源和最终 KV writer |
| Graph 模式 | eager、piecewise/full graph、capture sizes | 确定事件和 metadata 更新方式 |
| 软件与硬件 | vLLM、CANN、torch-npu、Mooncake、算子版本、NPU 型号 | 固定可复现构建和测试环境 |
| 服务行为 | prefix cache、取消、preemption、恢复、chunked prefill | 决定生命周期和失败恢复支持边界 |
| 容量配置 | block size、hot budget、最大序列长度、并发、Host/HBM 预算 | 检查 tail 容量和混合驻留预算 |

如果当前是 `use_fused_overlap`，先单独验证切换到 fused Copy-SFA 的正确性与性能，不把路径切换收益计入两个 PR。

本次不扩展 Decode CP、Runner V2、DSpark 或新的主 KV 量化格式。保留已有不兼容配置检查和受支持的量化 indexer。分支名中的 `kernel_v2` 不代表 Runner V2 已支持 sparse offload。

Prefill 侧保持脚本的 MultiConnector（AscendStoreConnector + MooncakeConnectorV1）；Decode 侧为直接 MooncakeConnectorV1，`dsa_pd_offload=true`、`host_backend=mooncake`。仅在传输协议确实缺少信息时增加兼容字段。MemFabric/SfaRemoteD2H 只保持新开关关闭时的旧路径回归，不是本次新功能的迁移目标，不替换用户的 MooncakeConnectorV1。

## 3. 已确认的迁移风险

| 风险 | 代码依据 | 处理要求 |
| --- | --- | --- |
| Async MTP 的 CPU computed-token 数可能乐观 | [V1 Runner](vllm_ascend/worker/model_runner_v1.py) 的 `update_num_computed_tokens_for_batch_change` 和 `_correct_optimistic_seq_lens_cpu` | 使用 rejection 修正后的 frontier，验证 event 和请求重排 |
| 短序列写 dense 区，而 #20289 planner 固定从 tail 读 | [attention metadata](vllm_ascend/attention/sfa_kv_offload.py) 的 `copy_sfa_device_slots`；#20289 的 `plan_d2h_requests` | 按 dense/ring 布局生成 source descriptor |
| Mooncake 接收记录不覆盖整个 decode 生命周期 | [Mooncake scheduler](vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py) 的 `update_connector_output`、`request_finished` | 独立管理 decode 所有权和延迟释放 |
| Prefix-cache 发布早于 Host D2H 完成 | [AsyncScheduler patch](vllm_ascend/patch/platform/patch_async_scheduler.py) 的 `cache_blocks` 调用依据 computed tokens | 增加完成 frontier 约束的发布或命中等待协议 |
| 协议只表达一组 main Host IDs 和一组 indexer IDs | [DsaStepRequest](vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_dsa_metadata.py)、[group 推断](vllm_ascend/distributed/kv_transfer/kv_p2p/sfa_pd_rd2h/protocol.py) | 显式建立 layer/group/block-ID 映射 |
| 私有 owner HBM 不能沿用共享 Host 的 TP 分工 | [transfer builder](vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_dsa_transfer.py) 的 main writer filter | 保留 source CP 选择，独立选择 destination 写入策略 |
| Owner table 不一定能寻址 member Host KV | #20478 的 `_launch_lim_group_ksc` 使用 owner metadata 和 member bases | 显式使用 member 的 Host-group table，或证明编号一致 |

其中短序列源地址和跨组 table 问题来自静态代码对比，实施时需用数据内容测试确认，不只测试 descriptor 数量。#20478 已有 per-group geometry 修复，应保留；这并不证明跨组 block-ID 寻址已经正确。

## 4. 必须保持的正确性约束

1. **完成长度可信。** Planner 使用经过 rejection 修正的 computed frontier，不使用 optimistic CPU length。
2. **Host prefix 可信。** Stable prefix 表示 offloaded KV 已完整写入、所有消费 rank 可见的连续前缀，且不超过有效 computed frontier。
3. **资源不提前复用。** DMA 完成前，source row 的所有权和 generation 不变，被引用的源字节范围不被覆盖，destination blocks 不释放或重新分配，descriptor staging 不改写。Compute 可写同一行中与 DMA 源范围不相交的区间。
4. **Group 映射明确。** 每层使用正确的 group、block IDs 和 table，不假设 group 0 或 owner/member 共享物理编号。
5. **TP 数据完整。** 共享 Host pool 可分工写入；rank 私有 owner HBM 必须获得本 rank 所需的完整 KV。
6. **两种 resident 布局正确。** Dense 与 circular tail 寻址、切换和回绕都保持数据正确。
7. **退出路径可完成。** 取消、最后请求、空调度轮次和 shutdown 都能 drain，不依赖下一次 forward。
8. **MTP reuse 语义不变。** 保留准确的 saved logical lengths，不重复附加 tail 或改变 selection 范围。
9. **失败不会伪装成功。** 失败的传输不推进 prefix、不确认普通释放、不静默启用旧数据。
10. **跨请求缓存可见性正确。** Prefix-cache 的发布或命中消费受完成 frontier 约束；禁止提前释放不能替代禁止提前读取。

每个约束必须有回归测试；涉及异步 NPU 执行的约束还必须有硬件验证。

## 5. 阶段 A 建立基线和公共接入点

固定启动配置、模型、请求集、预热方式及软件版本，记录正确性、失败率、MTP acceptance rate、TTFT、TPOT、吞吐、Host/HBM、最大可承载并发、P→D 传输耗时和本地 D2H/visibility/attention 耗时。

基线覆盖短 prompt、长 prompt、非 block 对齐 prompt、短 prompt 长生成和实际并发。保留请求与结果记录以便逐阶段复用。

| 文件或模块 | 预期职责 |
| --- | --- |
| `vllm_ascend/ascend_config.py` | 配置解析、默认值、兼容性检查 |
| `vllm_ascend/worker/model_runner_v1.py` | 最小生命周期 hook、层分类和 metadata 传递 |
| `vllm_ascend/attention/sfa_kv_offload.py` | Stable-prefix 消费、resident-owner dispatch、onboarding |
| `vllm_ascend/attention/utils.py` | Common metadata 的 stable-prefix 字段与 slicing |
| `vllm_ascend/spec_decode/llm_base_proposer.py` | V1 draft metadata 重建、压缩和各 draft step 的 prefix 传递 |
| `vllm_ascend/patch/platform/patch_async_scheduler.py` 及对应 scheduler 接口 | Prefix-cache 发布/命中与 Host completion 协调，覆盖实际启用的调度器 |
| `vllm_ascend/core/recompute_scheduler.py` | 脚本启用的 RecomputeScheduler/AsyncRecomputeScheduler；接收后 APC、压力回退、preemption 和迟到输出 |
| `sparse_kv_offload/sparse_kv_offload_manager.py` | D2H 状态机、层映射、buffer 和 event 所有权 |
| `sparse_kv_offload/copy_sfa_topk_slots.py` | Request row ownership、generation、释放协调 |
| `kv_p2p/mooncake_dsa_metadata.py` | Request identity、cache-group 和 block IDs |
| `kv_p2p/mooncake_connector.py` | Scheduler/worker 生命周期、按层 destination 路由 |
| `kv_p2p/mooncake_dsa_transfer.py` | Source CP 分片与 destination TP 策略分离 |
| `kv_p2p/sfa_pd_rd2h/` | 保持已有路径，按支持范围补齐对应协议 |
| `csrc/attention/lim_copy_descriptors/` | 新 miss-copy 算子 |

表中省略前缀的 Python transfer 模块均位于 `vllm_ascend/distributed/kv_transfer/`。

Runner 只负责调用时序和输入传递；可复用状态机放 manager/connector。新状态由实例持有，不新增可变全局变量。使用 additional config，不新增环境变量。不要新增模型文件，也不移植 #20289 中无关的 `runtime/config/runtime_config.json`。

**退出条件：**部署清单和基线可复现；已确定 source/destination 布局、MTP 最终 writer、scheduler/worker 释放边界；未确认的支持条件有明确记录。

## 6. 阶段 B 迁移延迟批量 D2H

### 6.1 准确 frontier

优先复用已有 accepted-count 异步回传及完成 event，构造 CPU planner 所需的准确 computed frontier。验证请求重排、新增、退出、padding 和上一轮 MTP rejection 的索引对应关系。

`_correct_optimistic_seq_lens_cpu()` 可作为参考，但它不直接修正原始 CPU computed 数，调用也受 attention 路径控制，不能假定所有 SFA 场景已经执行。

若必须新增设备长度快照，明确其 event 依赖、buffer 生命周期和等待成本。不得通过默认关闭线上依赖的 async MTP 来宣称迁移完成。允许先验证无 MTP 的中间阶段，但最终支持范围必须覆盖实际部署。

### 6.2 Dense 与 ring 源地址

冻结计划时保存实际 source 布局、slot、generation 和 target prefix：短序列 block 从 dense 区读取，长序列 block 从 circular tail 读取。

例如 prompt=100、hot=4096，增长到128 tokens 时，第一个完整 block 位于 dense offset 0，不能读取 tail offset4096。

实现前推导并测试以下边界：第一个完整 block、最后一个 dense block、hot boundary 两侧、ring 回绕、MTP 多 token 写入以及 rejection 回退。证明本轮 target/draft 写入范围不会覆盖正在 D2H 的 source block；仅检查 backlog 小于两个 block 不足以证明安全。

### 6.3 单个 in-flight batch 状态机

```text
上一轮最后一个 target/draft KV writer 完成 → source-ready
    ↓
本轮 retire 上一批 D2H，并验证 TP 完成状态
    ↓
准备准确 frontier、request slots 和 stable-prefix metadata
    ↓
冻结完整 block 计划，独立 stream 启动 D2H
    ↓
与本轮 forward 重叠
    ↓
本轮最后一个 target/draft KV writer 后记录 source-ready
```

Retire 先于本地 row 重用和 metadata 推进。Launch、Host 等待和完成 collective 放在 graph capture 外。Pinned staging 在拷贝期间不可变。TP0 物理拷贝完成后通过一致的完成协议发布 prefix；空 batch、rank 错误和异常路径不得造成 collective 顺序分歧。

按实际 Host backend 选择拷贝实现，保留 Mooncake/AscendVMM 内存注册和地址语义。D2H 只遍历实际 offloaded 层，不能依赖全模型层号连续等假设。

PR #20289 的 planner 假定 exactly one Host KV group。首版默认保留这一限制并在启动时验证；混合布局仍可包含独立 resident-owner HBM 和 indexer groups。如果实际部署有多个 Host groups，必须在交付前扩展为按 Host group 保存 block table、生成 destinations 和统计完成状态，不能复用单一 destination table。将此选择在阶段 A 固定，并为不支持的 group 数提供明确失败测试。

现有 Host visibility 同步先保留；只有证明新完成协议覆盖其依赖并通过测试，才能单独移除重复同步。

### 6.4 Attention 写回切换与 metadata 传递

开启 delayed D2H 的 fused decode 路径只向正确的 resident dense/tail slots 写入当前 KV，并停止该路径原有逐层 `offload_new_kv` Host 写回。Target 和 draft 都必须覆盖；关闭功能和未迁移的 fallback 路径保持原行为，不能用全局禁用写回来影响它们。

将 manager 发布的 stable prefix 加入 `AscendCommonAttentionMetadata`，贯穿 V1 runner、metadata slicing、dummy/capture、proposer 重建、请求压缩和每个 draft step。特别检查 `attention/utils.py` 与 `spec_decode/llm_base_proposer.py` 中显式构造 metadata 的调用点。有效请求的几何计算使用发布的 prefix，不能再次按即时 `seq_lens` 推进；dummy/padding 使用明确的无效行语义。

验证 target/draft 使用同一已完成 frontier，请求压缩后 prefix 与 slot/generation 对齐，启用路径没有重复 Host 写回，旧路径不丢写。保留 full-reuse 的准确 saved logical lengths。

### 6.5 首次接收与 decode 生命周期

P→D 成功接收所有必要数据、resident 初始化完成后，才初始化 generation 和 stable prefix。初始化使用实际可用的连续 KV 范围，并检查 prefix-cache 命中部分是否同样有效；不能仅依据请求长度或完成通知。

为 decode 生命周期维护独立记录，关联 request identity/generation、resident slot、Host block 所有权、in-flight batch、terminal 状态和各 worker 完成确认。接收 tracker 的删除不能删除这些资源的所有权记录。

Scheduler 只有收到匹配身份的安全释放确认后才能释放目的 blocks；worker 同样在所有使用者退出后才能重用 source rows。迟到通知不得作用于新 generation。

取消、正常结束、空调度轮次、最后请求退出和 shutdown 都必须有 drain 路径。错误不推进 prefix；不能在 DMA 状态未知时按成功路径释放其资源。

### 6.6 Prefix-cache 发布和命中

完成 block 的写回和保持 block 存活是不同条件。当前 async scheduler 按 computed tokens 调用 `cache_blocks`；delayed D2H 不能仅靠 refcount 或 terminal 释放握手防止其他请求命中尚未写回的 Host block。

阶段 A 选择并固定以下一种协议：通过 scheduler completion metadata 延迟发布，或在命中消费时等待对应 block 的 Host completion。覆盖实际启用的同步/异步调度器、prefix-cache 查询路径及 cached block 的共享生命周期，不只修改单个输出回调。混合 cache groups 的共享前缀需满足所有必需组的数据可见性。

增加跨请求内容测试：第二请求在第一请求的新 block 已生成但 D2H 尚未完成时命中该前缀，不得读取未完成 Host 数据；完成后可正常复用。再覆盖第一请求取消、迟到完成通知和 cached blocks 被其他请求继续引用的情况。未实现该协议时不能宣称支持开启 prefix caching 的目标部署。

### 6.7 Preemption 和 rollback

首版不声称支持 delayed D2H 下的抢占恢复。必须在调度或资源释放边界保证安全，不能仅在 worker 收到 preemption 后报错，因为 scheduler 可能已经释放 blocks。

若实际部署必须支持抢占，必须完成以下协议后才算满足需求：停止新增计划 → drain/flush 必要 KV → 释放握手 → 恢复 Host/resident 数据 → 重建 generation 和 prefix。

MTP rejection、prefix 回退和一般 preemption 分别测试。原有 tail restore 不能假设 Host 总有最新数据。无法恢复的模式必须明确拒绝，不能静默从不完整 Host 数据重建。

**退出条件：**准确 frontier、dense/ring 内容、资源延迟释放和 TP 完成协议通过测试；V1 与实际 MTP/graph 配置的硬件结果通过；能够单独关闭 delayed D2H 并重启回到基线。

## 7. 阶段 C 迁移 shared-indexer-only inline 模式

先实现 `offload_shared_indexer_only=true`、`miss_onboard_mode=inline`，此阶段不加入 side-stream 并发。

### 7.1 层分类与内存布局

Cache planning 前绑定 layer name、owner/member 关系和 MTP 身份：

| 层类别 | 主 KV 位置 | Attention 行为 |
| --- | --- | --- |
| 非 MTP indexer-owner | Rank 私有 NPU paged cache | Resident attention；有 shared member 时继续生成 LIM selection |
| Shared-indexer member | Host pool + resident working rows | Copy-SFA offload |
| MTP 层 | 依据已有 MTP/offload 语义确定 | 保留 draft reuse 和准确 logical lengths |

确认模型实际存在可卸载 shared layers。无共享层或无有效 Host 层的情况必须在配置/规划阶段明确处理，不能到 allocator 才意外失败，也不能声称与有共享层模型具有相同收益。

同步调整 `store_on_host`、KV specs、memory budget、profile buffers、实际分配、cache tuple 解析、offloaded-layer IDs、descriptor 索引、MTP 映射和 attention dispatch。Resident owner 不再执行 Host/hot-row 初始化。保留本地 AscendVMM、fallback storage 和量化 indexer 实现。

Owner 常驻会比全部卸载增加 HBM，占用必须计入目标并发预算。

实施审查补充：LIM 对长序列只选择已提交 Host prefix，Copy-SFA 会另外加入 tail；标准 paged SFA 不会自动加入。Resident owner 有 shared members 时，先运行 LIM 保留 members 的 miss/selection buffers，再用完整上下文的普通 LI 为自己的 paged attention 生成 indices。首版接受这里的第二次 LI，以确保 tail 和当前 token 可被选择；必须在真机性能中单独计入该开销，不宣称与上游 PR 有相同收益。若后续合并为一次 selection，需证明 cold-topk/tail/dense-short indices 与底层算子约束全部正确。

### 7.2 Cache-group 协议与 P→D 传输

显式建立并贯穿以下映射：

```text
layer → local cache group → local block IDs → destination layout
      → remote source group → source block IDs → source CP shard
```

Scheduler allocation、connector metadata、worker layout、transfer descriptor 和 attention table 使用一致映射。审查 `block_table[0]`、第一个非 indexer group、单一 `main_host_block_ids` 等假设。必要时扩展 metadata，旧配置保留原语义；若 wire schema 改变，明确版本校验和部署升级顺序，不能接受不兼容的旧 metadata 后继续传输。

保留 P 端 source DCP 分片选择，分别处理 D 端目标：

- Host-backed member 沿用共享池的 TP 写入分工。
- Resident owner 由每个消费 rank 从必要的 P shards 重组其 KV，不使用共享 Host 的 block writer filter。
- Replicated indexer 保持独立布局与规则，不能套用 main KV 的 CP source 规则。

Rank-local HBM 需注册正确的内存范围和布局；传输完成要覆盖所有相关层和 groups。失败上报保持正确的 cache-group ID 空间，不把 Host group IDs 直接当成 group 0 IDs。

**退出条件：**Inline 模式逐 rank 数据完整，owner/member 物理 IDs 不同也正确；HBM 预算、MTP、量化 LIM、full reuse 和目标 PD 拓扑验证通过。

## 8. 阶段 D 迁移 side-stream onboarding

引入 `npu_lim_copy_descriptors` 源码、构建配置、Torch binding 和 meta 注册，验证目标 CANN/NPU、Host 地址可访问性及 AscendVMM 兼容性。

保留已修复的 per-group geometry，capture 前创建 stream、events 和固定 buffers，使用 owner plan-ready、per-member copy-done 和当前 forward launch 身份。

```text
Owner 生成 LIM miss plan
    ↓ plan-ready
Side stream 按 member 拷贝 miss KV
    ↓ member-done
对应 member attention 读取已驻留的数据
```

显式选择 member 的 Host-group source block table；只有存在并验证编号一致的不变量时才能复用 owner table。测试中故意设置不同 group 和不同物理 IDs。

验证多 group 不覆盖 geometry/plan，batch 缩小时不读残留 miss，draft reuse 跳过 LIM 时不误用旧事件，graph replay 正确更新动态地址和有效行数。Member 读取前同时满足 copy completion 和 Host visibility。

Full-graph replay 直接执行捕获图，不重新执行 Python `ForwardContext` 身份检查。Replay 正确性必须由图内事件依赖、固定执行路径及 replay 前的 buffer 更新保证。对 owner 执行 LIM 与 draft reuse 跳过 LIM 的路径，明确 graph key/分图或等价机制；验证连续 replay、graph-key 切换和不同 capture size 不会消费上一轮 miss plan。Python launch 身份校验只用于实际执行该检查的 eager/capture 阶段，不能作为 replay 的唯一保护。

算子缺失可明确记录并退回 inline；执行中的拷贝错误不能作为普通缺失而静默 fallback。关于算子双缓冲是否需要重构，以目标硬件 profile 为依据，不把已有争议意见当成已确认性能缺陷。

**退出条件：**与 inline 内容一致，实际 graph/MTP 模式无竞态；能够测量到 copy/compute 重叠；性能不以错误结果或目标容量损失换取。

## 9. 阶段 E 组合验证

两个功能独立通过后，再验证组合：

- Delayed D2H 仅处理实际 offloaded 层，不处理 resident owner。
- LIM、attention 和 onboarding 消费一致的 stable-prefix 语义。
- Onboarding 只从已完成写回的 Host prefix 读取 miss KV；未写回部分由正确的 resident dense/tail 路径提供。
- D2H、onboarding、compute 的 buffer 读写没有冲突，descriptor 和 plan 生命周期明确。
- 资源释放覆盖 transfer、D2H 和 onboarding 的所有使用者。
- Full-reuse 的 saved logical lengths 在组合模式仍正确。

分别测量两个优化，再测组合，避免只测试最终组合而无法定位回归。

## 10. 配置和兼容性

以下字段合并进现有 `sparse_kv_offload_config`，不是完整启动配置：

```json
{
  "delayed_d2h": false,
  "offload_shared_indexer_only": false,
  "miss_onboard_mode": "inline"
}
```

新增配置默认关闭。新功能要求 fused Copy-SFA；side-stream 要求 shared-indexer-only；delayed D2H 要求 connector 完整支持其生命周期。保留现有 hot budget、query width、block size、主 KV dtype、D 端 CP 和其他兼容性限制。配置错误在启动阶段说明原因。

两项关闭时，现有行为和 schema 兼容性保持不变。新模式的能力由实际验证决定，不通过关闭必要的线上特性扩大所谓支持范围。

## 11. 测试矩阵

测试分 CPU 逻辑、NPU 算子/graph 和真实 P→D 系统三层。矩阵采用基础组合加高风险交叉场景，目标部署的完整组合必须做端到端测试。

| 维度 | 必须覆盖 |
| --- | --- |
| 功能组合 | 基线、D2H only、shared-only inline、shared-only side-stream、D2H+inline、D2H+side-stream |
| 拓扑 | P DCP=1 对照与实际 DCP；D TP=1 与实际 TP；最终实际 DP 部署 |
| 长度 | Block 前后、hot boundary 前后、长上下文、短 prompt 长生成、非对齐尾块 |
| MTP | 关闭、实际 token 数、async、rejection、请求重排、full reuse |
| Groups | 顺序变化、owner/member 不同物理 IDs、不同本地/远端 group 编号 |
| Host groups | 首版单 Host group 的校验；若实际部署要求多个，则覆盖逐组 destinations 和完成状态 |
| Batch | 1、小并发、实际并发、增大/缩小、退出后 slot 复用 |
| 生命周期 | 接收时取消、D2H 时取消、最后请求结束、空轮次、shutdown |
| 失败 | P→D 失败、D2H 失败、迟到通知、generation 不匹配、rank 完成分歧 |
| 执行模式 | Eager、实际 graph 模式、不同 capture size 的 replay |
| 恢复 | Prefix cache 命中、跨请求在 D2H 完成前后的复用、rollback；若要求支持则包含 preemption 恢复 |

使用可区分 layer/request/logical block 的数据，逐 rank 核验内容。特别覆盖 prompt=100、hot=4096 的首个完整 block，以及短序列增长跨过 hot boundary 的过程。

优先扩展现有单元测试：

- `tests/ut/kv_offload/test_fused_copy_sfa_config.py`
- `tests/ut/kv_offload/test_copy_sfa_topk_slots.py`
- `tests/ut/kv_offload/test_copy_sfa_mooncake.py`
- `tests/ut/kv_offload/test_mooncake_dsa_metadata.py`
- `tests/ut/kv_offload/test_mooncake_dsa_scheduler.py`
- `tests/ut/kv_offload/test_mooncake_dsa_transfer.py`
- `tests/ut/kv_offload/test_mooncake_dsa_shared_pool.py`
- `tests/ut/distributed/kv_transfer/sparse_kv_offload/test_sparse_kv_offload_manager.py`
- `tests/ut/attention/test_copy_sfa_metadata.py`
- `tests/ut/attention/test_copy_sfa_quantized_lim.py`
- `tests/ut/attention/test_copy_sfa_fallback.py`

新增 delayed-D2H、shared-indexer-only 和 V1 hook 集成测试。扩展 `tests/e2e/nightly/single_node/ops/singlecard_ops/` 下的 metadata、writeback capture、VMM recovery、MTP reuse 测试，并为 LIM copy 算子增加内容和边界验证。实际跨 P/D、跨 TP 的测试放入适合目标硬件拓扑的 e2e 套件，单卡算子测试不能替代它们。

## 12. 验收标准和测量方式

所有寻址、内容和生命周期测试通过；真实拓扑无数据错误、挂起、提前释放或持续资源增长。功能关闭通过原路径回归。

精确 copy 用逐元素内容比较；模型输出和数值比较使用模型适用的确定性设置或既有准确性判据。MTP acceptance、TPOT 和吞吐使用相同请求集、配置、预热方式和多次重复结果，收益必须超过基线波动。

阶段 A 固定用于比较的容差、服务 SLO、目标并发和内存上限；不能看到实验结果后再改变阈值。没有超过波动的性能收益时可交付正确性适配，但不得宣称优化有效或默认启用。

Shared-only 的预期 HBM 增加必须在预算内；最终部署仍需满足目标并发。分别记录 P→D、D2H retire/collective、onboarding、attention、HBM 和端到端 TPOT，解释瓶颈是否转移。

实现后运行相关测试、仓库要求的全套测试和 `bash format.sh ci`。NPU 相关变更在实际硬件验证；缺失的环境和未执行项目如实记录，不标记为通过。

## 13. 实施顺序和提交边界

两个 PR 不存在必须串行的功能依赖。公共基线和正确性设计先行，之后按实际配置选择第一个交付：

| 条件 | 首个交付建议 |
| --- | --- |
| Corrected-frontier 与 decode 生命周期方案已确定 | #20289 |
| Async MTP 必须保留但 frontier 方案未确定 | 先评估 #20478 inline 的混合布局协议 |
| 模型没有有效 shared-indexer 分组 | 不优先投入 #20478 |
| 当前不是 fused Copy-SFA | 先完成路径兼容性验证 |

PR #20478 始终 inline 在前、side-stream 在后；组合验证在两个独立功能之后。调整顺序不能跳过任一阶段的退出条件。

按可审查的功能边界提交，每个行为变更同时包含测试：

1. 配置与兼容性校验。
2. 准确 frontier 和 dense/ring D2H planner。
3. Decode 生命周期、V1 hook 和 delayed D2H。
4. 混合 cache-group 协议与 resident-owner inline 路径。
5. LIM copy 算子与 side-stream onboarding。
6. 组合测试、性能结果和文档。

所有提交使用 Conventional Commits 和 sign-off。Runner/patch 修改按仓库要求说明架构必要性、性能影响和维护方案。若创建 PR，遵循仓库模板并从 fork 提交。计划文件本身不代表授权部署或发布。

## 14. 灰度与回退

新构建首先以新功能全部关闭启动，验证基线；再在隔离实例依次验证独立功能和组合。

开关改变 KV 布局或写回协议，不作为运行中热切换开关。回退应先排空请求、停止实例，再使用关闭新功能的配置重启。DMA 状态未知或已出错时，采用明确的故障终止/清理路径，不以切回同步拷贝掩盖错误。

保留基线构建、配置、算子产物和版本清单，确保能够恢复至 `4cac524d9` 对应部署。如果协议需要 P/D 协同升级，预先验证版本拒绝和回退顺序，避免混跑不兼容实例。

## 15. 最终交付清单

- [ ] 实际部署清单与固定基线可复现。
- [ ] 上游代码保留、重写、未迁移部分有对应说明。
- [ ] Accurate frontier、dense/ring、generation 和释放协议有回归测试。
- [ ] Target/draft metadata 完整传递 stable prefix，新路径无重复 Host 写回，旧路径不丢写。
- [ ] Prefix-cache 发布或命中消费不早于 Host completion，跨请求复用测试通过。
- [ ] 混合 cache-group 寻址、DCP 重组、TP 每 rank 完整性有硬件证据。
- [ ] Inline、side-stream、delayed D2H 及组合均按支持范围验收。
- [ ] Async MTP、full reuse、量化 LIM 和 graph 的实际配置通过验证。
- [ ] 失败、取消、最后请求和 shutdown 无资源生命周期缺口。
- [ ] 性能结果区分独立优化与组合，包含误差及内存/容量代价。
- [ ] 支持边界、启动配置、灰度及回退步骤已记录。
- [ ] 相关测试、全套测试和格式检查的执行状态如实记录。

迁移完成以目标部署通过验收为准，不以代码可应用、单元测试通过或服务能够启动为准。
