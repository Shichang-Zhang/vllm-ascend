# PR 20289 与 PR 20478 迁移进度

目标与验收要求见 [完整计划](MIGRATION_PLAN_PR_20289_20478.md)。本文件记录实施证据、审查和尚未完成的条件；代码或 CPU 测试通过不代表 NPU 验收通过。

用户已明确：本机没有真实部署环境，按指定 scripts 配置继续迁移，真实机器测试由用户后续执行。本次交付须完成代码、可运行的本地测试、逐步子代理审查及硬件验证说明；硬件和性能结果标记为待用户验证，不阻塞本地后续实现，也不虚报通过。

## 阶段状态

| 工作 | 状态 | 证据或剩余工作 |
| --- | --- | --- |
| A1 源码、部署脚本和本地测试基线 | 子代理审查完成，意见已修订 | 补录 RecomputeScheduler 与测试版本限定 |
| A2 目标硬件与性能基线 | 用户后续验证 | 用户明确本机无环境，按 scripts 继续本地迁移 |
| 配置与兼容性实现 | 子代理审查通过，建议已补充 | 新开关、模式/connector/PD限制及 MTP3 预算回归；补测旧非 Mooncake connector |
| B 准确 frontier、planner、V1、生命周期和 APC | 本地实现及逐步审查完成 | NPU 时序、准确率、性能和目标容量仍待用户验证 |
| C 混合 group 和 shared-only inline | 本地实现及逐步审查完成 | 层分类、分配、attention、v2传输和多组失败处理已接入 |
| D Side-stream onboarding | 本地实现及逐步审查完成 | 算子、stream/events、MTP分支和关闭时序；真实NPU验证待执行 |
| E 组合与硬件验收 | 本地交付及子代理复核完成 | 七组配置、组合数据验证及真机操作已提供；硬件验收待用户执行 |

## 固定代码与本地环境

- Ascend 起点：`4cac524d9d19fb71fee4940c782884ed0773782f`。
- 本地 vLLM 源码：`/home/sigh/proj/mayi/vllm`，HEAD `d02df748bf9efd99022f1a062597dc3cb3808485`。
- 仓库 `.venv`：Python 3.13.14、PyTorch 2.10.0+cpu、NumPy 2.4.6、pytest 8.4.2；有 ruff 和 pre-commit。
- `.venv` 没有 torch-npu，主机没有 `npu-smi`、`/dev/davinci*` 或部署脚本引用的 `/vllm-workspace`。
- vLLM 源码缺少安装版本元数据；UT 使用已有 `VLLM_VERSION=0.26.0` 覆盖，与部署脚本一致。这不是验证远端安装版本的证据。
- 原有未跟踪文件保持不变；本次迁移不改用户已有技能、工具或锁文件。

## 用户指定部署脚本

用户指定 `/home/sigh/proj/mayi/sig-ai-inference/scripts`。只读取脚本，未 source 或执行启动命令，也未修改该仓库。

| 项目 | 脚本中可确认的值 |
| --- | --- |
| Served model name | `glm-52`；真实模型路径及 revision 仍是占位符 |
| Prefill | 两节点，DP4、TP8、DCP8、PCP1、PP1，eager |
| Prefill MTP | `deepseek_mtp`、1 speculative token、eager |
| Prefill connector | 默认启用 KV pool，MultiConnector 包含 AscendStoreConnector 和 MooncakeConnectorV1 |
| Decode | 两节点，外部 DP16、每实例 TP2、PCP1、DCP1 |
| Decode connector | MooncakeConnectorV1、`dsa_pd_offload=true`、load failure policy `fail` |
| 当前 decode offload | `use_fused_overlap=1`、`host_backend=mooncake`、hot budget 4096、DRAM 100 GiB/DP、LRU threads 16 |
| Decode MTP | `mtp`、3 speculative tokens、draft eager；`use_index_cache=true` |
| Decode scheduler | `recompute_scheduler_enable=true`，实际为 RecomputeScheduler 或 AsyncRecomputeScheduler |
| Decode target graph | `FULL_DECODE_ONLY`，capture sizes 1,2,4,8,12,16,20,24,28,32 |
| Decode 容量 | max model len 1048576、max batched tokens 32、max seqs 8、block size 128 |
| Indexer | `enable_sparse_li_c8=true` |
| Async / APC | Decode 脚本没有显式开关，必须核对实际启动日志/解析后配置；Prefill 显式开启 prefix caching |
| Hardware access | P/D IP 和 MODEL_PATH 为占位符，docker 脚本仅是本机启动模板，无可连接的远端入口 |

### 必须增加的迁移前置步骤

当前脚本是 fused-overlap，而两项 PR 以 fused Copy-SFA 为目标。保留当前配置作为基线 A，另建立新功能关闭的 Copy-SFA 基线 B，不能把模式切换收益计入两个 PR。

基线 B 的必要变化是 `use_fused_overlap=false`、`fused_op_type=fused_copy_sfa`。本地配置要求 `topk_buffer_size >= (1 + num_speculative_tokens) * 2048`，所以 MTP3 的最小 hot budget 是 8192；当前4096不能原样切换。保持 MTP3，单独计入8192带来的 HBM变化，并验证目标并发。

实际硬件基线由用户后续执行。本地实现和审查继续，不将 A2 或任何硬件退出条件标为通过。

## 本地测试记录

首次无版本覆盖运行在 conftest 中因 vLLM 版本为 dev 而失败。使用部署已有版本设置重跑：

```bash
VLLM_VERSION=0.26.0 .venv/bin/python -m pytest -q \
  tests/ut/kv_offload/test_fused_copy_sfa_config.py \
  tests/ut/kv_offload/test_copy_sfa_topk_slots.py \
  tests/ut/kv_offload/test_mooncake_dsa_metadata.py \
  tests/ut/kv_offload/test_mooncake_dsa_scheduler.py \
  tests/ut/kv_offload/test_mooncake_dsa_transfer.py \
  tests/ut/attention/test_copy_sfa_metadata.py
```

结果：192 passed，15 warnings。这是所列 CPU UT 的基线，不包括全套 UT、模型运行、NPU算子或跨节点性能。

192项测试针对本地 vLLM `d02df748…` 源码及0.26.0兼容分支；版本覆盖不改变源码，不证明目标镜像实际 vLLM commit/ABI 兼容。硬件验证说明需要求记录目标镜像 commit。

### 配置步骤

增加 `delayed_d2h`、`offload_shared_indexer_only` 和 `miss_onboard_mode`。新模式限定当前目标的 MooncakeConnectorV1 DSA consumer；旧模式的其他 connectors 不受影响。新功能 runtime 尚待后续步骤接入，此时不声称功能已经可用。

用户再次确认目标为 MooncakeConnectorV1；计划已明确 P 端保留 MultiConnector 子连接器组合，D 端使用直接 MooncakeConnectorV1，新功能不迁移到 SfaRemoteD2H。

上面的六文件测试命令加入新配置用例后：229 passed，15 warnings；修改的两个 Python 文件 ruff format/check 通过，git diff --check 通过。

配置审查通过后补充新开关关闭时的 SfaRemoteD2H/MultiConnector 回归，配置文件与既有 `TestSparseKVOffloadConfig` 合计69项通过。

额外运行 `tests/ut/test_ascend_config.py` 出现26项设备自动识别失败；在独立 pytest 进程中通过 pytest_configure 加载 `git show HEAD:vllm_ascend/ascend_config.py` 的原始模块进行对照，仍为相同26项失败、33项通过。失败发生在无设备平台的 `VllmConfig()` 初始化；未改动生产代码或原有测试来掩盖这些既有环境失败。

### CPU frontier 与 D2H planner

新增 `delayed_d2h.py`：按上一轮真实 computed frontier 和 accepted counts 修正本轮 MTP 长度；仅计划完整 block，dense/ring 分别寻址。规划不推进 Host prefix，完成确认后才发布；冻结 descriptor 输入并禁止 in-flight row 释放或换代。

`tests/ut/kv_offload/test_delayed_d2h.py`：初始28 passed，15 warnings；ruff check/format 通过。测试包括 rejection 边界、请求重排、新请求、非连续 Host block IDs、短 prompt 持续生成跨 hot 边界和 ring 回绕、先执行未来写再拷贝的内容校验、generation 与错误完成保护。

子代理审查通过，另做30组随机内容校验通过。按建议增加非零 slot、写7接受1～4并填充 rejected poison 的内容测试，以及两个 ready 请求乱序 mixed dense/ring 的测试，现32项通过。此处尚未验证 NPU 拷贝、TP collective 或 runner/connector 集成。

### Mooncake 批量 D2H executor

新增 `delayed_d2h_executor.py`，manager 按真实 offloaded layer 列表注册各 K/V view，启动时要求单 Host group。使用 TP0 的本地 Mooncake VMM 地址和现有 batch memcpy kernel；单份 pinned descriptor staging、独立 stream、source-ready 与 completion events 均由 executor 实例持有。Close 先排空拷贝，失败时保留 Host allocation。

每个 retire 边界均执行 CPU TP completion collective，包括空批；比较 epoch、完整计划摘要与错误状态，避免空/非空分歧跳过 collective 或错误发布。该选择偏向确定的完成协议，额外 CPU collective 开销必须记录进真机性能结果，尚未宣称有性能收益。

Planner、executor 与既有 manager 测试初始94 passed。审查发现单 UniformType group 可同时包含 Host main KV 与 device indexer，已改为按逐层 `store_on_host` 筛选并用实际 Ascend/C8 specs 补测；另外补充连续两批的内容、staging 复用和 event 轮换测试。最终96 passed，15 warnings，ruff 与 diff check 通过；子代理复核通过。

Executor 测试用 CPU tensor 实际拷贝内容及两线程 collective 模拟，覆盖非零 row、不同 K/V widths、非连续模型层号、失败/迟到/不同计划、空批、部分提交和 shutdown。它们不证明 NPU kernel、真实 HCCL/Gloo 或跨节点执行通过。V1/connector/APC 接入尚未完成。

### Attention/MTP 完成前缀传递与写回切换

新增 CPU-owned `req_topk_buffer_stable_prefixes` 字段，贯穿 common metadata slicing、V1 runner buffer 传递、proposer dummy 构造及两条 rejection metadata 重建路径。Delayed fused target/draft 根据完成前缀计算 dense/ring 几何，只写 resident KV，停止该路径逐层 Host 写回。旧路径和 fallback 保留原写回。

子代理发现超出 `max_model_len` 的 draft 会被 proposer 重置为 `seq_lens=1`、`slot_mapping=-1`，保留长 prefix 会导致负 logical length 和真实 resident 行误写。已基于已有 device slot mapping 将无效行及 token padding 路由到 private dummy row，令 prefix/logical length 为0，并使相关 LIM history 失效；full-reuse 仅屏蔽当前调用长度，不修改 saved selection。

以下7个文件合计125 passed，15 warnings；子代理独立复跑并复核通过，ruff 和 diff check 通过：

```bash
VLLM_VERSION=0.26.0 .venv/bin/python -m pytest -q \
  tests/ut/attention/test_delayed_d2h_metadata.py \
  tests/ut/attention/test_copy_sfa_metadata.py \
  tests/ut/attention/test_copy_sfa_fallback.py \
  tests/ut/attention/test_sfa_kv_offload.py \
  tests/ut/spec_decode/test_llm_base_proposer.py \
  tests/ut/spec_decode/test_copy_sfa_mtp.py \
  tests/ut/attention/test_copy_sfa_quantized_lim.py
```

包含调用真实 `attn_update_stack_num_spec_norm` 的 eager/full 与 reuse on/off 超限回归，以及真实 proposer prepare_inputs/padded、metadata slicing、非连续 slot/generation、target/draft 完成边界和写回分流测试。没有执行真实 NPU graph。Runner 当前仅分配和传递 buffer，未由 planner 填入完成前缀；active row 的未初始化值为-1并明确拒绝，不能据此启用新功能。下一步接入运行时 retire/launch、准确 frontier、首次 Host/resident 初始化和 Mooncake release/APC 协议。

### Mooncake scheduler decode ledger 与消息协议

新增独立 `DsaDecodeLedger`，在接收 tracker 删除后继续持有 decode generation 和 Host 完成前缀。新增 register/release command、prefix/release/failure result；P→D 本地完成结果携带 generation，聚合保持 receive 和 decode 消息分离。Scheduler 对新模式只接受匹配 generation 的 TP 确认，不从无 generation 的原始通知释放 blocks。

正常接收仍通过 `finished_recving` 推进等待状态；取消中的接收完成不单独释放 decode blocks。接收已排空且所有 TP rank 确认 decode release 后，使用 `finished_sending` 触发最终释放，避免把运行中 decode completion 当作第二次 receive terminal。APC 全命中/notify-only 请求也有独立 decode lifetime；分配前取消则不持有不存在的 decode 资源。

Ledger、scheduler、metadata、shared-pool 四文件初始合计76 passed，15 warnings。审查发现已释放 rank 的重复 release 可以推进 prefix，已要求重复确认与最终 prefix 完全一致；新增跨批次推进/回退拒绝及幂等回放测试，生命周期测试现30 passed，子代理复核通过。

额外六文件 Mooncake 回归初始325 passed、1项本机 socket 被沙箱禁止；该项在允许本机 socket 的环境单独重跑1 passed。覆盖双 TP、取消与接收乱序、旧 generation、错误消息原子性、失败不发布、重复释放、完成前缀单调性及旧模式。Worker/runner 尚未产生新 decode 结果，因此此步骤仍不能用于实际启用 delayed D2H。

### APC 发布和 block 释放保护

新增实例级 `MooncakeDecodeCacheGuard`，仅在 Mooncake delayed D2H 生效。通过现有 KVDeliveryScheduler 构造链安装，统一拦截实际 coordinator 的发布与释放：`allocate_slots` 内部发布和显式 `cache_blocks` 均限制到 TP 确认的 Host prefix；`free` 与 deferred `pop_blocks_for_free` 在独立 decode lifetime 排空前拒绝释放。所有状态归 coordinator/connector 实例持有，不增加全局状态或环境变量。

新增18项测试通过，使用真实 KVCacheManager、block pool、APC 查找与 refcount，覆盖单/混合 group、未完成 Host 内容不可命中、单 TP 确认不足、取消后共享缓存引用存活、旧 generation 不能发布、APC开关、真实 preemption 与压力 fallback abort。构造链测试覆盖 Scheduler、AsyncScheduler、RecomputeScheduler、AsyncRecomputeScheduler。结合既有 scheduler/lifecycle 测试合计72 passed，ruff/diff check 通过；子代理独立验证18项并审查通过，按建议补录 patch 用途和移除条件。Worker/runner 接入仍待完成。

### Worker decode 生命周期接入

新增 worker 执行线程持有的 `DsaDecodeWorkerState`，接收线程仅把带 generation 的完成消息入队。保留与 receive tracker 独立的 Host-ready、resident-bound、terminal 和 failure 状态，worker metadata 独立序列化 prefix/release/failure。APC notify-only 可在清理通知仍在发送时执行 decode，但 release 必须等该线程退出。

审查发现并修复分配前取消的 generation=0 清理通知，以及旧/已完成 receive 命令可能再次实际派发的缺口。已加入执行线程持有的单次 dispatch claim，覆盖 callback 删除 active entry 与完成消息入队之间的并发窗口；旧代取消消息不能取消新代接收。

新增 worker 测试与原 shared-pool 测试合计50 passed，覆盖实际 P8/DCP8→D TP2 endpoint handler、接收后 tracker 删除、阻塞中的传输取消、失败和 notify-only、generation 重用与重放。子代理独立50项通过，并用两个线程暂停 completion 入队验证竞态修复，最终复核通过。

### Runtime 与 V1 调用时序

新增 `DelayedD2HRuntime`，在每个 execute step、包括空调度轮次中先 retire，再释放已排空的 terminal rows。新 row 从已验证 Host 范围恢复 dense 或 ring tail；使用实际 Host group 的 CPU block table、scheduler generation 和稳定 row。未调度的请求仍持有 row。`_prepare_inputs` 复用已有 accepted-count event，并按 device recurrence 生成准确 frontier；metadata 冻结 target+所有 draft 的写预算后 launch；sample_tokens 最后一个 draft 后记录 source-ready。

首次 block 对齐 prompt 的最后一个 token 重算可使用保守 planner prefix，不回退已验证的 Host prompt 发布。释放和 shutdown 额外排空计算流；shutdown 先排空接收和 D2H/compute，再注销 Mooncake 注册内存，失败则保留注册。

Runtime 初始12项通过，加入真实 V1 metadata hook 后的六文件回归141 passed、1项测试构造参数错误；修正后该项单独重跑1 passed。Manager/Host pool 两文件45 passed。

审查发现 notify-only 完成顺序可能不同，按各 rank 本地状态回收会导致 row 分配顺序分叉。已复用 executor 原有 retire collective 协调共同可释放集合，并按统一顺序回收；不增加第二次每步 collective。新增双 TP 乱序通知、满容量回收后新请求绑定相同 slot、后续 D2H 签名一致的回归。Runtime/executor 合计41 passed，子代理独立相关六文件140 passed，最终复核通过。阶段B本地实现及审查完成；不代表 NPU 时序、准确率、性能或容量验收通过。

### Shared-only 层分类

在 model/drafter 加载完成、cache planning 之前，以实际 attention 对象与 drafter 模块成员关系确定 MTP 身份。按模型执行顺序建立 owner/member 映射，支持多个 owner 重用同一 selection buffer；分类结果为不可变 plan。非 MTP owner 常驻，shared member 与 MTP 保留 offload；没有 target shared layer 时明确失败。

子代理审查通过，独立15项 CPU 测试与 ruff 通过；核对了真实 MTP 加载顺序、模块树和 static_forward_context 对象身份。按非阻塞建议补测 target/draft 使用同一 buffer storage、但 draft 独立计算 selection 的情况。此步骤尚未改变缓存分配或 attention dispatch，不据此宣称 shared-only 已可用。

### Shared-only 缓存分配与注册

Runner 的逐层 specs、allocation、reshape 依据 residency plan 设置 `store_on_host`；owner 使用私有 paged HBM，shared/MTP 使用原 Host/hot tuple，C8 indexer 保留 INT8 key 与 FP16 scales。Manager 在分配前展平真实逐层 specs，验证一个 Host group、各组 block size=128、完整私有 storage；支持独立 resident/indexer groups，明确拒绝 HMA tensor aliasing。Offload IDs 与 MTP IDs 来自实际 plan，不再根据总层数推测 MTP。

初始分类/布局35项通过；补充真实 runner spec 和真实 vLLM planner 组合测试后，混合布局文件21项通过。实际 planner 为默认 full-length 混合 specs 生成一个 UniformType group 和逐层独立 tensors；HBM budget 已计入 owner，profile 只为 Host 层保留 hot buffers。子代理审查通过，非阻塞建议已固化为 planner UT。

扩展 runner/manager/executor 回归101 passed、1项既有 `pin_memory()` 失败。子代理在独立 pytest 进程中用 AST 提取 HEAD 的 `_calc_spec_decode_metadata`，确认与当前方法完全相同，替换后该项仍因缺少 `PrivateUse1HooksInterface` 失败；没有改动生产代码或旧测试掩盖环境限制。此步骤不代表 attention 或 P→D 混合传输已经通过。

### Shared-only attention 与缓存组寻址

Common metadata 携带实际 group-indexed tables 和 Host group ID，贯穿 V1、MTP dummy、rejection rebuild、padding/unpadding。Indexer 依据独立 table、device seq_lens 和 query ends 计算 slots，保留 `-1` 屏蔽；C8 key/scales 使用普通 scatter，避免旧 block-store descriptors 指向 main group。Resident owner 使用私有 paged write/attention；Copy-SFA miss 与 tail restore 显式使用 Host group table，非 delayed dense-fill 同样不再假设 group0。

审查发现 LIM 只选择已提交 prefix，直接给标准 resident SFA 会遗漏 tail。已改为 owner 保留 LIM buffers 供 members，再执行完整上下文 LI 供自身 attention；singleton owner 只执行普通 LI，MTP full reuse 保持原路径。新增 inline/delayed 内容与参考 attention 测试验证当前 token 可被选择，结果匹配参考且 member LIM buffers 未改变。第二次 LI 的性能代价已写入计划，真机待测。

子代理最终复核8文件150 passed、ruff通过，无剩余阻塞。包含真实 V1 多 group metadata hook、实际 native forward 的 C8/BF16 indexer写入、不同 Host/owner物理ID的 poison 内容检查、Host tail descriptor、MTP maxlen mask（delayed开/关）与 full reuse。

额外9文件回归182 passed、3项原有 SFA C8 测试在 mock 装饰器处失败（CPU torch_npu stub 缺少 `npu_interleave_rope`、`npu_dynamic_quant`、`npu_kv_rmsnorm_rope_cache`）。独立 pytest 进程确认对应3个生产函数与HEAD的AST一致，替换为HEAD函数后仍为相同3项失败。没有将这些失败或 NPU graph、时序/性能标记为通过；Mooncake 混合传输尚待接入。

### Mooncake 分组协议

新增 v2 source/destination group records，携带全部 manager group IDs、block sizes 和物理 block IDs；新 Decode 必须先验证 Prefill 版本，旧 aliases 与完整 group geometry 冲突时拒绝。Scheduler 按实际 Host group 选择目的地，失败 IDs 覆盖所有组的 unhashed blocks；分配前取消保持无目的地的清理通知。

子代理独立五文件97 passed；按建议补充 alias 严格校验后分组文件27 passed，复核通过。原 scheduler/shared-pool 回归47 passed。此步只完成协议与调度命令，worker 分组寻址和多组失败处理仍需后续接入。

### Mooncake 混合缓存 worker 传输

Worker 按实际逐层 spec/group 注册 indexer 和每 rank 私有 owner，Host VMM 保持共享。接收端校验 GET_META v2、实际 layer→manager group 和 block size，分别规划 Indexer D2D、Host D2RH、resident Main D2D；Host 按 D TP 分工，每个私有 owner 接收全部 CP 分片。全部 phase 在提交前完成寻址验证，失败与取消仍等待 endpoint tasks 排空。

子代理复核通过，独立四文件180 passed；新增17项包含 CPU 实际字节拷贝、不同源/目的 group 顺序、C8 key/scales、P8/DCP8→D2、非连续 IDs、partial prefix、私有 tensor 注册及错误提前拒绝。原 connector 单独运行122 passed，socket 项因沙箱限制单独在沙箱外重跑1 passed。扩大同进程组合时出现10项旧测试 mock 定位失败：该测试文件导入时移除再恢复真实 modules，已导入的类与 patch target 分属不同 module；尚未将该组合运行标记为通过，后续整理测试隔离。多组 scheduler 失败路径待下步接入。

### Mooncake 多组失败处理

通过现有 scheduler 构造链接入实例级失败 adapter，保持 DSA 已要求的 `failure_policy=fail`。扫描全部 group 的实际 block IDs，以最早坏逻辑 block 截断，失败所有共享坏块的请求；同步失败驱逐所有组的依赖后缀，异步接收失败保留已缓存的共享 prefix 引用。此路径不释放仍被 worker 持有的 blocks，也不依赖 delayed D2H 开关。

五文件103 passed；子代理初审34 passed，并建议补充失败到释放的串联回归。补测经真实多组 `_handle_invalid_blocks`、`finish_requests` 和双 TP release ACK，第一条 ACK 保留物理块，第二条后才释放并删除请求；文件17 passed，子代理复核通过。阶段C本地实现与审查完成，硬件验证仍待用户执行。

### LIM miss-copy 算子

从固定 #20478 源码迁入11个算子文件，接入 A3 build、Torch binding 和 meta。实际目标 KV tensor 直接贯穿 Torch→ACLNN→kernel，显式声明 mutation，避免无输出 op 被 DCE 或图变换后原始目的地址失效。参数增加 Host/pool/plan 容量，UB latent/rope 边界按32字节对齐；更新本地 ABI 与硬件验证说明。

子代理复核通过，2项 CPU 测试通过：使用 g++/ASAN/UBSAN 和同步 AscendC primitives 执行真实 kernel，核对完整内容、变化的 active rows、live table stride、非法 IDs 和非对齐 head dims；另从实际 Torch schema 创建 FX 图，验证 DCE 保留可变缓存副作用。新增 NPU eager/graph 测试但未执行。无 CANN，尚未进行目标编译、NPU/VMM、事件时序或性能验证。Side-stream manager/attention 接入待下一步。

### Side-stream manager 与 attention 接入

每个 target owner 绑定独立 geometry 与 plan-ready event，每个 member 绑定实际 Host group、固定参数和 copy-done event；流和 buffers 在 KV cache 初始化后、capture 前创建。Owner 完成 LIM 后提交拷贝，再执行自身完整上下文 LI；member 等待自己的 copy 后使用 zero-miss attention。Draft owner/group 留在 inline/reuse 路径，graph 内固定执行的记录/等待承担 replay 同步，Python forward token 只检查 eager/capture。

初始化显式触发 lazy custom-op 注册，检查 A3 和新 ABI，并以 zero-active launch 验证实际 kernel 可调用。只对 schema 缺失或未构建的 SoC 回退 inline，执行错误不会吞掉。重新绑定缓存和关闭时先排空 side stream，失败则保留 Host allocation。

扩展8文件123 passed；按建议补充 lazy 加载测试后相关3文件47 passed。子代理最终独立回归83 passed，复核通过。新增真实 NPU 两owner/三member fork/join、capture size1/3反复切换和动态源表测试，尚未执行。CPU capture/replay模拟只验证捕获操作的内容与依赖结构，不代表NPU图捕获、Host可见性或性能通过。

### 组合回归与验证交付

提供 [七组完整 Decode 配置和真机操作](examples/disaggregated_prefill_v1/mooncake_dsa_migration/README.md)，覆盖原 fused-overlap、Copy-SFA 基线、delayed、shared-only inline/side-stream 及组合。保留参考 scripts 其他 additional-config 字段；未修改或执行外部部署脚本。新增组合测试8项通过，包括七组配置的实际 MTP3 校验，以及 delayed D2H→Host commit→side-stream onboarding 的完整内容，确认未提交的 future tail 不被读取。

修复上述旧 Mooncake 测试的 module 身份隔离：每个测试临时绑定该文件自身导入的 modules，结束后由 fixture 恢复，保证 string patch 与测试对象一致。组合七文件353 passed、1项 socket deselected；该 socket 项先前已单独通过，并在下面完整相关回归中再次执行。

按 [34文件清单](examples/disaggregated_prefill_v1/mooncake_dsa_migration/local_tests.txt) 合并运行，结果 **870 passed、2 failed**。两项失败均位于既有 `test_kv_delivery_preemption.py`：`test_all_sync_scheduler_backports_live_in_kv_delivery_patch` 和 `test_kv_delivery_schedule_body_matches_target_vllm_commit`。对照 HEAD 后仍复现相同两项：当前 schedule 的 AST 与 HEAD 一致，BalanceScheduler 文件与 HEAD 字节一致，再使用 HEAD 构造函数/schedule 重跑仍失败。未修改生产逻辑或原断言以掩盖基线问题。

尝试全套 `VLLM_VERSION=0.26.0 .venv/bin/python -m pytest -q tests --maxfail=5 --tb=short`，收集阶段出现4处错误：缺少 modelscope、ray、uvicorn，以及原 `test_sfa_pd_rd2h_connector.py` 的 examples 包导入冲突。不能把相关回归结果表述为全套测试通过。其他已证实的无NPU基线失败见前文。

格式 CI 在 `/tmp` 中的隔离副本运行，保留根仓库 index 和用户文件。初次检查发现一个测试空行及两个新测试禁止导入（pickle、re），均已修正：分组命令改用 msgspec 类型化往返，schema 正则使用 regex，相关两文件29 passed。`bash format.sh ci` 最终全部通过，`git diff --check` 通过。

新的 Stage E 子代理独立173项通过，并反转文件顺序验证隔离 fixture；socket 项在其沙箱内受限，不计入该173项。按建议将 NPU 数据改为 BF16 精确表示的 token/channel/layer 编码，KV/rope 使用独立值，并在 graph replay 后同时核对两者完整内容。双方独立 CPU 验证64/512维、896 tokens×6 salts 不重复；这只验证测试数据，未执行 NPU 用例。最终复核通过，无剩余建议。

## 设计接入点

- V1 frontier：优先使用上一轮 accepted counts 和已有 completion event，映射到本轮请求顺序；不修改 optimistic CPU 账本来影响原有调度逻辑。
- D2H：单 in-flight batch，dense/ring source 地址分别处理；支持本轮 compute 写入同 row 的不相交范围。
- APC：倾向在完成 frontier 处延迟发布。需要拦截 `KVCacheManager.allocate_slots` 内部直接调用 coordinator 的发布点，以及显式 `cache_blocks` 路径；仅改 async scheduler 输出回调不足够。
- Terminal：独立 decode ledger，区别 receive-complete 和 decode-release-complete；不能把后者错误地作为运行中请求的第二个 finished_recving。
- Preemption：现有 scheduler 有独立的 KV delivery/backpressure patch；保护必须发生在释放之前，不能仅在 worker 发现 preemption 后报错。
- 实际 scheduler：同时覆盖 RecomputeScheduler/AsyncRecomputeScheduler 的接收后 APC 发布、压力 fallback abort、async preemption 直接释放及迟到输出，不仅覆盖普通 Scheduler/AsyncScheduler。
- 混合 groups：现有 main-host/indexer 两组协议必须扩展；APC需等待所有必需组的数据可见。

## 交由用户后续验证的外部证据

- 可访问的 NPU 主机/容器及真实模型路径，现有脚本只提供占位符。
- 实际解析后的 async/APC配置、模型 revision 和 shared-indexer 分布。
- 原始 fused-overlap 与 Copy-SFA 基线的准确率、MTP acceptance、HBM、TPOT和容量。
- 实施前固定的比较容差、服务SLO、目标并发和内存上限。

本地实现不替代上述硬件证据；用户后续真机验证仍须满足原计划的验收要求。
