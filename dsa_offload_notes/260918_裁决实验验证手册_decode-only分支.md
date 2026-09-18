# 裁决实验验证手册 · decode-only 分支（`repo3/vllm-ascend-decode-fix`）

> **给执行实验的同学**：本文只讲"要验证什么、怎么跑、日志怎么看、结论怎么下"。
> 代码位置：`d:\hanjiang\0917mayi材料\repo3\vllm-ascend-decode-fix`
> 分支：`fix/decode-graph-kv-probe`（= 远端 `dsa_offload_rebase_pr15642_0911_decode_test` 的 `3fa38c72b` + 探针提交 `31d2e6edc` + 修复提交 `2e9f6194d` + 本次新增的裁决探针，最后一项为工作区未提交改动）
> 生成日期：2026-09-18
> 配套文档：`260918_裁决实验验证手册_主干分支.md`（同一套裁决逻辑在主干分支上的版本）

---

## 0. 三句话摘要

1. 本分支的探针**不开时完全短路**，开关是 `VLLM_ASCEND_SFA_INDEX_COPY_PROBE=1`；日志前缀有四个：`[SFA_INDEX_COPY_PROBE]`（eager 逐元素比对）、`[SFA_GRAPH_TRACE]`（图模式计数）、`[SFA_READY_BCAST]`（TP 放行广播）、`[SFA_PLAN_TRACE]`（planner 复用决策）。
2. 本分支要回答的核心问题是：**在这个"单独拉起 decode 服务"的代码上，(a) 时序修复是否真的生效、(b) eager 的"数据一致"结论为什么不能覆盖图模式、(c) 哪些层根本不发集合通信（peer 无从被放行）**。
3. 最需要盯的三类日志：
   - `[SFA_GRAPH_TRACE] stage=host_kv_visibility_post_write ... values=[n]`，`n>0` ⇒ 同一步 Host 读依赖真实存在且当时未落地；
   - `[SFA_PLAN_TRACE] DECISION ... can_reuse_owner_plan=True` ⇒ 该层**零集合通信**；
   - `[SFA_READY_BCAST] ENTER/RETURNED ... collective_executed=` ⇒ 放行广播是否真的执行。

---

## 1. 本分支是什么 / 与主干分支的差异

| 项 | 主干分支（`repo3/vllm-ascend-fix`） | 本分支（decode-only） |
| --- | --- | --- |
| 用途 | 标准 PD 分离服务的修复与裁决 | **单独拉起 decode-only 服务**（`task.md` 里的 `vllm-serve-26-mooncake-standalone`） |
| 探针总开关 | `VLLM_ASCEND_SFA_ADJ_PROBE`（默认关） | `VLLM_ASCEND_SFA_INDEX_COPY_PROBE`（默认关） |
| eager 逐元素比对 | 无 | **有**（`_probe_mooncake_index_copy`，是"数据一致"结论的来源） |
| 图模式计数探针 | `[SFA_ADJ_PROBE]` | `[SFA_GRAPH_TRACE]`（本次移植，字段格式一致：`frame=/t_us=/values=`） |
| TP 放行广播 | 无开关，始终执行 | 受 `VLLM_ASCEND_SFA_READY_BCAST` 控制（默认 `1` = 执行） |
| planner 决策日志 | 捕获期一条 `stage=plan_reuse_decision` | `[SFA_PLAN_TRACE] DECISION / REUSE_ENTER / REUSE_RETURN / NORMAL_PLANNER_ENTER`（更详细） |
| MemFabric 主机视图比对 | 有（`hostview` 通路） | **无**（可见性探针只在 Mooncake 下生效） |
| 必需的前置处理 | — | **见 §2：`71422442b`（图模式打桩提交）去留问题** |

> 结论对照：**两个分支的判据、日志字段、frame 语义完全一致**，可以互相印证。差异只在"前缀名"和"哪些通路可用"。

---

## 2. 开跑前必须先决定的事：`71422442b` 怎么处理

`task.md` 明确写着："decode only 服务拉起时，使用 decode only 分支时候，需要去掉 `71422442b3eedc615c946a30c976d4fd0e318f72` commit（图模式打桩使用）"。

实测该 commit 的内容（1328 行）分两类：

| 类别 | 内容 | 是否影响服务行为 |
| --- | --- | --- |
| 纯诊断（受开关控制） | `sparse_kv_offload_manager.py` 的探针方法、`sparse_kv_offload.cpp` 的 `[SFA_GRAPH_TRACE]`、`attention/sfa_kv_offload.py` 里的诊断调用 | **否**：开关默认关，全部 `if not self.index_copy_probe_enabled: return` 短路 |
| **功能性改动** | `mooncake_host_pool.py` 新增 `host_data_ptr` 字段，并新增一段**无条件 `raise RuntimeError`** 的 Host 地址范围校验；`attention/sfa_kv_offload.py` 新增 `debug_mooncake_selection()` 调用（有 try/except 兜底） | **是**：地址非法时会**直接中断服务** |

### 推荐流程

**方案 A（推荐，先用它把实验跑起来）**：保留当前工作区不动，**不设置** `VLLM_ASCEND_SFA_INDEX_COPY_PROBE`。

```bash
# 服务启动前确认探针关闭
unset VLLM_ASCEND_SFA_INDEX_COPY_PROBE   # 或确保环境里没有它
```

此时所有诊断短路，服务行为等价于"没有诊断"，跑一轮**基线实验**（见 §5 实验 D0），确认服务可正常出结果。

**方案 B（若基线跑不通 / 明确要求剥离打桩）**：丢掉该 commit 后重新构建。

```bash
cd <你的 vllm-ascend 仓>
git rebase -i 71422442b^
# 在编辑器里把 71422442b 那一行改为 drop，保留 3fa38c72b / 31d2e6edc / 2e9f6194d
```

> ⚠️ **会有冲突**：`31d2e6edc` 与 `2e9f6194d` 是在打桩代码之上写的，drop 之后会冲突。
> 解冲突原则：**删掉打桩新增的诊断代码，保留时序修复**（即 `wait_for_current_kv_writeback` 必须出现在 `fused_op` 之前）。
> ⚠️ 方案 B 之后 **`[SFA_INDEX_COPY_PROBE]` / `[SFA_GRAPH_TRACE]` / `[SFA_PLAN_TRACE]` 等探针会一并消失**，本文第 4/5 节里依赖这些日志的实验将无法执行；如需保留探针请走方案 A。
> ⚠️ `3fa38c72b`（UT 用 g++）与运行时无关，可保留可丢弃。

---

## 3. 实验准备

### 3.1 环境变量

| 变量 | 默认 | 作用 |
| --- | --- | --- |
| `VLLM_ASCEND_SFA_INDEX_COPY_PROBE` | `0`（关） | **探针总开关**。开启后：eager 逐元素比对 + 图模式计数 + `[SFA_READY_BCAST]` + `[SFA_PLAN_TRACE]` |
| `VLLM_ASCEND_SFA_PROBE_FAIL_FAST` | `1` | `1` 时 eager 比对失败会**抛异常中断请求**（便于第一时间抓到）；做长时间压测时可设 `0` 只记日志 |
| `VLLM_ASCEND_SFA_READY_BCAST` | `1` | `0` 时**不执行** TP 放行广播（用于对照实验，见 §5 实验 D3） |

### 3.2 推荐的三轮跑法

| 轮次 | 探针 | READY_BCAST | 目的 |
| --- | --- | --- | --- |
| D0 基线 | 关 | 1 | 确认服务正常、复现现象 |
| D1 诊断 | 开（`FAIL_FAST=0` 压测 / `=1` 短跑） | 1 | 取全部探针日志 |
| D3 对照 | 开 | **0** | 验证"放行广播"是否是真边（对照） |

```bash
# D1 示例
VLLM_ASCEND_SFA_INDEX_COPY_PROBE=1 VLLM_ASCEND_SFA_PROBE_FAIL_FAST=0 \
  <decode-only 启动脚本> > run_r${RANK}.out 2> run_r${RANK}.err
```

日志同样**必须每 rank 一份**（decoder 有 8 个 rank）。

---

## 4. 探针清单

### 4.1 四个前缀总览

| 前缀 | 触发时机 | 关键字段 | 备注 |
| --- | --- | --- | --- |
| `[SFA_INDEX_COPY_PROBE]` | **eager**（`not capturing`）写回后 / 广播后 / planner 后 | `PASS/FAIL stage=… tp_rank=… slots=…`；FAIL 时给 `K:source_row=,slot=,element=,expected=,actual=,max_abs=` | `FAIL` 默认 **抛异常**；图模式**不产出** |
| `[SFA_CURRENT_HOST_READ]` | 同上（每次比对前） | `stage/step_hint/layer_id/tp_rank/kind=CURRENT_THIS_FORWARD/rows/source_rows/host_slots` | 直接列出"本步写了哪些行、落到哪些 slot" |
| `[SFA_GRAPH_TRACE]` | **图模式 replay**（capture 期挂载、replay 期触发） | `stage= frame= t_us= call= layer_id= tp_rank= dtype= numel= values=[...] hash=` | 与主干分支 `[SFA_ADJ_PROBE]` 字段完全一致 |
| `[SFA_READY_BCAST]` | **eager**，TP 放行广播前后 | `ENTER/RETURNED call= layer= layer_id= tp_rank= tp_size= capturing= collective_executed=` | `call` 可把 ENTER/RETURNED 配对 |
| `[SFA_PLAN_TRACE]` | planner 决策处（eager + capture） | `DECISION … can_reuse_owner_plan= … / REUSE_ENTER … metadata_broadcast_will_run=False / REUSE_RETURN / NORMAL_PLANNER_ENTER … metadata_broadcast_will_run=True` | 判断"哪些层零集合通信" |

### 4.2 `[SFA_GRAPH_TRACE]` 的 stage 与含义

| `stage=` | 触发位置 | 判读 |
| --- | --- | --- |
| `host_kv_visibility_post_write` | `offload_new_kv()` 写回发起后（fork 之后） | `values=[n]`，`n>0` ⇒ 写回未落地。`layer_id==0` 时推进 `frame` 并清零所有计数 |
| `host_kv_visibility_pre_join` / `post_join` | `sfa_kv_offload.py:1036` / `:1038`，紧贴唯一的 join | join 前后各一次，见 §5 实验 D2 的矩阵 |
| `host_kv_visibility_pre_bcast` / `post_bcast` | ready 广播前 / 后 | peer rank 上 `pre_bcast>0 && post_bcast==0` 才说明"广播兜住了" |
| `planner_stats` | planner 调用返回后 | `values=[window_refs, window_loaded, window_dropped, dropped_misses]` |
| `index_descriptor_*`（打桩自带） | indicator 描述符生成 | 仅 `layer_id ∈ {0, mtp_layer_id}` |
| 其它 stage | 打桩自带（membership 等） | **只在 `SFA_GRAPH_TRACE_LAYER_ID = 4` 这一层打印** |

### 4.3 记录样例

```text
[SFA_GRAPH_TRACE] stage=host_kv_visibility_post_write frame=137 t_us=8421390 call=37 layer_id=0 tp_rank=0 dtype=4 numel=1 hash=... values=[3]
[SFA_GRAPH_TRACE] stage=planner_stats frame=137 t_us=8429000 call=12 layer_id=0 tp_rank=0 dtype=3 numel=4 hash=... values=[22,0,0,0]
[SFA_READY_BCAST] ENTER call=5 layer=... layer_id=0 tp_rank=0 tp_size=8 capturing=False
[SFA_READY_BCAST] RETURNED call=5 layer=... layer_id=0 tp_rank=3 collective_executed=True
[SFA_INDEX_COPY_PROBE] PASS stage=all_tp_post_ready_broadcast layer=... layer_id=0 tp_rank=0 token_count=1 slots=[1234]
[SFA_CURRENT_HOST_READ] stage=tp0_post_write step_hint=137 layer=... layer_id=0 tp_rank=0 kind=CURRENT_THIS_FORWARD rows=1 source_rows=[0] host_slots=[1234]
[SFA_PLAN_TRACE] DECISION call=9 layer=... layer_id=3 tp_rank=0 backend=mooncake capturing=True skip_topk=True can_reuse_owner_plan=True owner_layer_id=0 ... 
[SFA_PLAN_TRACE] REUSE_ENTER call=9 layer=... layer_id=3 tp_rank=0 copy_owner_map=True metadata_broadcast_will_run=False
```

字段字典（`[SFA_GRAPH_TRACE]`）与主干分支完全一致：`frame` 由 `post_write`+`layer_id==0` 推进；`t_us` 为进程内单调微秒；`dtype` `3`=int32 / `4`=int64；`values=[...]` 前 8 个元素原值（计数器即行数）；大张量只有 `hash`（用等/不等比较）。

---

## 5. 实验逐条

### 实验 D0 · 基线可用性（探针关）

* **目的**：确认这个分支能把 decode-only 服务跑起来并复现现象；也是一切结论的"无探测"参照。
* **操作**：不设探针变量，跑 `aime2026` 题目（`MAX_TOKENS=8192`）。
* **判读**：服务无 5xx、输出与预期基本一致（或复现你们观测到的精度问题）。若 5xx / 起不来，先走 §2 方案 B 再回来。

### 实验 D1 · eager 比对覆盖了什么、漏了什么（理解"数据一致"为何不可外推）

* **目的**：确认 `_probe_mooncake_index_copy` 只在 eager 生效 ⇒ 它的 PASS **不能证明图模式安全**。
* **操作**：开探针，**先跑一轮 eager 路径**（例如关闭图模式 / 让 `capturing=False` 的请求阶段），再跑图模式。
* **判读**：

| 观测 | 结论 |
| --- | --- |
| eager 全程 `PASS` 且图模式 `[SFA_GRAPH_TRACE]` 出现 `post_write>0` | **"eager 一致"与"图模式安全"无关**：eager 无 fork，天然有序；图模式有 fork 才有缺口 |
| eager 出现 `FAIL`（并抛异常） | 连 eager 都不安全 ⇒ 问题不在时序，优先查 descriptor/slot_mapping（`index_descriptor_*`、`slot_mapping`） |
| 图模式下完全没有 `[SFA_INDEX_COPY_PROBE]`/`[SFA_CURRENT_HOST_READ]` | 正常现象（探针在 `not capturing` 才挂载）——**这正是本次要补的盲区** |

* **`[SFA_CURRENT_HOST_READ]` 的用法**：它给出 `source_rows`/`host_slots`，可以用来确认"写回是否在写本步的行"。若 `rows>0` 且 `source_rows` 属于本步，则说明**本步写回的行确实存在**，配合 planner 统计即可回答"是否有同一步读者"。

### 实验 D2 · 图模式时序裁决（核心）

* **目的**：在 decode-only 服务上直接裁决"缺边"。
* **操作**：开探针 + 图模式，grep `[SFA_GRAPH_TRACE] stage=host_kv_visibility_`。
* **判读矩阵**（`n>0` = 该时刻仍有 n 行未落地）：

| `post_write` | `pre_join` | `post_join` | 含义 |
| --- | --- | --- | --- |
| 0 | 0 | 0 | 本步无脏行，不能作为证据 |
| **>0** | 0 | 0 | join 是有效边（缺边存在于 join 之前） ✅ |
| **>0** | **>0** | **>0** | **join 没覆盖读者** ⇒ 真实缺边，需定位 join 位置/条件 |
| 任意 | 任意 | 任意，且 `frame` 对不上 | 跨流回调顺序问题，用 `t_us` 兜底 |

| rank | `pre_bcast` | `post_bcast` | 含义 |
| --- | --- | --- | --- |
| peer(≠0) | >0 | **0** | 广播把 peer 正确放行 ✅ |
| peer(≠0) | >0 | **>0** | **广播只是"到达信号"** ⇒ peer 侧真实缺边 |
| TP0 | 任意 | 0 | TP0 由 join 保证 |

> ⚠️ peer 上 `pre_bcast > 0` 是**正常**的（peer 不执行 join，`wait_for_current_kv_writeback` 在 `tp_rank != 0` 时直接 return），只有 `post_bcast > 0` 才是问题。
> ⚠️ 本分支**只支持 Mooncake**（MemFabric 的 Host 视图是 CPU/GVA，设备比较会直接 return）——若这次 decode 服务用 MemFabric，本实验无输出，请改用主干分支的 `hostview` 通路。

### 实验 D3 · 放行广播是不是"真边"（对照）

* **目的**：单独验证 `tp_group.broadcast(...)` 的必要性，以及"TP0 join 后广播"是否构成 peer 的 happens-before。
* **操作**：两轮对照，其余设置完全相同：
  * 轮 1：`VLLM_ASCEND_SFA_READY_BCAST=1`（默认，执行广播）
  * 轮 2：`VLLM_ASCEND_SFA_READY_BCAST=0`（不执行广播）
  比对 `[SFA_GRAPH_TRACE] host_kv_visibility_*` 与业务输出。
* **判读**：

| 观测 | 结论 |
| --- | --- |
| 轮 1：peer `post_bcast==0`；轮 2：peer 出现非零 / 精度变差 | **广播是必需的边**（peer 完全依赖它） ⇒ 任何"去掉广播"的优化都会被否决 |
| 两轮都没差异 | 广播在当前负载下不必要（可能是 planner 元数据广播顺带同步了），但**不能删**——它是 peer 唯一显式同步点 |
| 轮 1 也出现 `post_bcast>0` | 广播不足以构成边 ⇒ 需要在广播**之前**让 TP0 的 join 真正完成（检查 `wait_for_current_kv_writeback` 是否被 `capturing` 判断绕过） |

* **注意**：`[SFA_READY_BCAST]` 是 **eager-only** 日志（`not capturing` 门控），图模式只能看 `host_kv_visibility_*`。

### 实验 D4 · 哪些层根本不发集合通信（结构性论据）

* **目的**：证明"仅修 TP0 的 join 不够"——复用 plan 的层没有任何集合通信，peer 无从被放行。
* **操作**：grep `[SFA_PLAN_TRACE]`（capture 期也有）。
* **判读**：

| 观测 | 结论 |
| --- | --- |
| 某些 `layer_id` 出现 `DECISION ... can_reuse_owner_plan=True` + `REUSE_ENTER ... metadata_broadcast_will_run=False` | 该层**零集合通信**（复用层集合 = 每步层数的一部分，例如 4 层里 3 层复用） ⇒ peer 在这些层上只能靠"更早的同步点"约束，时序完全依赖前面的 broadcast/join |
| 所有层都是 `NORMAL_PLANNER_ENTER ... metadata_broadcast_will_run=True` | 每层都有集合通信 ⇒ 时序容错性更好，缺边更难暴露 |

* 结合 `planner_stats`（`window_refs>0`）一起看：**"有依赖 + 无集合通信"** 的层就是最高风险层。

### 实验 D5 · 依赖规模与静默丢 key（`planner_stats`）

* 判据与主干分支 §4 实验 0 完全一致：

| 观测 | 结论 |
| --- | --- |
| `window_refs > 0` | 同一步 Host 读真实存在（时序缺边有杀伤力） |
| 全程 `window_refs == 0` | 本 workload 下无同一步读 ⇒ 时序假设不适用，应转向长序列结构性疑点 |
| `window_dropped > 0` | 存在"静默丢 key"（plan 保持 0，算子读 Host 现值），是独立于时序的错误来源 |
| `dropped_misses > 0` | LRU 驱逐压力出现 |

---

## 6. 日志分析流程

```powershell
Set-Location <你的日志目录>

# 1) 图模式时序：只看非零落地计数（出问题的 step 就在这里）
Select-String -Path run_r*.err,run_r*.out -Pattern 'SFA_GRAPH_TRACE.*host_kv_visibility_.*values=\[(?!0\])'

# 2) 按 frame 拉时间线（与主干分支同一脚本）
Select-String -Path run_r0.err,run_r0.out -Pattern 'SFA_GRAPH_TRACE stage=(host_kv_visibility_\w+|planner_stats)' |
  ForEach-Object { $_.Line } |
  ForEach-Object {
    if ($_ -match 'stage=(\S+).*?frame=(\d+).*?t_us=(\d+).*?layer_id=(\d+).*?values=\[([^\]]*)\]') {
      [pscustomobject]@{ stage=$Matches[1]; frame=[int]$Matches[2]; t_us=[long]$Matches[3]; layer=[int]$Matches[4]; values=$Matches[5] }
    }
  } | Sort-Object frame,t_us | Format-Table -AutoSize

# 3) eager 比对结果与放行广播配对
Select-String -Path run_r*.err,run_r*.out -Pattern 'SFA_INDEX_COPY_PROBE\] (PASS|FAIL)'
Select-String -Path run_r*.err,run_r*.out -Pattern 'SFA_READY_BCAST\] (ENTER|RETURNED)'

# 4) 零集合通信的层
Select-String -Path run_r*.err,run_r*.out -Pattern 'SFA_PLAN_TRACE\] (DECISION|REUSE_ENTER|REUSE_RETURN)'
```

### 上报给定位同学的最小信息集

1. 服务型号（decode-only / 标准 PD）、`host_backend`、是否开了 store、镜像版本；
2. 复现的题目与 `MAX_TOKENS`、现象出现的大致 step/时刻；
3. 三个环境变量的取值（`…_INDEX_COPY_PROBE` / `…_PROBE_FAIL_FAST` / `…_READY_BCAST`）；
4. 每个 rank 的原始日志（`.err`/`.out`）；
5. `host_kv_visibility_*` 中非零的 `frame` 列表及其 5 个观测点取值；
6. `[SFA_PLAN_TRACE]` 中 `can_reuse_owner_plan=True` 的 `layer_id` 集合；
7. `planner_stats` 里出现过的最大 `window_refs` / `window_dropped` / `dropped_misses`。

---

## 7. 已知局限与陷阱

1. **eager-only 的探针**：`[SFA_INDEX_COPY_PROBE]`、`[SFA_CURRENT_HOST_READ]`、`[SFA_READY_BCAST]` 都带 `not capturing` 门控；**图模式下不产出**。图模式只能看 `[SFA_GRAPH_TRACE]`。
2. **`[SFA_GRAPH_TRACE]` 的层过滤**：打桩自带的 stage 只在 `SFA_GRAPH_TRACE_LAYER_ID = 4` 打印；`host_kv_visibility_*` 与 `planner_stats` 已豁免（每层都产出）。别把"某 stage 只在 layer 4 出现"当成异常。
3. **采样规则**：`call <= 4` ∥ 2 的幂 ∥ `call % 64 == 0` ∥ 计数非零。**非零（异常）永远打印，零值多数被丢**。所以"某 stage 没出现"≠"没执行"。
4. **`frame` 的边界歧义**：`frame` 由 `post_write`+`layer_id==0` 推进；`planner_stats` 在另一条流（fused plan 流）上，可能被打到相邻帧。同帧内判读不受影响，跨帧请以 `post_write` 锚点为准，并用 `t_us` 校验。
5. **peer 的 `pre_bcast>0` 正常**（见实验 D2 注意项）。
6. **MemFabric 下可见性探针无输出**：设备比较需要 NPU 可寻址的 Host 视图。MemFabric 请改用主干分支的 `hostview` 通路。
7. **`FAIL_FAST=1` 会中断请求**：短跑抓问题用它；长跑压测请设 `0`，否则第一次 `FAIL` 就 500。
8. **观测扰动**：探针引入额外 D2H 与算子（会改变时序）。对照轮次必须同开关设置。
9. **`71422442b` 的功能性改动**（§2）是无开关的：如果 Host 地址校验触发，服务会直接 raise。出现这种情况请记录完整堆栈并上报，不要简单 `git revert` 掉整个 commit（会连带丢掉探针）。
10. **探针非零 ≠ 业务出错**：单个 key 错误可能被 2048 路 softmax 稀释，需与业务指标联合判定。

---

## 8. 附录

### A. 探针代码索引（行号基于当前工作区快照）

| 文件 | 位置 | 内容 |
| --- | --- | --- |
| `vllm_ascend/distributed/kv_transfer/sparse_kv_offload/sparse_kv_offload_manager.py` | L85 `SFA_GRAPH_TRACE_LAYER_ID` | 打桩 stage 的层过滤值 |
| 同上 | L98 / L101 / L105 | `SFA_HOST_KV_VISIBILITY_STAGES`、帧起点 stage、`SFA_ADJ_PLANNER_STATS_LAYOUT` |
| 同上 | L1075 `self.index_copy_probe_enabled` | 开关与缓冲分配、`set_planner_probe_stats_buffer` 注册 |
| 同上 | `offload_new_kv`（≈L1417–1600） | `post_write` / `pre_bcast` / `post_bcast` 锚点、`[SFA_READY_BCAST]` |
| 同上 | `_trace_planner_stats`（L2014） | planner 统计入队 |
| 同上 | `trace_graph_host_kv_visibility`（L2026）/ `_probe_host_kv_visibility`（L2057） | 可见性探针入口 |
| 同上 | `_probe_mooncake_index_copy`（L2119） | eager 逐元素比对（PASS/FAIL/必要 raise） |
| 同上 | `_enqueue_graph_trace_cpu`（L2403） | `new_frame` 透传 |
| 同上 | `prepare_fused_overlap_external_plan`（L2380+） | `[SFA_PLAN_TRACE]` 决策日志 |
| 同上 | `inject_current_kv_into_selection`（L3017） | `all_tp_post_plan` 比对（受 `…_PROBE_FAIL_FAST` 控制） |
| 同上 | `wait_for_current_kv_writeback`（L3049） | 唯一的 join（**TP0-only**） |
| `vllm_ascend/attention/sfa_kv_offload.py` | L1036 / L1037 / L1038 / L1039 | `pre_join` → join → `post_join` → `fused_op`（顺序即修复内容） |
| `.../sparse_kv_offload.cpp` | `g_planner_probe_stats`（L62）、`process_one_lru_resident_row`（≈L120–295） | planner 计数 |
| 同上 | `graph_frame_id`（L769）/ `graph_trace_callback`（L928） | frame `=`/`t_us=` |
| 同上 | `enqueue_graph_trace_tensor`（L993）、`set_planner_probe_stats_buffer`（L1168） | pybind 入口 |

### B. 本分支包含的提交（理解"修复前/后"）

| 提交 | 作用 |
| --- | --- |
| `193da4f99` | temp：CXX/CC 改 clang→gcc 等临时改动 |
| `c806ec153` | 单独拉起 Mooncake decode 卸载（配置放宽、MTP descriptor 刷新、prefill KV staging） |
| `71422442b` | **打桩**：图/eager Mooncake 路径 trace（见 §2，含无开关的功能性校验） |
| `3fa38c72b` | UT 用 g++（与运行时无关） |
| `31d2e6edc` | 让 Host-KV 可见性探针在**图模式**下可用（本分支图模式计数的来源） |
| `2e9f6194d` | **修复**：把图模式 Host-KV 写回 join 提到读者之前（`pre_join → join → post_join → fused_op`） |

### C. 判定速查

| `post_write` | `pre_join` | peer `post_bcast` | `planner_stats.window_refs` | 结论 |
| --- | --- | --- | --- | --- |
| >0 | 0 | 0 | >0 | **缺边确实存在于 join 之前，修复有效** |
| >0 | 0 | >0 | >0 | **peer 侧广播不是落地边** ⇒ 需在 peer 侧补边 |
| >0 | >0 | — | >0 | join 覆盖不到读者 ⇒ 查 join 位置 / `tp_rank` 条件 / `capturing` 判断 |
| 0 | 0 | 0 | >0 | 依赖存在但未观测到脏行 ⇒ 时序窗口极小，加长序列或提高探针频率 |
| 任何 | 任何 | 任何 | **==0** | 本 workload 下同一步不读 Host ⇒ 时序假设不适用，转向其它候选（长序列结构性疑点） |

### D. 与主干分支的分工建议

| 问题 | 建议在哪个分支做 |
| --- | --- |
| decode-only 服务是否仍需加固（本手册 D1–D4） | **本分支** |
| MemFabric 路 H2H 视图差分、`sparse_copy` 完成语义 | 主干分支（本分支无 hostview 探针） |
| MTP vs layer0 slot_mapping 差异 | 主干分支（有 `slot_mapping_l0/mtp` 探针） |
| 长序列结构性疑点 | 两分支都需跑，缺一不可 |
