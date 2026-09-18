# decode-only 服务分支代码审查

> 审查对象：`dsa_offload_rebase_pr15642_0911_decode_test`（HEAD `3fa38c72b`，任务清单里"decode only 服务"所用分支）
> 对照物：标准分支 `dsa_offload_rebase_pr15642_0911`（HEAD `ea8b8c1`）
> 审查方式：把这 4 个"分支独有提交"逐个拆开看，区分**运行态代码**与**打桩代码**；再逐行审打桩机制的可见性与扰动
> 审查日期：2026-09-17
> 关联文档：`260917_短序列图模式精度问题定位_复核与新增候选.md`、`260917_图模式HostKV时序问题_代码修正说明.md`

---

## 0. 结论摘要（TL;DR）

| 部分 | 行数 | 结论 |
| :--- | :--- | :--- |
| decode-only 分支**真正会跑到**的改动（`193da4f99` + `c806ec153`） | **124 行** | 逻辑基本正确。3 条发现：config 围栏完整 ✅、prefill 灌 Host 池缺 TP 同步边（当前用法不可达，属理论缺口）、编译器由 clang 改 gcc（标准分支没有，有移植风险） |
| decode-only 分支的**打桩代码**（`71422442b`） | **1328 行（占该分支独有改动的 91%）** | ⚠️ 有 **4 个结构性缺陷**，导致它在**图模式下几乎看不到东西**，而问题恰恰只在图模式出现 |
| 对既有结论的影响 | — | "打桩看数据一致"**不能**用来排除图模式时序问题；但同时，**eager 侧的探针是有效且实测通过的**，这反过来证明了 eager 的写与跨 rank 可见性没问题 |
| 是否要重修标准分支的同类问题 | — | 打桩缺陷是"定位能力"问题，不影响产品逻辑；本次已在新的 decode 分支工作区把它修成"图模式可用" |

---

## 1. 拆清楚：哪些改动真的会跑

`decode_test` = 标准分支 `ea8b8c1` + 以下 4 个提交：

| 提交 | 改动量 | 是否进入运行态 | 内容 |
| :--- | :--- | :--- | :--- |
| `193da4f99` | 5 行 | ✅ 会 | `_build_cpp` 的 `CXX/CC` 由 `clang++/clang` 改为 `g++/gcc`；`patch/recompute_proxy.py` 多 pop 一个 `prompt_token_ids` |
| `c806ec153` | 119 行 | ✅ 会 | standalone Mooncake decode：放宽 config 守卫 + MTP 描述符刷新 + 新增 prefill KV 灌 Host 池 |
| `71422442b` | **1328 行** | ❌（按 `task.md` 需去掉） | 全部探针 / trace / 断言 |
| `3fa38c72b` | 4 行 | ❌（仅 UT） | UT 里 `clang++` → `g++` |

> 结论：**"decode only 分支的代码"有 91% 是打桩**。这 1328 行不参与精度计算，但它们决定了"能看到什么、看不到什么"。

---

## 2. 运行态代码审查（124 行）

### 2.1 ✅ `ascend_config.py` 放宽 `keep_device_kv_cache` 守卫：逻辑正确、围栏完整

```python
            if (
                self.keep_device_kv_cache
                and getattr(vllm_config, "kv_transfer_config", None) is not None
            ):
                raise ValueError(
                    "sparse_kv_offload_config.host_backend='mooncake' with "
                    "keep_device_kv_cache=true is only supported for standalone "
                    "debug without kv_transfer_config"
                )
```

- 语义：**只在"没有 connector 的 standalone 场景"放行**，PD（有 `kv_transfer_config`）依旧拒绝；
- 新增 UT `test_mooncake_host_backend_rejects_connector_colocate_staging` 正是用来固定这个边界；
- 配对约束也仍然有效：`_offload_new_kv_on_current_stream` 里 `has_prefill and not keep_device_kv_cache` → 抛错，所以"prefill 灌 Host 池"这条新路径在 PD 场景不可达。

**判断：无需修改。**

### 2.2 ⚠️ `_offload_prefill_kv_via_index_copy`：缺 TP 同步边（理论缺口，当前不可达）

新增函数负责"standalone 下把本地 NPU paged KV 整片搬进共享 Host 池"。它写共享 Host 池（TP0 写、TP1-7 读），但 **`offload_new_kv` 里那条 ready broadcast 的条件含 `not has_prefill`**，也就是说 prefill 阶段**没有任何 TP 间同步边**。

为什么当前不可达（因此不必改）：

1. 融合算子只在 `_is_decode_only(attn_metadata)` 为真时被调用（要求 `num_prefills == 0`），所以**同一个 step 内不会有"prefill 写 + fused op 读"**；
2. prefill 的写发生在 TP0 的 compute 流上，而**下一个 decode step 的 ready broadcast 也在 TP0 的 compute 流上、且在其之后**（同一流天然有序）⇒ TP1-7 在那个 broadcast 上被正确卡住；
3. 于是"prefill 写 → TP1-7 读"的顺序由"下一个 decode step 的 broadcast"兜住。

**判断：不需要加（额外加一条 collective 只会引入死锁风险）。本次仅在代码里补注释说明该不变式。**

### 2.3 ⚠️ `_build_cpp` 编译器改动：标准分支没有，存在移植风险

```text
标准分支 ea8b8c1  _build_cpp:680-681   CXX=clang++  CC=clang
decode 分支 193da4f99                   CXX=g++      CC=gcc     ← 提交信息标着 "temp"
UT 3fa38c72b  shutil.which("clang++") → shutil.which("g++")
```

UT 都跟着改了，说明**在 0.26 镜像下 gcc 才是更可靠/唯一可用的路径**。

**判断：这不是"bug"，但移植标准分支的补丁时要注意**——标准分支（以及本次基于标准分支的修复分支）仍写死 clang，若编译 planner helper 失败，需要把这两行一起改掉。

### 2.4 `recompute_proxy.py` 的 `prompt_token_ids` pop

改的是**响应体**（把 `prompt_token_ids` 从每个 choice 里去掉），与精度无关，但要知道它改变了对外协议；调试期无妨。

---

## 3. 🔴 打桩机制审查：4 个结构性缺陷

### 缺陷 A：图模式下 KV 一致性探针被整体关闭

`_probe_mooncake_index_copy` 的两个调用点都带 `if not capturing` 守卫：

```text
sparse_kv_offload_manager.py  inject_current_kv_into_selection()  -> if not capturing: _probe(..., stage="all_tp_post_plan")
sparse_kv_offload_manager.py  offload_new_kv()                    -> if tp_rank == 0 and not capturing: _probe(..., stage="tp0_post_write")
```

⇒ **问题只在图模式出现，而探针的盲区恰好完全覆盖问题区间。**

### 缺陷 B：图模式里唯一的写回比对位于 save 流内部

capturing 分支中被录制进图的读回比对（`index_copy_host_mismatch_kv`）**写在 `with torch_npu.npu.stream(self.current_kv_save_stream)` 块内**：

```python
            with torch_npu.npu.stream(self.current_kv_save_stream):
                self.current_kv_save_stream.wait_event(current_kv_ready)
                flat_host_k.index_copy_(...)          # 写回
                flat_host_v.index_copy_(...)
                if getattr(self, "index_copy_probe_enabled", False):
                    actual_k = flat_host_k.index_select(0, self.d2h_dst_idx_npu)   # ← 与写回同流
                    ...
                    torch.count_nonzero(actual_k.view(torch.int16) != expected_k.view(torch.int16))
```

它与写回**同流有序** ⇒ 结构上**不可能**发现"读得太早"，只能证明"写本身写对了"。
**"图模式打桩显示 host KV 一致"这一结论就是这个盲点造成的。**

### 缺陷 C：trace 通道本身不可靠且大量丢帧

`enqueue_graph_trace_tensor`（`sparse_kv_offload.cpp:889`）：

1. D2H 与 host callback 之间用的是**裸 `aclrtLaunchHostFunc`**（与复核文档 §3.2.3 的 S1 同一平台疑点）——若该原语不保证"前置 D2H 的主机可见性"，trace 出来的 hash 就是**旧值**；
2. `graph_trace_callback` 是 `noexcept` + `try/catch`，失败只 push 一行 `callback_failed`；**不会报错**；
3. 采样策略 `call <= 4 || (call & (call-1)) == 0 || call % 64 == 0` ⇒ 前 4 次、2 的幂次、每 64 次才记录，**多数帧不记录**，无法逐帧对照。

⇒ "hash 看起来是旧值"既可能是被测对象的真实状态，也可能是 trace 通道自己的问题，**两者不可区分**。

### 缺陷 D：打桩本身改变被测对象（观测即扰动）

capturing 分支被录进图、逐帧 replay 的节点，在开启探针后多了：`index_select` × 2、`count_nonzero`、一次 D2H、一次 host callback。对时序敏感的竞态来说，这是**在被观测路径上增加了设备工作与流节点**。

⇒ "加上打桩后复读频率变了""去掉打桩就复现"**不能作为证据**；正确做法是"探针全程开启，只切换被测变量"。

### 补充：探针的 slot_mapping 缓存只在 eager 填充

```python
            if getattr(self, "index_copy_probe_enabled", False):
                self.graph_trace_capture_layer_id = layer_id
                if not capturing:
                    self.index_copy_probe_slots_by_layer[layer_id] = slot_mapping
```

⇒ 设计上就注定它**无法**在图模式复用（这是缺陷 A 的技术原因之一，修复时要换数据来源，见 §4）。

---

## 4. 一个反向的重要结论：eager 探针是有效且实测通过的

`index_copy_probe_enabled` 由 env 控制、**默认关闭**：

```python
self.index_copy_probe_enabled = os.getenv("VLLM_ASCEND_SFA_INDEX_COPY_PROBE", "0").lower() in ("1","true","yes","on")
```

一旦开启，eager 下会做两类断言，且 `raise_on_failure` 默认 `1`（`VLLM_ASCEND_SFA_PROBE_FAIL_FAST`）：

| stage | 执行者 | 断言内容 |
| :--- | :--- | :--- |
| `tp0_post_write` | 仅 TP0 | 写回后立即回读：Host 行内容 == 本 step 的 K/V（bit 级，用 int16 视图比较） |
| `all_tp_post_plan` | **所有 rank** | plan 之后、算子之前：**本 rank 的本地 Device-VA 视图**上的 Host 行 == 本 rank 自己算出的 K/V |

第 2 条尤其有价值：decode K/V 在各 TP rank 上是复制的，所以 TP1-7 可以用**纯本地比较**验证"TP0 刚写的行我能不能看到"——这正是跨 rank 可见性检查。

⇒ 由此可得两条推断：

1. **"eager 正常"是有硬证据的**，不是"没看到问题"；
2. **eager 侧残余的"极其偶发段落重复"不可能来自 Host 池的写入或可见性**（已被断言覆盖），只可能来自 **plan 内容 / 静默丢键 / 算子读取逻辑 / P→D 传输** —— 这直接抬高了复核文档 §4 里那几个候选的优先级。

---

## 5. 改进设计（把"看不到"变成"看得见"）

### 5.1 三条改动

```text
① 位置：在 compute 流上、fused_op 之前（以及 wait 之后）各做一次"Host 行 vs 本 step K/V"比较
   └ 现有比对在 save 流内，物理上抓不到"读得太早"；挪到 compute 流才能测出算子实际会看到什么
② 形态：全部用设备算子 + 不 sync，图模式才合法
   - 取数用 slot_mapping（device 张量）而不是 Python 侧的 list
   - 比较：index_select + int16 视图 != + any(dim=-1) + count_nonzero → 写入常驻 device 计数器
   - 回读：用已有 _enqueue_graph_trace_cpu（D2H + host callback，都在当前流上）
   └ 关键：所有 rank 都能做（K/V 跨 rank 复制），所以同一机制天然覆盖 E1（TP0 自身）与 E2（TP1-7）
③ 通道：让"非零计数"永远被记录，且 callback 异常不再静默
   - graph_trace_callback 增加"单元素且非零 ⇒ 无视采样策略强制输出"
```

### 5.2 为什么必须用"设备端比较"而不是 `.cpu()`

图捕获期间不允许同步（`.item()`/`.cpu()`/`torch.equal` 都会触发同步），一旦在捕获期调用会直接报错或破坏图。所以图模式探针只能：

```text
device 比较 → device 计数器 → 当前流 D2H → host callback 打印
```

### 5.3 实验纪律（避免缺陷 D 重演）

```text
- 探针全程保持开启，只切换"是否打产品修复"，两臂的扰动相同
- 关注指标：每个 stage 的非零计数（>0 就是"读到未写完数据"的直接证据）
- 建议顺序：pre_wait 计数 > 0 且 post_wait 计数 == 0 ⇒ 竞态确诊且修复有效
```

---

## 6. 本次落地（详见 `260917_decode-only分支_代码修正说明.md`）

```text
工作区： d:\hanjiang\0917mayi材料\repo\vllm-ascend-decode-fix
基线：   decode_test  HEAD 3fa38c72b
分支：   fix/decode-graph-kv-probe
commit 1：图模式可用的 Host-KV 可见性探针（修 §3 缺陷 A/B/C/D）
commit 2：移植 E1/E2 产品修复（等待前移 + TP0 broadcast 前 join save 流 + 写回状态位）
```

### 未修项

| 项 | 理由 |
| :--- | :--- |
| §2.2 prefill 缺 TP 同步边 | 当前用法不可达（融合算子只在 `num_prefills==0` 时调用），且顺序由"下一个 decode step 的 broadcast"兜住；额外加 collective 只增加死锁风险。改为在代码里补注释说明该不变式 |
| §2.3 编译器 clang/gcc | decode 分支已经是 gcc；需要在**标准分支**侧注意（见 §2.3），不由本分支处理 |
| §2.4 响应体改动 | 与精度无关，调试期保留 |
