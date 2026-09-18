# decode-only 分支代码修正说明（探针 + 产品修复移植）

> 审查结论见 `260917_decode-only分支代码审查.md`；标准分支侧的修复见 `260917_图模式HostKV时序问题_代码修正说明.md`
> 修正日期：2026-09-17
> 工作区：`d:\hanjiang\0917mayi材料\repo\vllm-ascend-decode-fix`
> 基线：decode-only 分支 HEAD `3fa38c72b`（= `dsa_offload_rebase_pr15642_0911_decode_test`，**含打桩提交 `71422442b`**）
> 分支：`fix/decode-graph-kv-probe`
> 补丁：`0001-debug-sparse-kv-make-the-Host-KV-visibility-probe-wo.patch`、`0002-fix-sparse-kv-order-graph-mode-Host-KV-writeback-bef.patch`

```text
commit 1  31d2e6edc  debug(sparse-kv): make the Host-KV visibility probe work in graph mode
commit 2  2e9f6194d  fix(sparse-kv): order graph-mode Host-KV writeback before its readers
改动量：  commit 1 = 3 文件 +121 / -5；commit 2 = 3 文件 +58 / -10
```

---

## 一、commit 1：把打桩从"图模式看不到"改成"看得见"

### 1.1 针对审查文档 §3 的四个缺陷

| 审查缺陷 | 本次如何解决 |
| :--- | :--- |
| **A** 图模式下 KV 一致性探针被整体关闭（调用点带 `if not capturing`） | 新增**图模式专用**探针 `trace_graph_host_kv_visibility()`，与 eager 探针并存：eager 用 `.cpu()/torch.equal` 的强断言（保持原样），图模式用纯设备算子实现（新增） |
| **B** 图模式唯一比对位于 save 流内部，与写回同流有序 | 新探针**挂在 compute 流上、夹住写回归结点**：`pre_join` / `post_join` 两个观测点 |
| **C** trace 通道采样丢帧、异常被吞 | C++ 侧 `graph_trace_callback` 增加"**单元素且非零 ⇒ 无视采样必然输出**"；`host_kv_visibility_*` 的 stage 绕过"仅第 4 层"的过滤，保证任意层触发都能被看到 |
| **D** 打桩改变被测对象 | 探针默认关闭（env 控制）；输出改为"零值基本静默、非零必报"，把扰动压到最小；验证纪律见 §3 |

### 1.2 代码改动清单（commit 1）

| 文件 | 改动 |
| :--- | :--- |
| `sparse_kv_offload_manager.py` | ① 新增模块常量 `SFA_HOST_KV_VISIBILITY_STAGES = ("pre_join", "post_join")`；② `__init__` 新增设备计数器 `graph_host_kv_visibility_npu`（2 槽，仅探针开启时创建）；③ `offload_new_kv` 把 `slot_mapping` 缓存改为**两种模式都填**（图模式下它是对静态输入缓冲的引用，replay 时自动是当帧值）；④ `_enqueue_graph_trace_cpu` 放行 `host_kv_visibility_` 前缀的 stage；⑤ 新增探针本体 |
| `sparse_kv_offload.cpp` | `graph_trace_callback` 增加非零单元素计数器强制输出 |
| `sfa_kv_offload.py` | 在 `wait_for_current_kv_writeback` 前后各插一次探针调用 |
| `tests/ut/attention/test_sfa_kv_offload.py` | 为 SimpleNamespace manager 补 `trace_graph_host_kv_visibility` stub |

### 1.3 探针语义（关键）

```python
        manager.trace_graph_host_kv_visibility(layer_name, stage="pre_join")
        manager.wait_for_current_kv_writeback(get_forward_context().capturing)
        manager.trace_graph_host_kv_visibility(layer_name, stage="post_join")
        attn_output = fused_op(**fused_inputs)
```

它统计的是：**当帧每个真实 token 的 Host 行内容，是否等于当帧为它算出的 K/V**（bf16 用 int16 视图逐 bit 比较，K/V 两部分取或）。

- `pre_join`：位于 `wait` 之前。**只要它非零，就证明融合算子有可能读到未写完的行**（E1：TP0 自身；E2：TP1-7 被提前放行）——因为没有 join 时算子就在这个位置读；
- `post_join`：位于 `wait` 之后。**必须恒为 0**，否则说明 join 没能覆盖写回（修复无效或写错行）；
- 两个观测点都**只做设备算子**（`index_select` / `!=` / `any` / `count_nonzero` / `add_`），不触发同步，因此**在 graph capture 里合法**；
- 计数器按 forward step 归零（在第 0 层重置），所以"某帧非零"是可归因的，不会累积成永久脏值；
- **所有 TP rank 都会跑**：decode K/V 在各 rank 上复制，所以"本 rank 本地视角的 Host 行 vs 本 rank 自己算出的 K/V"这个**纯本地比较**就能发现"TP0 写了但我看不到"。

### 1.4 怎么用

```bash
# 启动 decode 服务时打开探针（默认关闭）
export VLLM_ASCEND_SFA_INDEX_COPY_PROBE=1
# 可选：默认 1，探针不一致时 fail-fast（eager 路径）
export VLLM_ASCEND_SFA_PROBE_FAIL_FAST=1
```

输出落在**进程 stderr**（即 serve 脚本里配置的 debug 输出），前缀 `[SFA_GRAPH_TRACE]`，由后台线程异步打印：

```text
[SFA_GRAPH_TRACE] stage=host_kv_visibility_pre_join call=7 layer_id=3 tp_rank=0 dtype=4 numel=1 hash=... values=[2]
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^                              ^^^^^  ^^^^^^^^
                   stage                                       replay 第几次调用   本帧本层到该点为止，未写完的行数
[SFA_GRAPH_TRACE] dropped=12          ← 出现这行说明队列溢出丢过帧，别用"没看到"作结论
```

判读规则：

| 观察 | 结论 |
| :--- | :--- |
| `pre_join` 出现非零（未打产品修复时） | **E1/E2 命中**：算子会读到未写完/旧数据 |
| `pre_join` 非零、`post_join` 全 0 | 竞态窗口真实存在，且当前等待点能覆盖它 |
| `post_join` 出现非零 | 修复无效，或 Host 行写错了位置（转 MTP 描述符/写回地址方向） |
| 两者全 0 且 `hash` 恒定 | 本帧没有竞态（注意：单帧为 0 不能证明全部帧无竞态，需长跑统计） |

---

## 二、commit 2：移植产品修复（E1/E2）

decode-only 分支的 **MTP 描述符刷新已经存在**（`c806ec153`），所以只移植其余三条 + 一条注释：

| # | 改动 | 位置 |
| :--- | :--- | :--- |
| 1 | `wait_for_current_kv_writeback` 从算子**之后**移到**之前**（E1：TP0 自身读到本 step 未写完的行） | `sfa_kv_offload.py` |
| 2 | TP0 进入 ready broadcast **之前**先 join 自己的写回流（E2：TP1-7 被提前放行） | `sparse_kv_offload_manager.py` `offload_new_kv` |
| 3 | 新增 `current_kv_writeback_on_side_stream` 状态位，等待条件改为 `capturing or flag`（修 `_in_graph_runtime()` 与 `forward_context.capturing` 不一致时的漏等） | 同文件 3 处 |
| 4 | 为 standalone Prefill 灌 Host 池补注释，说明**为什么不需要额外 TP 同步边**（审查文档 §2.2 的澄清项，不是 bug） | 同文件 `_offload_new_kv_on_current_stream` |
| 5 | 更新单测把"等待在算子之后"的旧断言改成新顺序 | `tests/ut/attention/test_sfa_kv_offload.py` |

> 与标准分支那份补丁的差别：本分支的 `offload_new_kv` 多了 `VLLM_ASCEND_SFA_READY_BCAST` 环境开关与 `[SFA_READY_BCAST]` 探针日志，`git am` 会因上下文不同而冲突；两边的补丁**不要混用**。

---

## 三、A/B 验证矩阵（建议三臂）

```text
臂①  基线：            git checkout 3fa38c72b
臂②  只打探针：        git checkout 31d2e6edc        （commit 1）
臂③  探针 + 产品修复： git checkout 2e9f6194d        （commit 2，= 分支 HEAD）

三臂都用 VLLM_ASCEND_SFA_INDEX_COPY_PROBE=1 跑同一组短序列图模式用例，比较：
  臂①  └ 无探针（无输出）
  臂②  └ 期望 pre_join 出现非零  ⇒ 直接证明"算子会读到未写完的行"
  臂③  └ 期望 pre_join 可能仍非零（竞态窗口仍在），但 post_join 恒 0，且复读率下降
  另加：同臂③但关掉探针，确认复读率一致（排除探针自身影响）
```

需要采集的数据：每个 stage 的**非零出现次数 / 出现在哪些层 / 出现在哪些 rank / 每帧计数分布** —— 这些足以把 E1 与 E2 分开（E1 只在 `tp_rank=0` 出现，E2 在所有 rank 出现）。

---

## 四、注意事项

1. **不要再 `revert 71422442b`**：commit 1 的改动就在这份打桩代码上，revert 会与它冲突。本分支把打桩改成了"默认关闭 + 图模式可用"，直接用它跑服务即可。若确实需要一份"完全不含打桩代码"的 release 分支，告诉我，我基于 `c806ec153` 单独出一个（只带改进后的探针）。
2. 探针开启会**增加设备节点**（每层 2 次比较 + 2 次 D2H），对时序敏感问题属"观测即扰动"；因此**必须三臂都用同样的探针设置对比**，只切换产品修复。
3. `[SFA_GRAPH_TRACE] dropped=` 出现时不可用"没看到"下结论。
4. 编译器：本分支已是 `g++/gcc`（`193da4f99`）；**标准分支仍是 `clang++`**，移植补丁时留意。
5. 本地只做了 `py_compile` 与 IDE 诊断校验（无 NPU 环境），未执行真实用例；请在有 NPU 的容器里跑 `pytest tests/ut/attention/test_sfa_kv_offload.py tests/ut/kv_offload/ -x`。
