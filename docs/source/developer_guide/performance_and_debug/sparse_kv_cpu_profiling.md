# Sparse KV CPU profiling：TP=2、DP=16 与受限 Kubernetes

本指南针对当前 fused-overlap、已捕获图中的 CPU LRU planner。每个 TP 组只有
TP0 执行 planner；TP1 接收计划。TP=2、DP=16 的部署共有 16 个 planner 执行者。
当一个 Pod 承载 TP=2、DP=8、16 张 NPU、128 CPU 时，要研究的是该 Pod 中
8 个 planner 及其他运行时线程之间的竞争。128 CPU 并不是每个 planner 的预算。
当前分支 `lru_workspace_threads=8`，活跃线程数还受图内行数限制。因此每 Pod 的
planner 并发线程合计最多约 64 个；仍需加上 TP1、通信等线程，不能直接推出“没有限流”。

本工具不覆盖 eager planner、地址计算的其他 OpenMP 循环、全部 KV 搬运或 NPU
实际等待时间。捕获后新增的 graph payload 不属于已开启的窗口，需要重新开始采集。
不要仅因为某个 DP 的 callback 较慢，就断言其他 DP 一定被它阻塞；跨 DP 的
通信依赖、请求工作量和调用序号可能不同。

## 1. 首次采集：不需要宿主机权限

在所有相关 worker 启动前设置：

```bash
export VLLM_ASCEND_PLANNER_TRACE_ONLY=1
```

这是 `vllm_ascend/envs.py` 中集中定义的非敏感调试开关，合法值 0/1，默认 0。
1 表示使用现有 profiler 窗口，但跳过 torch/NPU profiler；0 保留组合采集。
不会修改线程数、CPU 绑定、OpenMP 等待策略或任务分配，也不新增逐层同步。
不要同时开启会逐层同步的 `VLLM_ASCEND_DEBUG_SFA_SYNC_TIMING`。

在原来的服务命令上添加已有配置：

```text
--profiler-config '{"profiler":"torch","torch_profiler_dir":"/tmp/planner-profile","torch_profiler_with_stack":false}'
```

1. 固定请求集、并发、输入长度、解码长度及线程配置。
2. 完成模型加载、图捕获和 OpenMP 预热，再开始窗口。
3. 通过已有的 profiling 路由开启采集，运行有代表性的解码，随后停止。
4. 对每个分片执行已有服务的 profiling 控制流程；不要假定负载均衡器转发了所有控制请求。
5. 一个 Pod 中预期每个参与且执行 callback 的 DP TP0 都有输出；检查 DP 标识，不能仅数文件。

```bash
curl -X POST http://localhost:8080/start_profile
# 在另一终端运行固定的请求负载。
curl -X POST http://localhost:8080/stop_profile
```

端口替换成实际服务端口；控制方式详见 [Service Profiling Guide](service_profiling_guide.md)。

每个 TP0 导出 `*_lru_callback.json`，包含 DP/TP 标识、进程与线程 ID、
`CLOCK_MONOTONIC` 时间、图/层/本地调用序号、线程工作量及系统快照。
默认每个 graph payload 保存最先 256 次调用；超出部分计入 `dropped`，不覆盖旧记录。
先选择短窗口；若有丢弃则缩短并重复。环境变量 `VLLM_ASCEND_PLANNER_TRACE_CAPACITY`
可以调整容量（默认 256，合法范围 1–65536，非敏感），底层接口为
`debug_start_planner_trace(capacity=256)`。内存随 payload 数 × 容量 × 线程数增长。
捕获了很多图形状/层时，建议先设为 32 并运行短窗口，检查 memory.current/内存压力后再增大。
8 线程、256 次调用的线程记录约占每个 payload 256 KiB，另有调用记录和可选 OMPT 存储；
所有已捕获 payload 都会预分配，不能仅按当前正在使用的图估计总内存。
不要为了增加容量在测量期间动态扩容。

开始/结束需要在窗口边界 drain NPU stream，以便安全重置、导出记录；这会影响边界附近
行为。排除最初和最后的过渡样本，比较稳定区间。系统快照只在边界读取，不在逐行循环中读取。
每线程仅在工作边界读时钟/CPU ID，行循环只累计行数与 miss 数。
这些操作仍有扰动，不能承诺“绝对无开销”或固定百分比的准确性。

## 2. 输出含义与离线分析

在仓库根目录执行；脚本只需要 Python 标准库，不需要 torch/NPU：

```bash
python vllm_ascend/profiler/sparse_kv_cpu.py analyze \
  /tmp/planner-profile/ACTUAL_lru_callback.json \
  --output /tmp/planner-analysis.json \
  --chrome /tmp/planner-timeline.json
```

Chrome trace JSON 可以在 Perfetto 中打开。每个文件保留自己的主机/进程身份；
不同主机的单调时钟不能直接拼接。`graph_id/layer_id/call_index` 只在当前进程、当前
payload 和采集窗口内有意义，不是跨 DP 的统一解码步号。

| 字段 | 含义和限制 |
| --- | --- |
| `callback_ns` | 被记录 callback 的耗时，包括记录操作；不包括 callback 开始前的流排队 |
| `serial_setup_ns` | callback 进入到并行/串行工作阶段开始，包含准备和少量记录操作 |
| `entry_delay_ns` | 并行阶段开始到线程进入记录点；不能单独区分唤醒、运行时与 OS 延迟 |
| `work_wall_ns` | 线程处理所分配行的经过时间 |
| `work_cpu_ns` | 同一区间附近采样的线程 CPU 时间；包括内存停顿、自旋和内核执行 |
| `wall_minus_cpu_ns` | 线索，不是精确调度时间；时钟采样边界不同，极短区间可略为负数 |
| `tail_to_parallel_return_ns` | 工作结束到主线程离开并行区；不是精确 barrier 等待时间 |
| `observed_wait_ns` | OMPT 实际观察到的完整等待区间之和；需同时看 `wait_complete` |
| `observed_wait_in_parallel_ns` | 完整等待区间与当前并行阶段的交集，排除主线程返回后的等待 |
| `rows/misses` | 实际分配行数及累计 miss 数；用于排除工作量差异，不代表全部指令数 |
| `cpu_begin/cpu_end` | 工作两端 CPU ID；相同不代表中间没有迁移 |
| `last_worker_tid` | 最晚完成工作者，不一定是唯一瓶颈或等待最长者 |

建议先按 callback 长尾排序，比较同层、相近行数/top-k/miss 数的样本，再查看线程分布。
不同 DP 的请求内容、缓存命中、首次初始化等仍可能不同，不应只比较一个平均值。
`num_reqs` 和 `rows` 是 planner 实际处理的图内行数，可能包含填充，不一定等于活跃请求数。

## 3. 可选 OMPT：精确观察同步等待区间

基础采集无需 OMPT。要启用同步等待，使用目标镜像的 C++ 工具链和与 planner 相同的
C++ 标准库编译可选工具。需要 LLVM/OpenMP 提供的 `omp-tools.h`；不要用编译器版本
推断运行时支持，也不要把另一个 OpenMP runtime 注入服务以“补足支持”。

```bash
clang++ -std=c++17 -O2 -fPIC -shared \
  -I/path/to/directory/containing/omp-tools.h \
  tools/profiler/planner_ompt.cpp -o /tmp/libascend_planner_ompt.so
```

启动服务之前设置，覆盖到实际 worker 进程：

```bash
export OMP_TOOL=enabled
export OMP_TOOL_LIBRARIES=/tmp/libascend_planner_ompt.so
export LD_PRELOAD="/tmp/libascend_planner_ompt.so${LD_PRELOAD:+:$LD_PRELOAD}"
```

`LD_PRELOAD` 使版本化的记录桥接符号对 JIT 扩展可见；OMPT 必须在 OpenMP 初始化时接入。
仅在已经运行的服务中修改环境变量无效。已有其他 OMPT 工具时，请分开运行，不假定两者可组合。
工具不链接第二个 OpenMP runtime；同一进程存在多个 runtime 时不保证都能接入。

`ompt_support` 保存运行时对同步等待回调的支持结果：

| 值 | 含义 |
| --- | --- |
| 0 | 无可用桥接/工具、必要的生命周期回调不完整，或此调用是串行路径 |
| 1 / 2 | 注册错误 / 不支持该等待回调 |
| 3 / 4 | 部分事件 / 成对但不完整的事件；不能声称完整等待时间 |
| 5 | 运行时承诺报告全部相关事件；仍需检查记录是否完整 |

工具只绑定被采集 planner 的最终并行 barrier，记录 `ompt_callback_sync_region_wait`。
等待包括自旋或睡眠，不能单独说明“线程没有占 CPU”。某些 worker 的 wait-end
可能晚于主线程离开并行区；导出时仍未结束的区间保留 `end=0`，不能当作零等待。
完整区间也可能延伸到下一次并行任务，包含主线程返回后的运行时空闲；分析当前 callback
的延迟时使用与并行阶段的交集，不能把整段运行时等待全部归给当前 callback。
每线程每调用保留最多 4 段，额外事件计入 `ompt_dropped`。

参考：[OpenMP implicit barrier 事件](https://www.openmp.org/spec-html/5.1/openmpsu101.html)、
[回调支持约定](https://www.openmp.org/spec-html/5.1/openmpsu207.html)。

## 4. Kubernetes 检查清单：先排除限流与资源重叠

在目标容器中找到真正的 worker PID；不是 API server PID，也不是交互 shell PID：

```bash
worker_pid=12345  # 替换成目标 TP0 worker
python vllm_ascend/profiler/sparse_kv_cpu.py snapshot \
  --pid "$worker_pid" --memory --output /tmp/worker-snapshot.json
```

脚本解析目标进程的 cgroup membership 与 mountinfo，兼容 v1/v2 及 subtree mount，
读取可见祖先。权限错误保留在 JSON 中；不可见的宿主机祖先限制不会被当成不存在。
`hostname` 通常是容器/Pod 名，不一定是物理节点名称。

检查顺序：

1. `identity.Cpus_allowed_list` 与各线程的 CPU 集合：是否意外只允许少数 CPU？
2. `cpuset.cpus.effective`：cgroup 实际 CPU 集合。可见 CPU 多不代表独占。
3. `cpu.max` 或 v1 quota/period：确认实际限制，而不是仅看 YAML。
4. `cpu.stat` 的窗口增量：是否出现限流？这是共享 cgroup 的统计，不是单个线程等待。
5. `cpu.pressure`、线程 `schedstat`：查看竞争线索；schedstat 可能被禁用，零值不能自动证明没有排队。
6. `memory.events`、`memory.pressure`、页错误：排除内存回收或缺页造成的停顿。
7. 记录同 Pod 的 8 个 TP0、8 个 TP1 与通信等线程的 CPU 集合是否重叠。

128 CPU request/limit 不等于 8 个 planner 各有 16 个独占 CPU。8 × 16 个同时运行的
planner 线程已经达到 128，其他线程仍需要 CPU。128 CPU quota 也不保证没有瞬时限流。
只有在相应 Kubernetes CPU Manager 策略和资源条件满足时，容器才获得独占 CPU。

| DP | TP0 PID | 物理节点/Pod | CPU 集合 | quota | 内存允许节点 | 工作区实际页节点 |
| --- | --- | --- | --- | --- | --- | --- |
| 实际 DP ID | 从记录读取 | 从部署信息补充 | 从线程读取 | 从 cgroup 读取 | 从 cpuset 读取 | 从映射读取 |

该表填写 8 行；第二个 Pod 单独填写。不要假定 DP 0–7 一定在第一个 Pod。

参考：[CPU Manager](https://kubernetes.io/docs/tasks/administer-cluster/cpu-management-policies/)、
[cgroup v2 CPU/cpuset](https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html)。

## 5. 系统调度采集：能力可用时再启用

基础 profiling 不要求 perf、root、hostPID、privileged Pod 或修改内核参数。
如果容器允许相关 tracepoint 采集，可以在短窗口执行：

```bash
python vllm_ascend/profiler/sparse_kv_cpu.py record-sched \
  --seconds 10 --output /tmp/scheduler.json
```

这是系统范围的 `perf record`，采集 sched_switch/wakeup/wakeup_new，使用 monotonic 时钟。
必须让该窗口覆盖目标 planner 调用。工具保存原始 `.perf.data`、错误信息和解码结果；
不自动提升权限或改变 sysctl。若不可用，基础 planner 数据依然有效。
系统范围事件量可能较大，采集与写盘也会争用资源，因此单独验证其扰动。

若需要管理员协助，在对应宿主机运行同一个 CPU-only 脚本即可，不需要安装 torch。
同时从宿主机读取目标进程各线程的 `/proc/<PID>/task/<TID>/status` 中 `NSpid`，
构建容器 TID 到宿主机 TID 的映射，例如：

```json
{"123": 40123, "124": 40124}
```

内核 tracepoint 中的 TID 与容器内的 TID 可能不同。不能依靠线程名字匹配。
确认两份记录来自同一物理节点、同一次进程运行，且不存在未校正的 monotonic
时间命名空间偏移，再关联：

```bash
python vllm_ascend/profiler/sparse_kv_cpu.py analyze \
  /tmp/planner-profile/ACTUAL_lru_callback.json \
  --scheduler /tmp/scheduler.json \
  --tid-map /tmp/container-to-host-tids.json \
  --same-host --output /tmp/with-scheduler.json
```

`--same-host` 是对不同 hostname 实际属于同一物理节点和时钟的明确确认，不是跨机校时。
若 planner 本身已经使用宿主机 TID，可用 `--host-tids` 代替 `--tid-map`。

分析分别对线程进入之前、工作区间、工作结束后的尾部给出 running、runnable、blocked、unknown 时间。
不完整的前后边界保留 unknown；事件丢失、未解析行、错误映射或运行时 TID 复用均会
降低可信度。检查 `scheduler_quality`，不能把丢失事件后的重建结果当成精确事实。
调度中的 running 也不是“有效指令执行”；中断、内存停顿等需要其他证据。

## 6. NUMA：容器内可做的检查

```bash
lscpu -e=CPU,NODE,SOCKET,CORE,ONLINE
numactl --hardware
numastat -p "$worker_pid"
cat "/proc/$worker_pid/numa_maps"
```

这些命令可选；缺少 numactl 不影响基础采集。以线程允许的 CPU 与
`Mems_allowed_list`/`cpuset.mems.effective` 为边界，不使用容器仅“看得到”但不允许的节点。

导出的 `buffers` 包含 planner 的 token_mark、token_pos、slot_to_token、lru_slots 和
输入 top-k 地址范围。`buffer_mappings` 将它们对应到重叠 VMA 的 numa_maps。
一个 VMA 可能包含多个分配，因此这是 **VMA 级分布，不是每个缓冲区的精确页分布**。
更精细的只读页位置查询可由管理员在允许时使用 `move_pages(..., nodes=NULL, ...)`；
失败或不支持的映射必须保留为未知，不能靠进程级 numastat 比例推算。

共享 CPU KV、注册/锁定内存及外部分配器可能由不同进程创建；更改 TP0 的启动策略不一定
改变这些页的位置。CPU planner 元数据与 NPU 读取的 host KV 是不同访问路径，需分别分析。

参考：[NUMA 映射](https://www.man7.org/linux/man-pages/man7/numa.7.html)、
[move_pages 查询](https://www.man7.org/linux/man-pages/man2/move_pages.2.html)。

## 7. NUMA 对照实验：逐步执行

### 7.1 固定实验条件

1. 保存原始 Pod manifest、worker 启动参数、CPU affinity、线程数与内存分布。
2. 保留完整 TP=2、DP=8 Pod 负载；不要把单 worker 空闲机器结果当成实际部署结论。
3. 固定请求，保存实际行数、top-k、miss 数，交替重复每组实验至少 5 次作为初步检查。
4. 不在正在计时的进程中迁移内存、改 affinity 或改线程数。
5. 分别比较 profiling 关闭、CPU-only、CPU+OMPT、CPU+调度采集；识别采集引入的长尾。
   关闭时以服务原有延迟指标比较；其他组还可比较 callback 分布。

### 7.2 先处理 CPU 竞争

在容器允许的 CPU 内，为目标 DP 组选择足够的 CPU，并保留通信及其他运行时线程的空间。
先不改变内存策略，比较固定 affinity 前后的 runnable 时间、限流和 callback 长尾。
改变绑定可能同时改变 NUMA 距离，因此需要核对实际页位置，不把所有改善都归为调度。

这是对照实验，不是 profiler 自动执行的优化。不要把整个多 worker launcher
绑到某个小 CPU 集合；按真正的 worker 启动入口配置。

### 7.3 固定 CPU，改变内存放置

仅当允许至少两个内存节点且内存策略调用可用时执行：

| 实验 | CPU | 目标工作区内存 |
| --- | --- | --- |
| 基线 | 当前配置 | 当前配置 |
| 本地 | 固定节点 A 的 CPU 集合 | 节点 A |
| 远程 | 同一 CPU 集合 | 节点 B |

模板（替换占位符后，仅应用于目标 worker）：

```text
numactl --physcpubind=<allowed-cpus> --membind=<allowed-node> <worker-command>
```

1. 检查目标节点可用容量，不能为了实验把整个大型 host KV 池挤到容量不足的节点。
2. 从新进程开始设置策略，再创建/初始化内存、预热。
3. 初始化后验证关键工作区的页位置；已存在的共享池不会因为启动参数自动重新放置。
4. 保持其他 DP 组配置和负载稳定，记录目标组与邻组是否一起改变。
5. 比较相近工作量的 work_wall、work_cpu、调度排队、限流和 callback 分布。
6. 交替本地/远程顺序，避免温度、频率或系统负载趋势造成假相关。
7. 若只能改变整个进程内存策略，结果首先只能证明进程级放置影响，不能直接归因到某个缓冲区。
8. 实验后恢复原始配置并复测。

只允许一个内存节点时，不做强行跨节点实验。如果 CPU 集合跨多个节点，可在保持内存
不变的情况下比较 CPU 放置，但必须同时检查 CPU 竞争差异。只能观察不能干预时，
报告“NUMA 可疑、尚未确认”，不输出伪精确的远程访问耗时。

`numactl` 遇到 `Operation not permitted`，可能是 seccomp/权限限制，不等于机器没有 NUMA。
不要直接切换整个 Pod 为 privileged。转交下一节清单即可。

## 8. 可转交管理员的最小清单

- 该 Pod 所在物理节点、允许的 CPU/内存节点、128 CPU 的实际 quota 及隐藏祖先限制。
- CPU Manager 是否为 static；目标容器是否真正分配了独占 CPU；节点上其他系统负载。
- Topology Manager 的策略和 container/pod scope，以及 NPU device plugin 的 topology hints。
- Memory Manager 策略；16 卡 Pod 和内存请求是否实际跨 NUMA 节点。
- 在有问题的节点执行一次短调度采集，以及目标 worker 的 host/container TID 映射。
- 如需 NUMA 实验，允许的 CPU/内存绑定方式或管理员协助创建的单独测试 Pod。
- 如有受支持的硬件内存事件，再补采相应计数器；不预设特定 ARM CPU 的事件名。

不要默认要求 `single-numa-node`：一个 16 卡、128 CPU 的 Pod 未必能放进一个 NUMA 节点，
该策略可能使 Pod 无法被节点接纳。需要依据实际 topology、scope 和设备提示选择。
参考：[Topology Manager](https://kubernetes.io/docs/tasks/administer-cluster/topology-manager/)、
[Memory Manager](https://kubernetes.io/docs/tasks/administer-cluster/memory-manager/)。

## 9. 如何形成结论

- 工作量更多且最后完成：先考虑负载差异。
- 同工作量下 runnable 明显增多：支持 CPU 竞争/调度延迟方向；同时检查 cgroup 限流。
- blocked 增多：继续定位阻塞原因，不能直接称为线程饥饿。
- 本地放置在调度等待和工作量相近时反复改善工作耗时：支持 NUMA 影响。
- wall 与 CPU 时间都大但缺少内存证据：只能称为 on-CPU 变慢，不能直接断言远程内存。
- OMPT 等待长而自身工作短：该线程可能是等待慢线程的受害者。

保留原始 trace、采集错误、丢弃计数、绑定配置与每次实验条件。CPU-only 测试能验证
记录语义和 planner 输出一致性；真实 NPU/Kubernetes 环境中的观察扰动与原因归因仍需实测。
