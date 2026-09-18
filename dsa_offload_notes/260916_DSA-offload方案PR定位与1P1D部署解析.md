# DSA Offload 三 PR 定位与 1P1D 完整方案解析

> 任务来源：`flona-workspace/task/260916_DSA问题分析.md`
> 分析对象：`vllm-ascend` 上游 4 个 PR（本团队 PR1/PR2/PR3 + 合作团队前置 PR#15642）
> 代码基线：本地 `repo/vllm-ascend`，已 fetch `pr15642 / pr15883 / pr16219 / pr16378` 四个本地分支
> Mooncake 参考版本：v0.3.13post1（`repo/Mooncake`），本方案不涉及 Mooncake 代码改动

---

## 0. 一页速览（先看这个）

**问题**：GLM-5.2 / DeepSeek-V3.2 这类 DSA（DeepSeek Sparse Attention，稀疏注意力：不全看，只挑重点看）模型要跑 1M（100 万）上下文，KV cache（推理时缓存下来的 Key/Value，长上下文里最占显存的东西）放不进 HBM（NPU 上的显存，容量小但带宽极高）。

**方案**：`Prefill 开 DCP`（Prefill=处理用户输入 prompt 的阶段；DCP=Decode 侧上下文并行，用来切分超长序列）＋ `Decode 用 DSA offload`（Decode=逐字生成的阶段；offload=把数据从显存挪到主机内存）。具体做法：把**主 KV（main KV，即完整的那份 KV）放到 host DRAM**（主机内存，就是服务器内存条，容量大但带宽低），只把 **indexer cache**（"打分器"的缓存：先给所有历史 token 打分、挑出前 k 名，体积小且每次都用到）**和 top-k 热缓冲**（显存里那个只放最相关 k 个 token 的"最近常用小仓库"）**留在 HBM**。每次 decode 只用 top-k 个 token 的 KV，其余从主机内存按需搬回。

**四个 PR 的角色分工**（一句话版）：

| PR | 角色 | 一句话职责 | 依赖 |
| :--- | :--- | :--- | :--- |
| **#15642**（前置，合作团队） | 算子 + 基线框架 | 提供 `FusedSparseAttentionOverlap`（把"判断命中→搬运未命中→做注意力计算"合并进一个 kernel 的融合算子）＋ MemFabric 版（华为的主机内存池库）sparse KV offload 全链路；**定义整套语义契约**（各模块约定的输入输出格式与语义：KV 放哪、选哪些 token、membership map 格式、算子 I/O、PD 协议）——详见 §4.1 的契约清单 | 无（基线：作为对比起点的版本） |
| **PR1 #15883** | 资源层 | 把 host KV 分配从"写死 MemFabric"抽成 `HostKVAllocator` 协议，新增 **Mooncake Shared Segment** 实现 `MooncakeHostPool`，加 `host_backend` 开关。**只分配，不通数据** | #15642 |
| **PR2 #16219** | 单机数据通路 | 打通 Mooncake 在**同一节点内**的读写：membership map 两跳 staging（CPU planner → NPU staging → Mooncake 段）+ 图模式下 current-token KV 的 `index_copy_` 写回（C++ host callback 生成/刷新 index 描述符） | #15883 |
| **PR3 #16378** | 跨节点数据通路 | 把 P→D（P=Prefill 节点，D=Decode 节点）的 KV 搬运改成 **blockwise Mooncake DSA transfer**（按 KV 块搬运，用 Mooncake 传输引擎）：Main KV 走 **D2RH**（Device to Remote Host，远端设备→本地主机内存：P 的显存 → D 的主机内存池），Indexer KV 走 **D2D**（Device to Device，显存到显存，不经过主机内存）；支持 P_TP ≠ D_TP（两边张量并行卡数不同）、PP/CP（流水线并行/上下文并行）、多 TP 分区写、混合 location 注册（注册时显式说明每块内存是哪类地址） | #16219 |

**一句话理解三者关系**：
`#15642` 把"KV 放在 host、用融合算子算"这件事**跑通在 MemFabric 上**，并**冻结了整套语义契约**（见 §4.1）——**它定义"是什么"**；
`PR1` 换了**内存来源**（Mooncake Shared Segment 共享内存段：属主卡分配一次物理页，同组各卡都映射一份，且能被 Mooncake TE（Transfer Engine，实际负责搬数据的组件）通过 RDMA/ROCE（远程直接内存访问：网卡直接读写对端内存、不经过对端 CPU）跨节点直接寻址）；
`PR2` 换了**单机内的两条数据面**（membership map 里 plan 的生产与发布、当前 token 的 KV 写回），因为 Mooncake 段是"NPU 可寻址的 tensor"（存在主机里但 NPU 能直接读写，不是普通 pinned CPU 内存（锁页内存：被钉在物理内存里不被换出、可被加速卡直接访问的主机内存））；
`PR3` 换了**跨机数据面**（P→D 直写 D 的 Mooncake host pool），并把 Mooncake 注册从"通配 location"升级为**带 `npu:<device_id>` 的显式 location**。

### 0.1 术语速查（先扫一眼，遇到不懂的回来查）

> 这一节把全文档用到的缩写和变量都用大白话解释一遍。**括号里就是它在本文档中的含义**；
> 正文里每个术语**首次出现时**也会再补一次括号说明。

**① 这个方案到底想干什么（最通俗版）**

大模型生成一个 token（词元，文本的最小切分单位）时，要回头"看"之前所有 token 的
KV cache（Key/Value 缓存：模型在推理时把每一层算过的注意力的"键"和"值"存下来，
避免下次重算。它就是长上下文最占显存的东西）。
要看的东西越多，缓存就越大；到 1M（100 万）token 时，**显存（HBM）根本装不下**。

DSA（DeepSeek Sparse Attention，稀疏注意力）这类模型的特殊之处是：
每次计算其实只需要看**最相关的 k 个 token**（叫 top-k，即打分排名前 k 名），
不是全部。所以本方案把**大部分 KV 放到主机内存（DRAM，就是服务器的主内存条）**，
只在显存里留一小块"最近常用的"（热缓冲）和负责挑人的索引（indexer cache）。
要用的时候再从主机内存把缺的那几个搬回来 —— 而且搬的动作跟计算**重叠**做，不额外等。

**② 模型结构与缓存（为什么 KV 被分成"两段"）**

| 术语 | 通俗解释 |
| :--- | :--- |
| **DSA**（DeepSeek Sparse Attention） | DeepSeek 提出的**稀疏注意力**机制：不全看，只挑重点看。 |
| **SFA**（Sparse Flash Attention） | 昇腾侧对同一类稀疏注意力的实现命名。**本文中 DSA 与 SFA 基本可当同义词**（前者偏算法/模型侧，后者偏算子/代码侧）。 |
| **MLA**（Multi-head Latent Attention，多头潜在注意力） | DeepSeek 的注意力结构：把每个 token 的 KV **压缩**成一个低秩"潜向量"（`kv_lora_rank`，GLM-5.2 为 512 维）——这是压缩后的主体；再单独留一小段位置编码分量（`qk_rope_head_dim`，64 维）——这是给 token 打"位置标记"的部分。<br>**所以"main KV 走 MLA（两段）"的意思就是：主 KV 被拆成 512 维的压缩主体 + 64 维的位置分量这两块来存。** |
| **`kv_lora_rank`** | MLA 里压缩主体那一段的维度（512）。变量名直译是"KV 低秩分解的秩"。 |
| **`qk_rope_head_dim`** | 位置编码那一段的维度（64）。RoPE = Rotary Position Embedding，旋转位置编码，用来让模型知道 token 的先后顺序。 |
| **`index_topk`** | 模型配置里的一个数字：稀疏注意力每次要看多少个 token（例如 2048）。**它就是本文档里的 `topk`**。 |
| **indexer**（索引器）与 **indexer cache** | indexer 是一个**轻量级的"打分器"**：先快速给所有历史 token 打个分，选出前 k 名（top-k）交给正式注意力去算。它自己的缓存就叫 indexer cache（索引缓存），**体积小、每次都用到，所以留在显存里**。 |
| **main KV** | 真正的、完整的 KV（就是上面 MLA 那两段）。**体积大、按需取用，所以放到主机内存**。 |
| **KV cache** | 推理过程中缓存下来的 Key/Value。类比：写文章时把前面查过的资料剪贴下来，后面直接翻，不用重查。 |

**③ 硬件与内存（数据放在哪）**

| 术语 | 通俗解释 |
| :--- | :--- |
| **HBM**（High Bandwidth Memory） | NPU 上的**显存**：带宽极高、容量很小（几十~上百 GB）。 |
| **DRAM** | **主机内存**（服务器内存条）：容量大（几百 GB~TB）、带宽低得多。本文里 "host DRAM"、"host 池" 都指它。 |
| **host / device** | host = 主机（CPU + 内存这一侧）；device = 加速卡（NPU + 显存这一侧）。 |
| **NPU** | 昇腾 AI 加速卡（华为的 GPU 等价物）。 |
| **H2D**（Host to Device） | 数据从主机内存**搬到**显存。 |
| **D2H**（Device to Host） | 数据从显存**搬到**主机内存。 |
| **D2D**（Device to Device） | 显存到显存（两个节点/两张卡之间），不经过主机内存。**该词由 PR3 引入**。 |
| **D2RH**（Device to Remote Host） | "远端设备 → 本地主机内存"。**该词由 PR3 引入**；#15642 里等价的说法是 **RD2H**（R = remote，远端），见 `sfa_remote_d2h_connector.md:10`。 |
| **GVA**（Global Virtual Address） | MemFabric 提供的"**跨进程同一个虚拟地址**"能力。通俗说：0 号卡申请了一块主机内存，把地址告诉其他卡，其他卡**直接用这个地址**就能访问同一块内存。 |
| **pinned memory**（锁页内存） | 被"钉"在物理内存里、不会被操作系统换出的主机内存。**只有这种内存才能被加速卡直接访问/异步拷贝**。 |
| **VA**（Virtual Address） | 虚拟地址。程序看到的地址，需要操作系统/驱动映射到真实物理内存。 |
| **mmap**（memory map） | 把一块内存/文件"映射"进进程地址空间，多个进程可映射同一块。 |
| **HostRegister** | 驱动提供的能力：把一块主机内存**注册成设备可直接访问的地址**（NPU 就能像访问显存一样访问它）。 |
| **NPU 可寻址的 tensor** | 存在主机里，但 NPU 能像访问显存一样直接读写的张量。**Mooncake 段就是这个性质**，这正是 PR2 所有麻烦的根源。 |
| **对齐（alignment）** | 要求数据的起始地址是某个数的整数倍（本文是 2 MiB）。目的是让数据的相对偏移稳定、便于批量处理。 |
| **int16 / int32 / int64** | 16/32/64 位整数。int16 能表示 −32768 ~ 32767，**这也是"长序列下会不会溢出"的排查点之一**。 |
| **dtype** | 数据类型（如 bfloat16 / int16）。 |
| **BF16 / bfloat16** | 16 位浮点数。模型权重和 KV 通常用它，省一半显存。 |
| **`block_size` / block / page** | 显存分配的最小单位。KV 被切成一个个"块"来管理，类似操作系统的内存页。 |

**④ 并行与部署形态**

| 术语 | 通俗解释 |
| :--- | :--- |
| **Prefill**（预填充） | 处理用户输入 prompt（提示词）的阶段：一次性算完所有输入 token。**计算密集**。 |
| **Decode**（解码） | 逐字生成的阶段：每步只生成 1 个 token。**访存密集**，KV cache 读得多。 |
| **PD 分离（disaggregated P/D）** | 把 Prefill 和 Decode **拆到不同节点**分别部署、各自调优。 |
| **1P1D** | 1 个 Prefill 节点 + 1 个 Decode 节点（本文的分析对象）。 |
| **P / D** | 本文中 **P = Prefill 节点，D = Decode 节点**（不是别的缩写）。 |
| **TP**（Tensor Parallel，张量并行） | 把一个大矩阵切成几份放到多张卡上算。`P_TP=8` 表示 Prefill 用 8 卡。 |
| **PP**（Pipeline Parallel，流水线并行） | 把模型的不同层分到不同卡上，像流水线一样接力。 |
| **CP / DCP / PCP** | Context Parallel（上下文并行）：序列太长时，把序列切段分给多卡。DCP = Decode 侧 CP，PCP = Prefill 侧 CP。 |
| **DP**（Data Parallel，数据并行） | 不同卡跑不同请求。 |
| **rank** | 进程/卡的编号（0 号、1 号…）。`tp_rank` 就是在 TP 组里的编号。 |
| **MTP**（Multi-Token Prediction） | 一次预测多个 token 的加速技术（投机解码的一种）。它多出来的层也要参与 KV 搬运。 |
| **spec decode / 投机解码** | 先用小模型猜几个 token，再用大模型一次验证，提速。 |
| **DCP offloading** | 项目名里的说法：Prefill 开 DCP（上下文并行）配合 Decode 侧的 offload。 |

**⑤ 共享内存与传输机制**

| 术语 | 通俗解释 |
| :--- | :--- |
| **MemFabric** | 华为的一套主机内存池/共享内存库。**#15642 里 host KV 池就是它做的**。 |
| **Mooncake** | 蚂蚁开源的一套 KV cache 传输/存储引擎（Transfer Engine + 共享内存段）。**PR1/PR2/PR3 就是要把 host 池换成它**。 |
| **Mooncake Shared Segment**（共享段） | Mooncake 提供的一块主机共享内存：owner（属主 rank）分配一次物理页，同组各 rank 都映射一份，**且能暴露成 NPU 可直接访问的地址，还能被远端 RDMA 直接寻址**。 |
| **owner_rank** | 共享段的"属主"：由它真正分配物理内存，其他 rank 只是映射。本文固定为 0。 |
| **Transfer Engine / TE** | Mooncake 里负责实际搬运数据的组件（走 RDMA/ROCE 网络或本机拷贝）。 |
| **RDMA / ROCE** | 远程直接内存访问：网卡直接读写对端内存，**不经过对端 CPU**，所以快。ROCE 是它在以太网上的实现。 |
| **注册（register_memory）** | 把一块内存"登记"给 TE，之后它才能对该地址做 RDMA 读写。未注册的地址不能搬。 |
| **location** | 注册时告诉 TE"这块内存是什么类型/挂在哪张卡上"：`"npu:0"` 表示 0 号 NPU 可访问的地址，`"*"` 是通配（让 TE 自己判断）。**PR3 的核心改动之一就是把这个参数显式传下去**。 |
| **`npu:<device_id>`** | 上面 location 的具体写法，device_id 是 NPU 卡号。 |
| **64 GiB 上限** | Mooncake 单次注册的内存大小有上限（本 PR 取 64 GiB），超过就要切成多块注册。 |

**⑥ 稀疏注意力的核心数据结构（本文重点）**

| 术语 | 通俗解释 |
| :--- | :--- |
| **top-k** | 打分排名前 k 名。TopK = 挑出前 k 名的操作。 |
| **热缓冲 / selection buffer** | 显存里那块"最近常用的小仓库"，只放 top-k 个 KV。`selection_kv_cache` 是它的 K 部分，`selection_k_rope` 是位置分量部分。 |
| **`topk_buffer_size`** | 热缓冲能放多少个 slot（槽位），默认 4096。 |
| **Hit / Miss** | 要用的 token 已经在热缓冲里 = Hit（命中）；不在 = Miss（未命中，需要从主机内存搬回来）。**命中率越高越快**。 |
| **LRU**（Least Recently Used） | "最久没用过的先淘汰"的缓存策略。planner 用它决定把哪个 slot 腾出来给新数据。 |
| **planner / 规划器** | 一段**跑在 CPU 上的 C++ 代码**（用 OpenMP 多线程）：负责算"这步哪些命中、哪些未命中、未命中的要放到哪个 slot"。它只出计划，不搬数据。 |
| **plan（计划）/ external plan** | planner 产出的结果：一张表，告诉 GPU/NPU 算子"每个 token 该怎么办"。**编码方式：+ (slot+1) 和 − (slot+1)，正负号区分用途**。 |
| **membership map** | **本文最关键的共享数据结构**：一张 int16 的表，记录"某个 token 现在在不在热缓冲里、在哪个 slot"。CPU 上的 planner 写它，NPU 上的算子读它 —— 所以它是**跨 CPU/NPU 的契约**。行宽固定 16400。 |
| **slot**（槽位） | 热缓冲里的一格，放一个 token 的 KV。 |
| **pre-load / onload** | 把数据从主机内存**装进**显存热缓冲（overload 的反面）。 |
| **offload** | 把数据从显存**挪到**主机内存（腾显存）。本文标题的 "offload" 就是这个意思。 |

**⑦ 融合算子与执行机制**

| 术语 | 通俗解释 |
| :--- | :--- |
| **算子（operator / op）** | 一个具体的计算功能实现（类比一个函数），通常指跑在加速卡上的 kernel。 |
| **融合算子（fused operator）** | 把原本要分好几步做的计算**合并进一个 kernel**，省掉中间数据的来回搬运和等待。 |
| **FusedSparseAttentionOverlap** | 本方案的核心算子：一个 kernel 里同时做"判断命中/未命中 → 把未命中的从主机内存搬进热缓冲 → 做稀疏注意力计算"，让搬运时间被计算掩盖。名字里的 Overlap 就是"重叠"。 |
| **kernel** | 跑在加速卡上的计算程序。 |
| **`use_fused_overlap`** | 配置开关：是否启用上面这个融合算子（默认 false）。 |
| **graph / 图模式 / ACL graph** | 把一整段计算流程**提前录制成一张静态图**，之后每步"重放"这张图。省掉每步的调度开销，但**要求所有张量形状固定**——这正是 PR2 麻烦的根源。 |
| **capturing / capture** | 正在"录制"图的过程。 |
| **replay（重放）** | 执行已录制的图。 |
| **host callback** | 在图上挂一个"重放到这里时执行一段 CPU 代码"的回调。**PR2 用它每帧刷新索引表**。 |
| **stream（流）** | 加速卡上的任务队列，同一个流里串行、不同流可并行。本文有三条流：计算流、planner 流、写回流。 |
| **eager（即时执行）** | 不录图，算一步走一步（与 graph 模式相对）。 |
| **`index_copy_`** | 按索引把一批数据"贴"到目标张量的指定位置（类似按地址列表批量赋值）。 |
| **`index_select`** | 按索引从源张量里挑出一批行。 |
| **`nonzero()`** | 找出所有非零元素的位置。**问题在于它返回的长度不固定**，所以不能录进图里。 |
| **描述符（descriptor）** | 一组"源地址 + 目标地址 + 长度"的表，告诉底层驱动怎么搬数据。 |
| **staging（中转/暂存）** | 数据不能直接到达最终位置时，先放到一个中转缓冲区。**PR2 的"两跳发布"第二跳前的中转就是它**。 |
| **padding（填充）** | 把长度不齐的数据补齐到固定长度（这里是保证图模式下形状稳定）。 |
| **OpenMP** | CPU 上的多线程并行库，planner 用它并行处理多行。 |

**⑧ 软件工程与流程**

| 术语 | 通俗解释 |
| :--- | :--- |
| **PR**（Pull Request） | 向开源仓库提交代码改动的请求。 |
| **基线 / baseline** | 作为对比起点的版本（本文指 #15642 的状态）。 |
| **契约（contract）** | 各模块之间约定的"输入输出格式和语义"，改它就等于改接口。 |
| **schema** | 算子的参数签名（有哪些输入输出、类型是什么）。 |
| **Connector（连接器）** | vLLM 里负责 P/D 之间搬运 KV 的插件（如 `SfaRemoteD2HConnector`、`MooncakeConnectorV1`）。 |
| **proxy / metaserver** | 请求分发与 P/D 配对（rendezvous）的服务，也用来公告"D 的地址在哪"。 |
| **rendezvous** | 配对/会合：P 和 D 需要先互相找到对方才能传数据。 |
| **ZMQ** | 一个高性能消息队列库，本方案用它做 P/D 之间的控制面通信。 |
| **wire 协议** | 网络上传的报文的字段定义。 |
| **side channel** | 控制面通道（传元数据），与传数据的通道分开。 |
| **block ID** | KV 块的编号（相当于页码）。 |
| **slot_mapping** | 记录"这一步新生成的 token 应该写到哪个槽位"的映射表。 |
| **chunked prefill（分块预填充）** | prompt 太长时切成几段分步处理。 |
| **recompute** | 不存缓存、需要时重算（用算力换显存）。 |
| **barrier / broadcast / all_gather** | 多卡之间的**集合通信**操作：barrier = 大家等齐；broadcast = 一个人说给所有人；all_gather = 每个人都把数据交出来，最后人人都有全量。 |
| **`tp_group` / `cpu_group`** | 进行上面这些集合通信的进程组（`cpu_group` 是走 CPU 的通信组）。 |
| **UT（Unit Test）** | 单元测试。 |
| **CI** | 持续集成（提交后自动跑检查/测试）。 |
| **HEAD / head 分支** | 一个 PR 当前最新的提交。 |

**⑨ 读文档时最容易看反的三组概念**

| 容易混 | 区别 |
| :--- | :--- |
| **Prefill vs Decode** | Prefill 一次算很多 token（算力瓶颈）；Decode 一次算 1 个 token（显存带宽瓶颈）。所以两者优化手段完全不同。 |
| **H2D 与 D2H** | 看箭头方向：H2D = 主机→显存（装进来）；D2H = 显存→主机（挪出去）。**名字里第一个字母是起点**。 |
| **P 与 D** | P = **P**refill 节点；D = **D**ecode 节点。**不是** producer/decoder 之外的缩写。 |

---

### 0.2 分工总图

![三个 PR 在 1P1D 里的分工总图](plantuml/img/10-pr-scope-map.png)

> **图 0** · 三个 PR 在 1P1D 里的分工总图 —— 蓝=P 侧基线，绿=D 侧基线，橙=★PR1/★PR2，深橙=★PR3。源文件：`plantuml/10-pr-scope-map.puml`

**全部 13 张图（按阅读顺序）**

| 图 | 看什么 | 章节 |
| :--- | :--- | :--- |
| 图 0 | 三个 PR 分工总图（先看这张建立全局） | §0.2 |
| 图 1 | 1P1D 物理部署 + 三个 PR 落点 | §2.1 |
| 图 2 | Decode 单节点三层存储 + 三条数据面 | §2.2 |
| 图 3 | 契约层 ↔ 两个后端依赖面（为什么需要三次替换） | §3.1 |
| 图 4 | 启动阶段：host 池怎么建起来（PR1 落点） | §3.2 |
| 图 5 | **一次请求完整时间线 T0–T7**（PR2/PR3 主战场） | §3.3 |
| 图 6 | ★PR2 membership plan 两跳发布 | §4.3① |
| 图 7 | ★PR2 图模式 current KV 写回 4 阶段 | §4.3② |
| 图 8 | ★PR3 D 的两种 cache 两个内存域 | §4.4① |
| 图 9 | ★PR3 Endpoint 规划 | §4.4③ |
| 图 10 | membership map 行布局（跨内存域契约） | §5 |
| 图 11 | ★PR3 三条硬约束 → 对策 | §7 |
| 图 12 | 速查：PR ↔ 1P1D 位置 | 附录 B |

> **判断"是否动了契约"的实用判据**：三个 PR 全程**没有改过 `selection_*` 五个张量的 shape/dtype/语义、没有改过 membership map 的行宽与控制字段含义、没有改过融合算子的 schema、没有改过 `SfaRemoteD2HConnector` 的 wire 协议**。所有改动都发生在"内存从哪来"和"数据怎么进出这块内存"两层。一旦发现某处改动触碰了上面任何一项，就说明方案在语义层发生了演进，需要重新对齐基线。

---

## 1. 背景与目标

### 1.1 要解决什么

| 维度 | 说明 |
| :--- | :--- |
| 模型 | GLM-5.2、DeepSeek-V3.2 等**稀疏注意力**模型（DSA = DeepSeek Sparse Attention，算法/模型侧叫法；SFA = Sparse Flash Attention，昇腾侧对同类算子的叫法；**两者本文中基本同义**），`index_topk`（模型配置里的数字：每次只看多少个 token，例如 2048；**它就是本文档里的 `topk`**）决定每次 attention 只看 top-k 个 token |
| 目标 | 单请求 **1M 上下文**（100 万 token 的上下文长度；100k = 10 万量级）可部署 |
| 瓶颈 | 1M 上下文的主 KV cache（完整那份 KV 缓存）远超单卡 HBM（显存）容量 |
| 关键洞察 | DSA 每次只用 top-k 个 token 的 KV ⇒ **没必要把所有 KV 放 HBM**，大部分可以放到便宜得多的主机内存里 |
| 手段 | main KV 落主机内存（DRAM）；indexer（打分器）+ top-k 热缓冲留 HBM；top-k 识别、miss 判定（Miss = 要用的 token 不在热缓冲里，需要从主机内存搬回来）、H2D onload（Host to Device：从主机内存装进显存）**全部塞进一个融合算子**，与 attention 计算重叠，让搬运时间被计算掩盖 |

### 1.2 部署形态：1P1D

- **1P（Prefill 节点，P）**：`kv_role="kv_producer"`（角色=生产者，即"我把 KV 给你"）。开启 DCP（decode context parallel 的 prefill 侧配合项；Context Parallel = 上下文并行：序列太长时把序列切段分给多卡算），Prefill 自己的 KV 存在 HBM 分页 cache 中（分页 cache = 像操作系统管理内存页一样，把 KV 切成一个个"块"来管理；可选叠加 AscendStore 的 layerwise offload（逐层卸载：算完一层就把那层 KV 挪走、复用同一块 buffer）来复用少量物理 buffer）。
- **1D（Decode 节点，D）**：`kv_role="kv_consumer"`（角色=消费者，即"我去把 KV 拉过来"）。开启 `sparse_kv_offload_config`（稀疏 KV 卸载配置项），main KV 常驻主机内存（DRAM），indexer cache 常驻本 rank（本进程/本卡，rank = 进程编号）的 HBM。
- 中间可有一个 **proxy / metaserver**（代理/元数据服务）做请求分发与 P/D 会合（rendezvous = 配对、会合：P 和 D 要先互相找到对方才能传数据）。
- 两个节点之间：**Mooncake Transfer Engine**（TE，Mooncake 里真正负责搬数据的组件）负责实际 KV 搬运；控制面（传元数据、不发数据）用 ZMQ（一个高性能消息队列库）side-channel（旁路控制通道）+ HTTP metaserver。

### 1.3 关键术语（带通俗解释）

> 更完整的名词表在 §0.1；这里只列后续章节反复用到的核心概念。

| 术语 | 含义（括号内为通俗解释） |
| :--- | :--- |
| **DSA / SFA** | DeepSeek Sparse Attention（DeepSeek 提出的稀疏注意力：不全看，只挑重点看）/ Sparse Flash Attention（昇腾侧对同类稀疏注意力的实现命名）。<br>**这里最容易看不懂的是"main KV 走 MLA（`kv_lora_rank` + `qk_rope_head_dim` 两段）"**，解释见下方 ☞ **MLA 与"两段 KV"**。<br>除 main KV 外，还有一份独立的 indexer cache。 |
| **MLA 与"两段 KV"** ☞ | MLA = Multi-head Latent Attention（多头潜在注意力），DeepSeek 用来把 KV **压缩**的结构。它把每个 token 的 KV 拆成两块存：<br>· **压缩主体**：维度是 `kv_lora_rank`（GLM-5.2 为 512）。名字直译"KV 低秩分解的秩"，可以理解为"把原本很大的 KV 压成一个 512 维的摘要向量"。<br>· **位置分量**：维度是 `qk_rope_head_dim`（64）。RoPE = Rotary Position Embedding（旋转位置编码），负责让模型知道 token 的先后顺序。<br>所以"main KV 走 MLA 两段"= 主 KV 分成 **512 维主体 + 64 维位置分量** 两块来存储和搬运。后续代码里 `k_cache`/`v_cache` 对应的正是这两块。 |
| **main KV** | 每个 token 完整的主 KV，就是上面 MLA 那两段（K 段 = `kv_lora_rank` = 512，V 段 = `qk_rope_head_dim` = 64）。存成 BF16（bfloat16，16 位浮点数，省一半空间），只有 1 个 head（不做多头拆分）。 |
| **indexer cache** | "打分器"自己的缓存（indexer = 先快速给所有历史 token 打分、选出前 k 名的那套轻量机制）。用 LI C8 量化（8 位低精度，进一步缩小体积）。**常驻 HBM**（每次都用到，必须低延迟），参与 top-k（挑前 k 名）打分。 |
| **top-k buffer / selection buffer** | HBM 上的**热缓冲**：只放"最近最可能被用到"的 top-k 个 KV 行。容量 = `topk_buffer_size`（能放多少格，默认 4096）。代码里 K 部分叫 `selection_kv_cache`，位置分量部分叫 `selection_k_rope`。 |
| **membership map** | **本文档最关键的共享数据结构**（见 §5、§0.1⑥）。它是一张 int16（16 位整数）的表，记录"某个 token 现在在不在热缓冲里、在哪个 slot（格子）"。**CPU 上的 planner（规划器）写它、NPU 上的融合算子读它**，所以它是跨 CPU/NPU 的"契约"。行宽固定 16400。 |
| **Hit / Miss** | Hit = 命中：要用的 token 已经在热缓冲里，直接用。Miss = 未命中：不在，需要从主机内存搬回来。 |
| **LRU planner** | LRU = Least Recently Used（最久没用过的先淘汰），一种缓存淘汰策略。planner 是一段**跑在 CPU 上的 C++ 多线程代码**：算出这步哪些命中、哪些未命中、未命中的该放进哪个 slot。**它只出计划、不搬数据**。产出物叫 plan（计划）。 |
| **D2RH** | Device to Remote Host。"远端设备 → 本地主机内存"（远端 P 的显存 → 本地 D 的主机内存）。**注意：这个词由 PR3 引入**（`mooncake_dsa_metadata.py::DsaTransferPhase.MAIN_D2RH`）；#15642 里并不存在 "D2RH" 这个字面量，它的等价说法是 **RD2H**（R = remote，`SfaRemoteD2HConnector` 的 R 就是这个意思，见前置设计文档 `sfa_remote_d2h_connector.md:10`）。 |
| **D2D** | Device to Device（显存到显存，不经过主机内存）。这里指 P 的 NPU HBM → D 的 NPU HBM。**同样由 PR3 引入**（`DsaTransferPhase.INDEXER_D2D`）。 |
| **D2H / H2D** | 本地语义，看**箭头方向的起点**：D2H = Device to Host（显存→主机内存，本文用它把当前 token 的 KV 写回主机池）；H2D = Host to Device（主机内存→显存，本文用它把未命中的 token 装进 top-k 热缓冲）。#15642 用的是这两个。 |
| **GVA** | Global Virtual Address（全局虚拟地址）。MemFabric 的"跨进程/跨 rank 同一虚拟地址"能力：通俗说就是 0 号卡申请了一块主机内存，把地址告诉其他卡，**其他卡直接用这个地址就能访问同一块内存**。 |
| **fused_overlap** | 配置开关 `use_fused_overlap=true`：用 `FusedSparseAttentionOverlap` 融合算子替代"先跑 planner、再做 attention"的两段式，让未命中 token 的 H2D 搬运与 attention 计算**重叠**（名字里的 Overlap 就是"重叠"）。<br>**注意**：融合算子内部（arch22 = 910C）本身带 planner；**arch35（950）没有算子内 planner，只吃 external plan**（外部 planner 产出的计划，即上文的 plan）。 |

---

## 2. 1P1D 部署示意图

### 2.1 物理部署总览

![1P1D 物理部署总览](plantuml/img/01-deployment-1p1d.png)

> **图 1** · 1P1D 物理部署总览 —— 三个新 PR 的落点一览（★ 标记处）。源文件：`plantuml/01-deployment-1p1d.puml`

**图注（PR 归属）**：

| 图中元素 | 由谁引入 |
| :--- | :--- |
| P-NPU 的 main/indexer 分页 cache、D-NPU 的 indexer/top-k buffer、host 侧 main KV 池的**存在性** | **#15642** |
| host 池的"内存从哪来"（MemFabric → **Mooncake Shared Segment** 共享内存段）、`host_backend` 开关、TP 各 rank（各进程/各卡）**本地视图** | **PR1 #15883** |
| D 节点内：membership map 放进 Mooncake 段、planner→operator 的 staging、current KV 写回 host 池 | **PR2 #16219** |
| ③④ 两条跨节点搬运路径、`dsa_pd_offload` 开关、混合 location 注册（注册时显式声明每块内存是哪类地址）、D_TP（Decode 侧张量并行卡数）分区写 | **PR3 #16378** |

### 2.2 Decode（D）节点内部结构——重点

![Decode（D）单节点内部结构](plantuml/img/02-decode-node-internal.png)

> **图 2** · Decode（D）单节点内部结构 —— 三层存储（NPU HBM 显存 / Mooncake 段 主机共享内存 / CPU）+ 三条数据面。源文件：`plantuml/02-decode-node-internal.puml`

---

## 3. 完整方案的设计与路径

### 3.1 问题拆解：为什么需要三次"数据面替换"

先把**不依赖内存后端**（"后端"= 实际提供内存的那套库，如 MemFabric / Mooncake）的部分，和**依赖内存后端**的部分分开：

![问题拆解](plantuml/img/03-dependency-layers.png)

> **图 3** · 问题拆解 —— 契约层（#15642）不变，三个 PR 只替换"内存来源"与"数据面"。源文件：`plantuml/03-dependency-layers.puml`

**替换的动机**（关键，理解 PR2/PR3 的一切都从这里出发）：

1. **MemFabric 的 host 池是"本机、靠对称 GVA 共享"的语义**（`Scene.SHARED`：rank0 出真实物理页，其余 rank 预留同尺寸 VA（虚拟地址）窗口，并把 rank0 给的裸指针当本地地址用）。这带来两个问题：**VA 开销随 TP 线性膨胀**（TP16 + 128 GB ⇒ 约 2 TB VA），以及**这块内存无法被外部 RDMA 引擎按 host 语义注册**。跨节点让 P 直接把 main KV 写进 D 的 host DRAM（D2RH），需要一块**能被 Mooncake Transfer Engine 注册、并让远端通过 RDMA/ROCE 直接寻址**的 host 内存。Mooncake Shared Segment 正是为此设计：`owner_rank` 分配一次 host 物理页，全体 TP rank `mmap` + `HostRegister`，同时暴露 NPU 可寻址视图，且**不要求跨 rank 虚拟地址一致**（每 rank 只预留 1×size）。
2. **Mooncake 段不是"普通 pinned CPU 内存"，而是"NPU 可寻址的 tensor"**（pinned CPU 内存 = 被钉在物理内存里、不被操作系统换出、可被加速卡直接访问的主机内存；NPU 可寻址 = NPU 能像读显存一样直接读它）。这一个事实同时打断了 MemFabric 路径上的两个隐含假设，而这两处正是 PR2 的全部内容：
   - 假设一：**CPU planner（跑在 CPU 上的规划器）和融合算子可以直接共享同一块内存**。Mooncake 段对 CPU 侧来说不是可写的 pinned int16 缓冲（对 CPU 而言它表现为一个 device tensor，即"显存里的张量"），于是需要"两跳 staging"（staging = 中转/暂存：数据先放到中转缓冲区再进最终位置）。
   - 假设二：**"把当前 token 的 KV 写到主机内存"可以用 `offload.sparse_copy`（MemFabric 的批量拷贝接口）+ 指针数组完成**。一旦目标变成 NPU 可寻址视图，最自然的方式是 `index_copy_`（按索引把一批行"贴"到目标张量的指定位置）；但 `index_copy_` 需要**形状稳定的索引张量**，而 `nonzero()`（找出所有非零元素位置）的输出长度会随 batch（这一批请求的数量）变化，**无法在图（ACL graph：提前录制成静态图、之后每步重放，要求所有张量形状固定）里重放**。于是需要 C++ host callback（在图重放到此处时执行的一段 CPU 回调）生成并每帧刷新描述符（描述符 = 一组"源地址+目标地址+长度"的表）。
3. **跨机搬运需要区分"这段内存到底是主机内存还是显存"**。Mooncake 的 `register_memory(ptr, size)`（注册：把一块内存登记给 Transfer Engine，之后才能对它做 RDMA 读写）只按 device（显存）语义注册；混用 host pool 与 HBM 必须显式传 `location`（告诉 TE"这块内存是什么类型、挂在哪"，`"npu:<device_id>"` = 该卡号可访问的地址，`"*"` = 通配让 TE 自己判断）。这就是 PR3 对 `GlobalTE.register_buffer` 的改造，以及 P 侧/D 侧两套注册策略的由来。

### 3.2 启动阶段：host 池是怎么建起来的（PR1 的落点）

![启动阶段](plantuml/img/08-startup-activity.png)

> **图 4** · 启动阶段 —— host 池是怎么建起来的（★ = PR1 新增/改动）。源文件：`plantuml/08-startup-activity.puml`

**启动阶段各 PR 归属**：

| 步骤 | PR |
| :--- | :--- |
| `sparse_kv_offload` 配置、池大小规划、MTP dummy metadata（MTP = Multi-Token Prediction，一次预测多个 token 的加速技术；它的额外层也要参与搬运，占比对阶段需要一份占位元数据） | #15642 |
| allocator（分配器）抽象、`MooncakeHostPool`（Mooncake 主机内存池）、`host_backend`（选后端：memfabric 还是 mooncake）选项、`prepare_host_kv_allocation` 调用点、`shutdown()` 释放 | **PR1** |
| `describe_local_views()` 一致性校验（只比较"在共享段内的偏移量"，不比较绝对虚拟地址——因为各 rank 映射到的虚拟地址可能不同） | **PR3**（PR1 只有 base 版本） |

### 3.3 运行阶段：一次请求的完整时间线

![一次请求的完整时间线（T0–T7）](plantuml/img/04-request-timeline.png)

> **图 5** · 一次请求的完整时间线（T0–T7）—— T4 是 ★PR3 主战场，T5/T6 是 ★PR2 主战场，也是 0916 问题的重点排查区。源文件：`plantuml/04-request-timeline.puml`

---

## 4. 四个 PR 的精确定位（文件 / 接口 / 职责）

### 4.1 前置 PR #15642（合作团队 Bourn3z，base 分支 `releases/v0.26.0rc`）

**这个 PR 定义的是"是什么"——即整套语义契约。** 后续三个 PR 只替换契约下面的实现（内存来源 + 两条数据面），契约本身一格未动。下面这张表是本方案最该背下来的东西：

| # | 定义的契约 | 具体内容 | 后续 PR 是否改动 |
| :--- | :--- | :--- | :--- |
| 1 | **KV 该放在哪**（内存布局） | main KV 全量落主机内存（DRAM）；indexer cache 与 top-k 热缓冲（selection buffer）留 HBM；decode 侧**完全不分配 NPU 主 KV cache**（`_compute_kv_only` 这个只算 KV、不写显存分页 cache 的路径，绝不碰 NPU paged cache = 显存里的分页 KV 缓存）| ❌ 未动 |
| 2 | **"哪些 token 需要被算"的语义** | 由 indexer（打分器）给出 top-k 个 token 的 id（数量取自 `hf_text_config.index_topk`），只有这 k 个 token 参与 attention（注意力计算） | ❌ 未动 |
| 3 | **membership map 这个跨 CPU/NPU 共享契约** | 行宽 `16400`（每行 16400 个 int16）；token→slot+1 映射区 `[0,16376)`（前 16376 列：记录每个 token 在哪个 slot 槽位）；控制区 `[16384,16392)`（8 个 int16 = 4 个 int32，存 marker（标记值，用来告知"计划已就绪"等状态）与参数）；控制区起点 `plan_start = 16384 − topk`；**plan 值 = ±(slot+1)**（正负号区分用途）| ⚠️ **格式与语义不变**；PR2 只换了"谁写、怎么写进去"（两跳 staging）|
| 4 | **融合算子的输入/输出契约** | `op_host/..._def.cpp`：14 个输入 / 2 个输出 / 5 个 attr（属性参数）；`selection_*` 五个张量**全部原地更新**（就地改，不新建张量）；`attention_out` 宽度 = `full_kv_cache` 最后一维（512，仅 NoPE = 不含位置编码那部分）；`selection_kv_actual_seq` 是 **alias（别名，与另一块内存指向同一处）到 `status[topk]` 槽位的副作用输出** | ⚠️ schema（算子参数签名）未变；PR2 只改了 membership 存储合法性的判定方式 |
| 5 | **planner 与 attention 的分工** | CPU 侧 C++ OpenMP（CPU 多线程并行库）LRU planner 只负责"选谁、放哪个 slot、哪些未命中"，产出一份紧凑的 plan（计划表）；算子负责"取数 + 算"，H2D 搬运与 MM1/softmax（注意力计算内部的两个矩阵乘/归一化步骤）重叠（3 级流水 + 4 槽 GM ring = 显存里 4 个格子的环形缓冲区）| ❌ 分工不变 |
| 6 | **PD 拉取的控制协议** | `SfaRemoteD2HConnector`（一个 P/D 搬运插件）：ZMQ 消息 `READ_READY_BATCH` / `READ_DONE` / `READ_FAILED`、layer 级（每一层）"物理存储确实写完了"这一不变量、Decode 侧的目标块按 req_id（请求编号）解析 | ❌ 协议不变（PR3 是**另起一条** `MooncakeConnectorV1` 的 DSA 通路，两条并存）|
| 7 | **用户可见的配置面** | `sparse_kv_offload_config.{enabled（总开关）, topk_buffer_size（热缓冲容量）, dram_size_per_dp_GB（每个 DP 组的 host 内存上限）, keep_device_kv_cache（是否保留全量显存 KV，仅调试）, use_fused_overlap（是否启用融合算子）}`；`kv_role`（角色：生产者/消费者）分工；`transfer_backend=memfabric`（传输后端必须是 memfabric）约束 | ⚠️ PR1 加 `host_backend`、PR3 加 `dsa_pd_offload`；**旧键语义全部保持兼容** |

> **实用判据**：三个 PR 全程**没有改过 `selection_*` 五个张量的 shape/dtype/语义、没有改过 membership map 行宽与控制字段含义、没有改过融合算子 schema、没有改过 `SfaRemoteD2HConnector` 的 wire 协议**。所有改动都落在"内存从哪来"与"数据怎么进出这块内存"两层。这正是 PR1 敢写 "resource-layer change only; it does not modify attention or KV computation logic" 的底气。一旦某处改动触碰了上表任一项，就说明方案在语义层演进，需要重新对齐基线。

**已核实的两条事实**（会直接影响怎么读这份方案）：

1. `git rev-list --count ae638fecd..pr15642` = **4**；GitHub API 也报 `commits: 4, changed_files: 36, +15612/-252`，head `7c964767a`，base 分支 `releases/v0.26.0rc`。所以**规范 diff 就是 `git diff ae638fecd..pr15642`**，这就是该 PR 自身的全部改动，**没有需要过滤的无关提交**。
2. ⚠️ **但 #15642 的基线是 `releases/v0.26.0rc`，而 PR1/PR2/PR3 的基线是 `main`**。`main` 在这期间已经把 `sparse_kv_offload.cpp`（包内 JIT 编译）**重构进 `csrc/torch_binding.cpp`**（改走 `torch.ops._C_ascend`），并新增了 `_CPU_CACHE_ALIGNMENT` / `empty_aligned_int8_cpu_tensors` / 内存规划函数。所以 **PR1/PR2/PR3 不是直接叠在 `pr15642` 分支上**，而是叠在"已吸收 #15642 等价能力 + 完成 csrc 重构的 main"上（PR1 base = `9e5fc3dcf4`）。理解这一点，才能看懂 PR2 为什么改的是 `csrc/torch_binding.cpp` 而不是包内 `.cpp`。

该 PR 的 4 个 commit：

| commit | 内容 |
| :--- | :--- |
| `1dd633e1c` | 新增融合算子 `fused_sparse_attention_overlap`（op_host / op_kernel arch22+arch35 / torch adapter） |
| `d9cc7fb7f` | `sfa_kv_offload` 支持融合算子 + 为融合算子新增 LRU mapper |
| `faf1f3869` | 按 NPU / DRAM / workload 三重上限约束 sparse KV offload 内存（原 PR #14921）|
| `7c964767a` | 修 SFA PD 目的端 block 注册竞态（`sfa_pd_rd2h/read_thread.py`） |

**⚠️ 这 4 个 commit 必须一起看**：`1dd633e1c` 是**纯算子**（+12357 行 C++/测试），它的 commit message 明确写着 *"The code on the framework side will be submitted in a subsequent pull request"* —— 也就是 `d9cc7fb7f`。**只摘 `1dd633e1c` 不可运行**（没有 Python 调用方）。

**这些契约落在哪些文件里**：

| 契约 | 位置 |
| :--- | :--- |
| 融合算子本体 | `csrc/attention/fused_sparse_attention_overlap/`（`op_host` 定义/proto/tiling + `op_kernel` arch22/arch35 + torch adapter `*_torch_adpt.h`）|
| 算子 Python 入口 / 调用点 | `torch.ops._C_ascend.npu_fused_sparse_attention_overlap`；唯一调用点 `vllm_ascend/attention/sfa_kv_offload.py::_execute_fused_overlap_offload_decode` |
| host 池 + membership map + LRU planner | `vllm_ascend/distributed/kv_transfer/sparse_kv_offload/sparse_kv_offload_manager.py`（1668 行）|
| planner 的 C++ 实现 | `vllm_ascend/distributed/kv_transfer/sparse_kv_offload/sparse_kv_offload.cpp`（#15642 里是包内 JIT；`main` 已重构进 `csrc/torch_binding.cpp`）|
| PD 拉取连接器 | `vllm_ascend/distributed/kv_transfer/kv_p2p/sfa_pd_rd2h/`（`SfaRemoteD2HConnector`，MemFabric 版）|
| 用户文档 / 设计文档 | `docs/source/user_guide/feature_guide/layerwise_and_sparse_kv_cache_offloading.md`；`docs/source/developer_guide/Design_Documents/sfa_remote_d2h_connector.md` |

**#15642 版本里 host 内存只有一条路（MemFabric 共享 GVA 池）**：

```python
# pr15642: SparseKVOffloadManager.__init__ —— MemFabric 池的建立
config = offload.OffloadConfig()
config.device_id = torch_npu.npu.current_device()
config.reserve_size = actual_pool_size_bytes               # 每个 rank 都预留同样大小
config.alloc_size = actual_pool_size_bytes if self.tp_rank == 0 else 0   # 只有 rank0 真正分配
config.world_size = self.tp_size
config.rank_id = self.tp_rank
config.scene = offload.Scene.SHARED                        # ← 关键：对称 GVA 场景
assert offload.initialize(config) == 0
self.tp_group.barrier()
```

```python
# pr15642: allocate_kv_cache_tensors_for_sparse_kv_offload —— 只有 tp0 从池里取内存
if tp_rank == 0:
    [k_tensor_cpu, v_tensor_cpu] = empty_aligned_int8_cpu_tensors([k_tensor_size, v_tensor_size], alignment)
else:
    k_tensor_cpu = None      # ← 非 0 rank 不分配
    v_tensor_cpu = None
```

非 0 rank 靠**广播出来的同一批裸指针**（MemFabric 对称 GVA）访问：

```python
# pr15642: register_kv_caches 的 GVA 分支（PR1/PR2 把它改成 _register_local_mooncake_views）
gvas_k_tensor = torch.zeros([num_layers], dtype=torch.int64, device="npu")
...
self.tp_group.broadcast(gvas_k_tensor, src=0)          # 把 tp0 的 data_ptr 当"本地地址"用
self.tp_group.broadcast(gvas_v_tensor, src=0)
...
self.k_caches_cpu = [self._restore_bfloat16_tensor(ptr, cpu_k_shape) for ptr in self.gvas_k_bases]
```

**这里有两个容易忽略但很重要的性质**：

1. **`offload.empty(..., pin_memory=True)` 取的不是普通 pinned 内存，而是 MemFabric 共享池里既 CPU 可写、又能被 NPU 直接寻址的页面**。所以 `_restore_bfloat16_tensor(ptr, shape)` 造出来的"非拥有视图"在非 0 rank 上同样可被算子访问；`_flatten_pa_cache(host_cache)` 只是 `reshape`（view，不是 copy），融合算子因此能零拷贝读 host 侧全量 KV。
2. **VA 开销随 TP 线性膨胀**：每个 rank 都要预留 `reserve_size` 的虚拟地址窗口（只有 rank0 有物理页）。TP=16 + 128 GB 的配置下，进程要预留约 **2 TB VA**。这也是换 Mooncake 的动机之一——Mooncake 共享段明确**不要求跨 rank 虚拟地址一致**（owner 分配物理页并导出 handle，其余 rank `Import + MapMem`），每 rank 只预留 1×size。

**未在 #15642 完成 / 明确留作后续**（注意：PR body 是空的，说明写在 commit message 里）：
- **Ascend 950（arch35）端到端未开启**，commit 原文：`End-to-end integration on Ascend 950 is not enabled yet and will be followed up in a subsequent PR.` 代码侧已核实：`csrc/build_aclnn.sh` 把 `fused_sparse_attention_overlap` 加进了 **`ascend910b` 列表（第 138 行）和 `ascend910_93` 列表（第 192 行），但没有加进 `ascend950` 列表（第 196–236 行）**——arch35 的 kernel 源码与 `AddConfig("ascend950")` 都在，只是这个算子从不为 950 构建/安装；
- 910C（`ascend910_93`，arch22）端到端已验证，H2D onload 与 attention 计算重叠（叠加 MM1/softmax，3 级流水 + 4 槽 GM ring）；
- 仅支持 TP，**不支持 CP / PP**（`_validate_preconditions` 里显式拒绝）；
- 仅 MRV1（不支持 `use_v2_model_runner`）；
- 仅 MemFabric 后端；
- 新增环境变量 **0 个**；唯一新配置键是 `sparse_kv_offload_config.use_fused_overlap`（默认 `False`）。

#### 4.1.1 融合算子契约速查（理解 PR2 的必要前置）

`FusedSparseAttentionOverlap`（aclnn 名 `aclnnFusedSparseAttentionOverlap`，torch 名 `npu_fused_sparse_attention_overlap`）在 `op_host/..._def.cpp` 里声明的**输入 14 个 / 输出 2 个 / attr 5 个**：

| 类别 | 关键项 | 说明 |
| :--- | :--- | :--- |
| 输入（语义名） | `query` / `key` / `value` / `sparse_indices` / `block_table` / `actual_seq_lengths_query` / `actual_seq_lengths_kv` / `query_rope` / `key_rope` | aclnn 层用通用名；torch adapter 把 `key = full_kv_cache.unsqueeze(2)`、`value = key`（MLA-absorb 下 V 与 K-nope 同源）|
| 输入（offload 专属，`selection_*` 五个） | `selection_k_rope`（热缓冲的位置分量）/ `selection_kv_cache`（热缓冲的压缩主体）/ `selection_kv_block_table`（热缓冲的逻辑块→物理块对照表）/ `selection_kv_block_status`（每个 slot 存的是哪个 token、以及本行有效数量）/ `selection_membership_map`（token→slot 映射 + 控制字段，见 §5） | **全部原地更新**（就地修改传入的张量；torch schema 里标 `(a!)`–`(e!)` 表示"这个参数会被改写"）|
| **输出** | `attention_out` | 注意力计算结果；宽度 = `full_kv_cache` 的最后一维（512，只含 NoPE，不含位置编码部分）。`torch_binding_meta.cpp:415-421` 专门做宽度覆盖，注释写明"否则 graph capture（图录制）会按错误宽度分配输出" |
| **输出** | `selection_kv_actual_seq` | int32 类型、形状 `(row,)`（每行一个数），表示**每行有效/常驻的 top-k 数量**。它不是普通输出，而是**副作用别名**（借用了输入张量里的一格内存，与它指向同一处）：`B==1`（一个 batch）且 status（`selection_kv_block_status`）内存连续时，`ConstructSelectionKvActualSeqForSideEffect` 直接**alias 到 `status[topk]` 这一格**（kernel 侧写的是 `selectionKvActualSeqGm_.SetValue(selectionRow, validTopkNum)`）。这与 Python 侧 `_fsa_selection_status_stride(topk) = align8(topk+1)`（状态行宽 = 把 topk+1 向上取整到 8 的倍数）一一对应 |
| attr（属性参数） | `scale_value`（注意力的缩放系数）/ `sparse_block_size` / `selection_topk_block_size` / `layout_query`（query 的内存排布格式）/ `layout_kv`（KV 的排布格式）/ `sparse_mode`（稀疏模式） | 框架侧固定传 `layout_query="TND"`、`layout_kv="PA_BSND"`（PA = Paged Attention，分页注意力）、`sparse_mode=3`、`sparse_block_size=1`、`selection_topk_block_size=1` |

**两组 GPU/NPU 侧状态量的真实语义**（容易看反，务必记住）：

| 状态 | 位置 | 布局与语义 |
| :--- | :--- | :--- |
| `selection_kv_block_status` | **纯 NPU tensor**（干净的显存张量，不在 host pool 里） | int32，形状 `(row, 1, align8(topk+1))`（`align8(x)` = 把 x 向上取整到 8 的倍数）；前 `topk` 格存**常驻 token 的绝对编号**，**第 `topk` 格存"本行有效数量"** |
| `selection_kv_block_table` | NPU（显存） | **恒等映射**（第 i 行就指向第 i 块，不做真实分配）：`torch.arange(...).reshape(row_capacity, cache_blocks_per_row)` ⇒ 热缓冲其实是一整块连续 buffer（缓冲区），**根本没有分配器** |
| `selection_membership_map` | #15642 在 MemFabric host pool，PR2 起在独立 Mooncake 段 | int16，形状 `(row, 16400)`：`[0,16376)` = token→slot+1 映射；`[16376,16384)` 对齐填充（为让控制区落在 16 的倍数上）；`[16384,16392)` = **8 个 int16 控制字**（= 4 个 int32，方便 Python 用 int32 视图一次性整块清空）；`[16392,16400)` 尾部填充 ⇒ stride（行的步长）= 16400 |

控制字（Python 初始化见 manager `_init_fused_overlap_membership_control`）：
`[0]` membership-ready `0x5A4D`、`[1]` **external-plan-ready `0x5A45`**、`[2]` plan count、`[3]` **plan offset = 16384 − topk**、`[4]` 外部物理 row 覆盖、`[5]` direct-layout marker `0x5A44`、`[6]` direct row stride、`[7]` paired-copy marker `0x5A56`。
**注意：plan 值本身编码 HIT/MISS**，没有单独的"next slot"字段。基准实现（`pr15642:sparse_kv_offload.cpp:195-211`）的精确编码是：

```cpp
const bool is_current_token = visible_seq_lens != nullptr && token == visible_seq_len - 1;
encoded_plan_row[pos] = static_cast<int16_t>(is_current_token ? slot + 1 : -(slot + 1));
```

即 **plan 值 = ±(slot+1)，符号区分"当前 token"与"miss 装入的 token"**；算子侧按 `planValue > 0` 判定走哪条取数路径（arch35 `service_vector_mla.h:445-455`）。另外 planner 每行**预留一个 slot 给当前 token**（`resident_capacity = capacity - 1`），并通过 `current_token_slots[logical_row] = physical_row * capacity + current_token_slot` 导出线性槽位给 Python 做 `npu_scatter_nd_update_` 注入。

**算子内部三个架构事实**：

1. **arch22（910C，昇腾 910C 芯片的算子架构代号）与 arch35（950）行为不同**：**arch35 没有算子内 planner**，只消费 external plan（由框架侧 C++ LRU 产出的外部计划）；arch22 两种模式都支持（`UseSetResidentSelection()` 为真走 `ProcessSetResidentSelectionRow`，为假时 `RunAllCoreSelectionUpdate` 直接返回，退化成"gather 仍融合进 `kvMergeGm_`"）。**这是读 PR2 的动力之一**：external plan 一旦成为唯一路径，plan 的产出/发布就必须跨内存域可靠。
2. **模板只编译一种组合**：`template_tiling_key.h:38-49` 的 `ASCENDC_TPL_SEL`（编译期模板选择）只注册 `FLASH_DECODE=0 / LAYOUT_T=TND / KV_LAYOUT_T=PA_BSND / TEMPLATE_MODE=V_TEMPLATE / IS_SPLIT_G=0`（注释点名 GLM-5.2 decode）。host tiling（主机侧的切分策略计算）却可以合法产出别的 key ⇒ 换模型/换 layout（内存排布）会踩空模板。
3. **重叠实现**：3 级软流水（AIC = AI Cube 单元负责矩阵乘，算 MM1/MM2；AIV = AI Vector 单元负责向量运算，同时做 `MergeKv` + `CopyOutSelectionUpdateFromKvMerge`；两者用 `CrossCoreSetFlag/WaitFlag` 同步），GM（Global Memory，显存）侧用 4 个槽位的 ring（环形缓冲区，`MERGE_CACHE_GM_BUF_NUM = 4`），UB（Unified Buffer，片上缓冲）侧 `mergeMte3Idx % 2` 乒乓（两个缓冲交替用），Cube 侧 L1 三缓冲。

#### 4.1.2 一处"配置名不一致"的小坑（读错误信息时会迷惑）

代码里 `use_fused_overlap` 的**实际配置路径是 `additional_config.sparse_kv_offload_config`**（即写在启动参数 `--additional-config` 的 JSON 里、`sparse_kv_offload_config` 这一节下），但 manager 里两处报错信息（如 `"mapped membership allocation requires kv_offload_decode_config.use_fused_overlap=true"`）写的是**过时的段名 `kv_offload_decode_config`**。看到这个错误信息时按 `sparse_kv_offload_config` 去改。

#### 4.1.3 一处"当前不可达"的代码（与失效语义有关）

#15642 里 `prepare_fused_overlap_external_plan` 的三条返回路径**都返回 `True`**，因此 `sfa_kv_offload.py` 中 `if not external_plan_prepared: self._invalidate_fused_overlap_selection_rows(...)` 这条分支**永远不执行**，`_invalidate_fused_overlap_selection_rows` 在当前代码路径上不可达 ⇒ **NPU 侧 status/membership 的失效完全依赖 C++ planner 的 epoch / 属主机制**（`lru_last_req_ids` 属主变更整行 reset + per-thread epoch tag 标记）。这一点在排查"复读/乱码"时值得先确认是否构成语义缺口。

### 4.2 PR1 #15883（`host_backend` 与资源层抽象）

**文件（11 个，+1216/-86）**

| 文件 | 作用 |
| :--- | :--- |
| `vllm_ascend/distributed/kv_transfer/sparse_kv_offload/mooncake_host_pool.py`（**新增 219 行**） | `HostPoolTopology` / `HostMemoryRegion` / `allocate_mooncake_host_region` / `MooncakeHostPool` |
| `sparse_kv_offload_manager.py` | 抽出 `HostKVAllocator` Protocol（协议/接口定义：规定"分配器"必须提供哪几个方法）；`MemFabricHostKVAllocator`；`prepare_host_kv_allocation`（预备分配）/ `allocate_host_kv_tensors`（真正分配）/ `close`（释放）；`register_kv_caches` 按后端分支（Mooncake = 各 rank 用本地视图；MemFabric = tp0（0 号卡）广播指针）|
| `vllm_ascend/ascend_config.py` | `sparse_kv_offload_config.host_backend: Literal["memfabric","mooncake"] = "memfabric"` + 校验（mooncake ⇒ 必须 `use_fused_overlap=true` 且 `keep_device_kv_cache=false`）|
| `vllm_ascend/envs.py` | 新增环境变量 `VLLM_ASCEND_SKIP_MIGRATEPAGES`（跳过 NUMA migratepages：不再把进程页面迁移到 NPU 所在 NUMA 节点）|
| `vllm_ascend/cpu_binding.py` | 配合 SKIP_MIGRATEPAGES（只跳过页迁移），CPU 绑核（把线程绑到固定 CPU 核）仍然启用 |
| `vllm_ascend/worker/model_runner_v1.py` | 调用 `prepare_host_kv_allocation(device_id, dp_rank)`（dp_rank = 数据并行的编号）；新增 `shutdown()` 调 `manager.close()` 释放；改用 `torch.npu.current_device()` 取卡号 |
| `mypy.ini` + 4 个 UT 文件 | mypy（Python 静态类型检查工具）白名单（`[mypy-mooncake.*] ignore_missing_imports`）+ 新增/扩展单测。UT 全部用 stub（桩：`sys.modules.setdefault("mooncake.shared_segment", stub)` 塞一个假模块），**不需要真机/Mooncake 环境** |

**PR1 实际调用的 Mooncake API（精确符号，别搞混）**

```python
# vllm_ascend/distributed/kv_transfer/sparse_kv_offload/mooncake_host_pool.py
from mooncake.shared_segment import shared_segment_supported, create_shared_segment
```

| 不是 | 是 |
| :--- | :--- |
| `mooncake.store.MooncakeDistributedStore`（这是 `kv_pool/ascend_store/backend/mooncake_backend.py` 用的） | `mooncake.shared_segment.create_shared_segment` |
| `mooncake_vllm_adaptor` | `mooncake.engine.TransferEngine`（这是 `kv_transfer/utils/mooncake_transfer_engine.py` 用的，属 PR3 范围） |

```python
# 只接受这一种模式（早期版本还探测 mmap=False 的 Ascend VMM 路径，head 已删除）
if shared_segment_supported(mmap=True, host_register=True):
    return True, True
raise RuntimeError("Mooncake shared_segment cannot expose an NPU-addressable address")

segment = create_shared_segment(
    segment_name,                                        # f"sparse_kv_offload_host_pool_dp{dp_rank}"
    blocks={"pool": {"count": 1, "shape": (allocation_size_bytes,), "dtype": torch.int8}},
    world_size=topology.tp_size, rank_id=topology.tp_rank,
    owner_rank=topology.owner_rank,                      # 固定 0
    device_id=topology.device_id,
    tp_group=topology.tp_group,                          # tp_size > 1 时必填
    mmap=True, host_register=True,
)
raw = segment.tensors("pool")[0].reshape(-1)             # ← 期望拿到 NPU 视图
```

**关键约束与风险（PR1 阶段）**

| 项 | 说明 |
| :--- | :--- |
| 版本/构建依赖 | 需要 Mooncake ≥ PR #3285（2026-08-13 合入，`mooncake/shared_segment.py`），且构建时定义 `USE_ASCEND_DIRECT`，否则 `shared_segment_supported(mmap=True, host_register=True)` 返回 `False` → 启动即抛 `RuntimeError`。仓库 `requirements*.txt` 未声明 `mooncake`，属**可选依赖** |
| 设备视图断言 | `allocate_mooncake_host_region` 会断言返回的 tensor 不是 CPU tensor（`device.type != "cpu"`），否则报 "Mooncake shared segment did not expose an NPU tensor" |
| `close()` 非确定性 | `HostMemoryRegion.release_callback` 恒为 `None`（没有注册真正的释放回调），`release()` 只把 handle 置 `None`；真正 unmap（解除内存映射）依赖 Python 引用计数（`region.tensor` 与各层 `narrow()` 出的视图都持有段对象）。`close()` 里的 `_closed=True` 只能阻止"再分配" |
| 分支只看配置 | `uses_local_views = self.host_backend == "mooncake"`（不是 `isinstance(allocator, MooncakeHostPool)`）。若 `prepare_host_kv_allocation` 没跑/失败，会走到 local-views 分支读空 `k_caches_cpu` → `IndexError`，目前靠调用顺序隐式保证（PR3 把它改成 `self._uses_mooncake_host_pool()` 实例判定）|
| 自动置位的时序 | `VLLM_ASCEND_SKIP_MIGRATEPAGES` 在**建段成功之后**才被写进 `os.environ`，而 `bind_cpus`（含 `migratepages`）发生在更早的 `compile_or_warm_up_model`（编译/预热模型，`worker.py:937-945`），host pool 要到 `initialize_kv_cache`（初始化 KV 缓存，`worker.py:1118-1129`）才创建 ⇒ **自动置位只对之后的调用生效，本进程第一轮 migratepages 挡不住**。要真正跳过，必须在**启动前先 export** |
| 文档未同步 | `docs/` 里 `host_backend` 零命中，新版用户文档只列 3 个键，新环境变量无文档 |
| 真机验证 | 无 e2e（端到端）/真机验证（PR body 自述 "CI: pending pipeline verification"，CI = 持续集成）；TP>1 的真机行为、`offload.sparse_copy` 是否接受 HostRegister 出来的 device VA（设备虚拟地址）、多个 DP 同机时段名 `..._dp{dp}` 是否冲突，均未验证 |

**核心接口（PR1 定义，PR2/PR3 全部建立其上）**

```python
class HostKVAllocator(typing.Protocol):
    def allocate_tensors(self, sizes: list[int], alignment: int) -> list[torch.Tensor | None]: ...
    def close(self) -> None: ...
```

```python
# MooncakeHostPool —— 一个 Mooncake 共享段之上的"对齐 bump 分配器"（bump = 顺序往后推游标，不做回收；对齐见 §0.1③）
class MooncakeHostPool:
    region: HostMemoryRegion          # 承载整段共享内存，含 NPU 视图
    topology: HostPoolTopology        # (tp_rank, tp_size, owner_rank=0, device_id, dp_rank, tp_group)
    _offset: int                      # bump 游标

    @classmethod
    def allocate(cls, *, size_bytes, alignment, topology) -> MooncakeHostPool: ...
    @property
    def data_ptr(self) -> int: ...    # 整段 host pool 的 NPU 视图基址
    @property
    def nbytes(self) -> int: ...
    def allocate_tensors(self, sizes, alignment) -> list[torch.Tensor]:   # narrow() 出视图
    def close(self) -> None: ...
```

**Mooncake 后端 vs MemFabric 后端的四个语义差异**

| 维度 | MemFabric（默认） | Mooncake（PR1 引入） |
| :--- | :--- | :--- |
| 分配位置（内存从哪来） | `offload.Scene.SHARED` 池（Scene.SHARED = 共享场景），`alloc_size`（真正分配的字节数）**只有 rank0 非零**（用 `offload.empty(pin_memory=True)` 取锁页内存） | `owner_rank=0` 创建段，**全体 TP rank** 各自 `mmap`（映射）+ `HostRegister`（注册成设备可访问）出**本地** NPU 视图 |
| 非 0 rank 怎么拿到这块内存 | **广播 rank0 的裸指针**并当本地地址用（依赖对称 GVA；再用 `_restore_*_tensor(ptr, shape)` 造一个"不拥有内存、只借用"的视图） | 直接用**自己映射出来的**本地视图（`uses_local_views = host_backend == "mooncake"`）；不再需要 `_restore_bfloat16_tensor` |
| 虚拟地址（VA）开销 | **每个 rank 都要预留 1×`reserve_size` 的地址空间**，TP16 + 128 GB ≈ 2 TB VA（很浪费） | 每 rank 只预留 **1×size**（不要求跨 rank 的虚拟地址一致） |
| CPU 上的 planner 能否直接写 membership map | 能（planner 与算子指向同一块主机内存，CPU 可写、NPU 可读） | **不能**（对 CPU 而言它是 device tensor）→ 需要 PR2 的 staging（两跳中转）|
| 这块池的物理含义 | 本机进程/设备之间共享 | **可被 Mooncake 注册、可被远端 RDMA/ROCE 直接寻址的共享段** → 这是 PR3 做 D2RH 的基础 |
| 各 rank 布局一致性怎么保证 | 靠"大家虚拟地址本来就一样"隐式保证 | 靠**字节布局 + 段内偏移**显式校验（PR3 的 `describe_local_views`）|

**PR1 明确声明自己"不通数据"**（PR 正文原文）：

> ⚠️ **Do NOT set `host_backend: "mooncake"`.** The Mooncake backend is incomplete in this PR: decode-side KV write-back / membership staging and PD cross-node transfer are delivered by follow-up PRs. With this setting, only the host pool is created; the data path is not functional.

也就是说：**PR1 只把"盆"换了，盆里的水（membership plan 的写入、current KV 的写回、P→D 的搬运）由 PR2 和 PR3 接上。**

**⚠️ 一处"PR 描述与代码不一致"需要知道**：PR1 body 提到 "owner-rank Transfer Engine registration hooks"，但**该功能在 head 已被整体删除**（首提交 `ae55632d6` 有 `register()/unregister()/is_owner/register_location`，head `a59d9581a` 的提交信息是 `drop Mooncake pool register/unregister; engine registration moves to the Mooncake connector`）。所以**注册这件事最终落在 PR3 的 `GlobalTE.register_buffer(..., locations)` 上**，PR1 只负责"建段 + 映射 + bump 分配"。同理 body 里的 UT 计数（10/7）与实际（8/6）不符。

### 4.3 PR2 #16219（单机数据通路：membership staging + 图安全写回）★重点

**文件（17 个，+2374/-125）**

| 文件 | 变化 |
| :--- | :--- |
| `vllm_ascend/distributed/kv_transfer/sparse_kv_offload/sparse_kv_offload_manager.py` | **+680/-125**。新增 membership staging 三件套（planner 蓝图 / NPU 中转 / 段内 map）、`_offload_new_kv_via_index_copy`（用 index_copy_ 写回）、`_prepare_current_kv_index_copy_descriptors`（准备索引描述符）、`get_local_host_kv_views`（取本 rank 的本地视图）、`_register_local_mooncake_views`（注册本地视图）|
| `csrc/torch_binding.cpp` | **+186**。`sparse_kv_compute_current_kv_index_copy_descriptors` 与 `sparse_kv_enqueue_current_kv_index_copy_descriptors`（两个 C++ 算子：算 / 挂索引描述符），含 ACL graph 生命周期管理（保证回调对象活到图销毁）|
| `vllm_ascend/attention/sfa_kv_offload.py` | 1 处关键判定：`selection_membership_map.device.type != "cpu"` → `manager.is_fused_membership_storage(tensor)` |
| `vllm_ascend/distributed/kv_transfer/sparse_kv_offload/mooncake_host_pool.py` | 由 PR2 一起带过来（与 PR1 同内容，因 PR2 基于 PR1） |
| `tests/ut/kv_offload/test_sparse_kv_offload_external_plan.py`（+229）<br>`tests/ut/kv_offload/test_sparse_kv_offload_index_copy.py`（**新增** 233）<br>`tests/e2e/nightly/.../test_sparse_kv_offload.py` | 外部 plan / index_copy / C++ 描述符 + graph 重放 |

#### PR2 解决的两件事

**① Fused-overlap membership plan staging（两跳发布）**

完整流程见下图（MemFabric 路径零拷贝；Mooncake 路径两跳）：

![★PR2：membership plan 的"两跳发布"](plantuml/img/06-two-hop-publish.png)

> **图 6** · ★PR2：membership plan 的"两跳发布"—— MemFabric 路径零拷贝，Mooncake 路径两跳。源文件：`plantuml/06-two-hop-publish.puml`

背景：membership map 是 CPU 上的 LRU planner（规划器）与 NPU 上的融合算子之间的"共享契约"（一张记录 token 在不在热缓冲、在哪个槽位的 int16 表）。MemFabric 下两侧就是同一块 pinned CPU 内存，**CPU 直接写、NPU 直接读**。而在 Mooncake 段里，这块 map 是 **NPU 可寻址的 device tensor**（对 CPU 而言像显存里的张量），CPU planner 没法直接写它。

PR2 的做法（`allocate_fused_overlap_membership_map`，manager 第 790–882 行）：

```python
if self._requires_fused_membership_staging():          # == 用的是 MooncakeHostPool
    region = allocate_mooncake_host_region(            # 独立的 Mooncake 段
        size_bytes=row_capacity * FSA_SELECTION_MEMBERSHIP_STORAGE_INT16_COUNT * 2,
        alignment=_CPU_CACHE_ALIGNMENT,
        topology=allocator.topology,
        name="sparse_kv_offload_fused_membership",
    )
    membership_map = region.tensor.view(torch.int16).view(shape)   # ← 算子读的那块（NPU 视图）
    if self.tp_rank == 0:
        plan_width = self.topk + FSA_SELECTION_MEMBERSHIP_CONTROL_INT16_COUNT
        planner_map = torch.empty([row_capacity, plan_width],
                                  dtype=torch.int16, device="cpu", pin_memory=True)  # ← planner 写的蓝图
        self.fused_overlap_membership_plan_device_staging = torch.empty(
            planner_map.shape, dtype=torch.int16, device=membership_map.device)       # ← 中转过桥
        self._init_fused_overlap_plan_staging(planner_map)
        self._init_fused_overlap_membership_control(membership_map)
        torch_npu.npu.current_stream().synchronize()
    self.tp_group.barrier()
    self.fused_overlap_membership_region = region
    self.fused_overlap_planner_membership_map = planner_map
else:  # MemFabric：保持老路径 —— planner 与算子共用同一块内存
    membership_map = offload.empty([...], dtype=torch.int16, pin_memory=True).view(shape)  # tp0
    ... owner_ptr broadcast → _restore_int16_tensor(...)  # 非 0 rank
    self.fused_overlap_planner_membership_map = membership_map
```

发布路径（`prepare_fused_overlap_external_plan` 内嵌 `publish_plan`，第 1842–1861 行）：

```python
def publish_plan(non_blocking: bool) -> None:
    if planner_storage.data_ptr() == plan_storage.data_ptr():
        return                                  # MemFabric：同一块内存，无需搬运
    device_plan_storage = device_staging[:num_tokens, :plan_width]
    device_plan_storage.copy_(planner_storage, non_blocking=non_blocking)   # 跳 1：CPU → NPU staging
    plan_storage.copy_(device_plan_storage,  non_blocking=non_blocking)     # 跳 2：NPU staging → Mooncake 段
```

**为什么是两跳而不是一跳？** 因为 `planner_storage`（planner 写的蓝图）在 CPU 锁页内存里、`plan_storage`（Mooncake 段内的视图）在 device 侧，而 torch 其实也支持"锁页内存→设备"的直接拷贝；两跳的真正价值在于：
- 第一跳走标准的 pinned H2D（锁页内存→设备）路径，可以 `non_blocking`（异步不阻塞）、**可以被 graph 捕获**（录进那张静态图里）；
- 第二跳是 **NPU 内部的 device-to-device 拷贝**，把数据写进 **Mooncake 段这块"NPU 可寻址"的地址空间**，从而让融合算子在 kernel（跑在卡上的计算程序）里直接就能读到——而不是让算子自己去走 PCIe 总线读主机内存；
- 这条路径还保留了"TP0（0 号卡）算、其他 rank 只消费"的分工：rank0 执行 `broadcast(fused_plan_metadata_npu, src=0)`（广播元数据）后，其他 rank 只需读段里已经发布好的 plan。

同时新增了严格的**plan 输入校验**（`_validate_fused_overlap_external_plan_inputs`）：校验 `topk_indices`（top-k 编号表）形状必须是 `(num_tokens, topk)`（只有一个 head，已压平）、`req_ids`（每行属于哪个请求）/ `stable_prefix_lens`（稳定前缀长度）/ `visible_seq_lens`（可见序列长度）必须都是 `(num_tokens,)` 一维、membership 存储必须是 `int16` 且被 `is_fused_membership_storage()` 认定为本进程的那块，否则直接抛错——**这些都是长输入/大 batch（大批次）场景下"数据错位"的高发点**。

**② Graph-safe current-token KV writeback**

背景：Decode 每一步新产生的 K/V（Key/Value）必须写回主机内存池，否则后续请求/后续层从主机内存里取不到这些数据。MemFabric 下走 `offload.sparse_copy(src_ptrs, dst_ptrs, lengths, size, device)`（给一组"源指针+目标指针+长度"就能批量搬），指针数组在主机侧由 LRU 逻辑（"最久没用先淘汰"的缓存策略代码）算好，**不需要索引张量**。

Mooncake 下目标变成 NPU 视图，改用 `index_select` + `index_copy_`：

```python
# manager._offload_new_kv_via_index_copy  (capturing=True 分支)
current_kv_ready = torch_npu.npu.current_stream().record_event()
with torch_npu.npu.stream(self.current_kv_save_stream):
    self.current_kv_save_stream.wait_event(current_kv_ready)
    flat_host_k.index_copy_(0, self.d2h_dst_idx_npu,
                            k_rows.index_select(0, self.d2h_src_idx_npu))
    flat_host_v.index_copy_(0, self.d2h_dst_idx_npu,
                            v_rows.index_select(0, self.d2h_src_idx_npu))
```

完整流程见下图（4 个阶段，重点是 host callback 与 `aclmdlRI` 生命周期绑定）：

![★PR2：图模式（ACL graph）下 current KV 写回的 4 个阶段](plantuml/img/07-graph-writeback.png)

> **图 7** · ★PR2：图模式（ACL graph）下 current KV 写回的 4 个阶段 —— 描述符每帧由 host callback 刷新。源文件：`plantuml/07-graph-writeback.puml`

难点在于 **graph 重放**（重放 = 反复执行提前录好的那张静态图）：
- `index_copy_` 需要**形状固定**的 `src_idx`（源索引）/ `dst_idx`（目标索引）两个张量；
- 而每步的有效 token 数（有效词元数）和 `slot_mapping`（"这一步新生成的 token 该写到哪个槽位"的映射表）内容都在变，`nonzero()`（找出非零位置）返回的长度是动态的 ⇒ **无法在 Python 侧录进图里**；
- 图一旦录好，就不能再用 Python 逻辑去刷新索引。

PR2 的解法（`csrc/torch_binding.cpp` 新增 186 行）：

```cpp
struct CurrentKvIndexCopyPayload {
  at::Tensor slot_mapping, src_idx_buffer, dst_idx_buffer, count_buffer;
  int64_t num_actual_tokens, max_num_tokens, num_host_slots;
  bool delete_after_run;
};

void build_current_kv_index_copy_descriptors(CurrentKvIndexCopyPayload* payload) noexcept {
  const auto* slots = payload->slot_mapping.data_ptr<int64_t>();
  int32_t num_copies = 0;
  for (int64_t i = 0; i < payload->num_actual_tokens; ++i) {
    const int64_t slot = slots[i];
    if (slot < 0 || slot >= payload->num_host_slots) continue;   // 过滤非法 slot
    src_idx[num_copies] = i;
    dst_idx[num_copies] = slot;
    ++num_copies;
  }
  // padding（填充）到 max_num_tokens：用最后一条合法项重复填充 → 形状稳定且重复执行结果一致
  if (num_copies > 0) {
    for (int64_t i = num_copies; i < payload->max_num_tokens; ++i) {
      src_idx[i] = src_idx[num_copies - 1];
      dst_idx[i] = dst_idx[num_copies - 1];
    }
  } else {
    std::fill_n(src_idx, payload->max_num_tokens, 0);
    std::fill_n(dst_idx, payload->max_num_tokens, 0);
  }
  count[0] = num_copies;
}
```

```cpp
void enqueue_current_kv_index_copy_descriptors(...) {
  auto payload = make_current_kv_index_copy_payload(...);         // CPU 侧只做一次校验+组包
  aclmdlRICaptureStatus capture_status;
  aclmdlRI model_ri = nullptr;                                    // model_ri = 图运行时实例（一张录好的图的句柄）
  aclmdlRICaptureGetInfo(stream, &capture_status, &model_ri);     // 判断当前是否正在"录制图"
  const bool graph_lifetime = (capture_status == ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE);
  payload->delete_after_run = !graph_lifetime;
  auto* raw_payload = payload.get();
  if (graph_lifetime) {
    raw_payload = retain_current_kv_index_copy_graph_payload(model_ri, std::move(payload));
                                                                    // 绑定到 aclmdlRI 生命周期
  }
  aclrtLaunchHostFunc(stream, current_kv_index_copy_descriptor_callback, raw_payload);
                                                                    // 每帧重放都会执行一次 callback
}
```

三个设计要点：
1. **Host callback（主机回调）在图重放时重新执行**：`aclrtLaunchHostFunc` 入队的这段 CPU 函数会被图一起捕获，并在每次重放时被调用，于是描述符自动刷新为**当帧**的 `slot_mapping`。
2. **生命周期绑定到 `aclmdlRI`**：`aclmdlRIDestroyRegisterCallback` 注册一个销毁回调 `delete_current_kv_index_copy_graph_payloads`，从而同时避免两个方向的问题——"重放时对象已被释放"（use-after-free）和"对象永远不释放"（泄漏）。
3. **每步只生成一次**：`offload_new_kv` 里 `prepare_index_copy_descriptors = (layer_id == 0)`，即只在第 0 层生成描述符，因为所有 decode 层共享同一个 `slot_mapping`。

另外 PR2 把 stream 依赖也补齐了：
- 写回发生在 `current_kv_save_stream`；
- 消费前 `wait_for_current_kv_writeback(capturing)` 让当前流 `wait_stream(current_kv_save_stream)`；
- plan 相关在 `fused_plan_stream` 上，消费前 `wait_stream(fused_plan_stream)`。

PR2 的另一处"小改动、大含义"：`sfa_kv_offload.py` 的缓存失效条件

```python
- or self.selection_membership_map.device.type != "cpu"
+ or not get_sparse_kv_offload_manager().is_fused_membership_storage(self.selection_membership_map)
```

原来"membership map 在 CPU 上"是硬编码假设（MemFabric 语义）；现在改为向后端提问。`is_fused_membership_storage()` 在 Mooncake 下比较 **`data_ptr` 是否等于本进程的 `fused_overlap_membership_map`**（因为 Mooncake 段各 rank 虚拟地址可能不同，不能靠"是不是 CPU tensor"判断）。

### 4.4 PR3 #16378（跨节点数据通路：blockwise Mooncake DSA transfer）★重点

**文件（27 个，+5550/-229；当前状态 draft）**

| 文件 | 变化 |
| :--- | :--- |
| `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py` | **+1373**。新增 `_MooncakeDsaDecodeScheduler`（Decode 侧调度器）、`_execute_dsa_receive`（执行接收）/ `_handle_dsa_request`（处理请求）、`_dsa_consumer_register_regions` / `_dsa_producer_register_regions`（消费端/生产端内存注册）/ `_build_dsa_local_layouts`（构建本地布局）、`_plan_dsa_endpoints`（端点规划）、`_dispatch_dsa_commands`（下发命令）等 |
| `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_dsa_metadata.py`（**新增 248 行**） | `RemoteEndpoint`（远端端点：host/port/engine_id）/ `RemoteSource`（远端源信息）/ `DsaStepRequest`（一步接收请求）/ `DsaConnectorMetadata`（连接器元数据）/ `DsaLocalResult`（本 rank 结果）/ `DsaWorkerResultMetadata`（worker 结果汇总）/ `DsaTransferPhase`（传输阶段：indexer 还是 main，用于报错定位）|
| `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_dsa_transfer.py`（**新增 294 行**） | `DsaCacheLayout`（一份 cache 的地址布局描述）/ `DsaRegisterAtom`（一段"不可跨注册边界"的地址区间）/ `collect_bounded_register_regions`（合并成不超过上限的注册区间）/ `build_component_read`（把 token 区间翻译成"本地地址+远端地址+字节数"三元组）/ `coalesce_transfer_lists`（两端都连续才合并）/ `split_transfer_lists_at_region_boundaries`（在注册边界处切分）|
| `vllm_ascend/distributed/kv_transfer/utils/mooncake_transfer_engine.py` | `register_buffer(ptrs, sizes, locations=None)`（多了一个 locations 参数，用来声明每块内存的地址类型）；`unregister_buffer()`（注销，之前只注册从不注销）；`_registered_regions` 记账（记录注册过哪些区间便于回收）|
| `sparse_kv_offload_manager.py` | `get_mooncake_host_pool()`（取主机内存池）、`get_local_host_kv_views()`（取本 rank 本地视图）、`_register_local_mooncake_views()`（**把"注册本地视图"从"各 rank 各读各的指针"升级为"all_gather 汇总后校验段内偏移是否一致"**，all_gather = 每个人都交出数据、最后人人拿到全量）|
| `sparse_kv_offload/mooncake_host_pool.py` | `describe_local_views()`：只暴露**段内偏移 + 形状/步长/数据类型**，不暴露绝对虚拟地址（因为各 rank 的虚拟地址可能不同，只比偏移才安全）；`HostMemoryRegion.segment_offset` |
| 6 个 UT 文件 | metadata / scheduler / shared_pool / transfer / connector / transfer_engine |

#### ① 为什么需要它：D 的两种 cache 在两个不同的内存域

![★PR3：D 的两种 cache 落在两个不同的内存域](plantuml/img/12-dsa-memory-domain.png)

> **图 8** · ★PR3：D 的两种 cache 落在两个不同的内存域 —— Main 走 D2RH，Indexer 走 D2D。源文件：`plantuml/12-dsa-memory-domain.puml`

这要求同时具备：
1. **两种 destination（数据落点）语义**在一次请求里混用——一个是"远端的主机内存"，一个是"本端的显存"；
2. **Mooncake 注册能区分 location**（注册时声明这块内存是什么类型、挂在哪）：D 的 host pool 用 `location="npu:<device_id>"`（HostRegister 出来的 device VA（设备虚拟地址）+ 宿主的卡号），Indexer HBM 与 P 的源 HBM 用 `"*"`（通配，让 TE 自己判断）。

```python
# utils/mooncake_transfer_engine.py
_WILDCARD_LOCATION = "*"

def register_buffer(self, ptrs, sizes, locations=None) -> None:
    normalized_locations = locations if locations is not None else [_WILDCARD_LOCATION] * len(ptrs)
    regions = list(zip(ptrs, sizes, normalized_locations))
    ...
    for ptr, size, location in regions:
        ret_value = self.transfer_engine.register_memory(ptr, size, location)   # ← 三个参数
```

```python
# mooncake_connector.py::_dsa_consumer_register_regions（D 侧）
host_location = f"npu:{pool.topology.device_id}"                 # Main → Host pool
atoms.append(DsaRegisterAtom(layout.base, end, host_location, ("host", pool_start)))
...
atoms.append(DsaRegisterAtom(layout.base, end, "*", ("hbm", storage_start)))   # Indexer → HBM
```

#### ② 注册边界安全（`mooncake_dsa_transfer.py`）

Mooncake 单次注册有大小限制（本 PR 取 64 GiB，GiB = 1024³ 字节），所以**不能把整块内存一次性丢过去**，必须：

```python
MAX_REGISTER_MEMORY_BYTES = 64 * 1024**3

def collect_bounded_register_regions(atoms, *, max_region_bytes=MAX_REGISTER_MEMORY_BYTES):
    """Merge atoms per allocation without cutting an atom at a region edge."""
    # 1) 按 (allocation, location) 分组 → 组内排序
    # 2) 丢弃完全重复的 alias（alias = 别名；共享层可能把同一个张量暴露多次）
    # 3) 合并有重叠的 atom（地址区间）为一个 cluster（簇）；若 cluster 超限，要求每个 atom 恰好落在
    #    确定性 chunk（分块）内，否则直接报错（"overlapping DSA register atoms cross a registration boundary"）
    # 4) 单体超限 → 以 atom 基址为锚点切确定性 chunk（切分位置固定，保证传输侧用同一套边界）
    # 5) 否则顺序合并到不超限为止
```

对应的 transfer 侧也要在**两端**的注册边界切分：

```python
def split_transfer_lists_at_region_boundaries(local, remote, lengths, *, local_base, remote_base,
                                              max_region_bytes=MAX_REGISTER_MEMORY_BYTES):
    """Split reads at both endpoints' deterministic registration edges."""
    local_available = max_region_bytes - (dst - local_base) % max_region_bytes
    remote_available = max_region_bytes - (src - remote_base) % max_region_bytes
    part = min(remaining, local_available, remote_available)
```

#### ③ Endpoint 规划（`_plan_dsa_endpoints`）

![★PR3：Endpoint 规划流程](plantuml/img/13-endpoint-plan.png)

> **图 9** · ★PR3：Endpoint 规划 —— 把"源端点"适配成"参与方任务"，并保证逻辑完成不依赖"恰好有人真读了数据"。源文件：`plantuml/13-endpoint-plan.puml`

#### ④ 多 TP / PP / CP 的传输规划（`build_component_read`）

这是 PR3 最核心的算法函数，等价于"把 token 区间 [start_token, end_token) 映射成 (本地地址, 远端地址, 字节数) 三元组列表"：

```python
def build_component_read(local, remote, source_ids, destination_ids,
                         start_token, end_token, *, cp_size, cp_rank,
                         writer_rank, writer_size, indexer, statistics=None):
    """Map token intervals, retaining full-request ordinals and page offsets."""
    remote_span = remote.block_tokens * (cp_size if indexer else 1)
    ...
    token = start_token
    while token < end_token:
        global_source_block, source_offset = divmod(token, remote.block_tokens)
        destination_ordinal, destination_offset = divmod(token, local.block_tokens)
        source_ordinal = global_source_block // cp_size                       # Main 是 CP 分片的
        if indexer:
            source_offset += global_source_block % cp_size * remote.block_tokens   # Indexer 复制了全部 CP 页
        ...
        selected = indexer or (
            global_source_block % cp_size == cp_rank            # 只搬本 CP rank 的源分片
            and destination_ordinal % writer_size == writer_rank   # 目标 block 按 writer 轮转分区
        )
        if selected:
            local_id  = destination_ids[destination_ordinal] * local.scale + local_slot
            remote_id = source_ids[source_ordinal] * remote.scale + remote_slot
            result[0].append(local.base  + local_id  * local.stride + local_offset  * token_bytes)
            result[1].append(remote.base + remote_id * remote.stride + remote_offset * token_bytes)
            result[2].append(count * token_bytes)
        token += count
    merged = coalesce_transfer_lists(*result)      # 两端都连续才合并
    ...
```

要点：
- **Main 按 `destination_ordinal % writer_size == writer_rank` 分区** ⇒ D 的每个 TP rank 只写 host pool 里**不重叠**的一部分；因为 host pool 是共享段，绝不能两个 rank 写同一目标区。
- **Indexer 不做 writer 过滤**（`selected = indexer or (...)`）⇒ 每个参与 rank 都拿到**完整的** indexer 数据（D 侧 indexer cache 是 rank-local 全量副本）。
- **`coalesce_transfer_lists` 只在"两端都连续"时合并**，避免把跨页/跨块的读合成一个错误的连续读。
- **完整性校验**：`source_ordinal >= len(source_ids)`、`destination_ordinal >= len(destination_ids)` 直接报错，`capacity` 越界也报错（`DSA physical block ID exceeds registered capacity`）。

**④ 完成协议（PR3 新增，位于 `mooncake_dsa_metadata.py`）**


```python
class DsaLocalResult:
    request_id: str
    tp_rank: int
    kind: DsaLocalResultKind               # RECEIVE_COMPLETE | TRANSFER_FAILED
    failure_phase: DsaTransferPhase | None # INDEXER_D2D | MAIN_D2RH
```

```python
class DsaWorkerResultMetadata(KVConnectorWorkerMetadata):
    results: tuple[DsaLocalResult, ...]
    def aggregate(self, other) -> DsaWorkerResultMetadata:
        return DsaWorkerResultMetadata(_merge_results((self.results, other.results)))
```

调度侧 `update_connector_output` **等齐所有参与 D rank 的结果**才释放请求；失败时把 `failure_phase` 带回来，只失效**受影响**的 Decode block（`bb7aabd83 fix: invalidate only unhashed DSA blocks`）。

#### ⑤ 注册生命周期（`GlobalTE`）

PR1/PR2 之前 `register_buffer` 是"一次性、不可逆"的：

```python
if self.is_register_buffer:
    return                      # 已经注册过就直接返回，永不 unregister
```

PR3 改成可逆 + 记账：

```python
self._registered_regions: list[tuple[int, int, str]] = []

def unregister_buffer(self) -> None:
    with self.register_buffer_lock:
        if not self._registered_regions:
            self.is_register_buffer = False
            return
        failed_regions, failure_messages = self._unregister_regions(self._registered_regions)  # 逆序
        self._registered_regions = failed_regions
        self.is_register_buffer = bool(failed_regions)
        if failure_messages:
            raise RuntimeError("Mooncake memory unregistration failed for regions: " + "; ".join(failure_messages))
```

并在 `MooncakeConnectorWorker.shutdown()` 里**先排空 worker，再 unregister**：

```python
def shutdown(self) -> None:
    if self._dsa_decode:
        with self._dsa_dispatch_lock:
            self._closing = True          # 阻止新的 DSA 命令入队
    try:
        if self.kv_send_thread is not None: ...stop/join...
        if self.kv_recv_thread is not None: ...queue.join/stop/join/executor.shutdown(wait=True)...
    finally:
        global_te.unregister_buffer()
```

#### ⑥ 开启方式（PR3 的开关）

```json
{
  "kv_connector": "MooncakeConnectorV1",
  "kv_connector_extra_config": {
    "dsa_pd_offload": true,
    "prefill": { "dp_size": 1, "tp_size": 8 },
    "decode":  { "dp_size": 1, "tp_size": 4 }
  }
}
```

D 侧 `additional_config` 还需要：

```json
{
  "sparse_kv_offload_config": {
    "enabled": true,
    "topk_buffer_size": 4096,
    "dram_size_per_dp_GB": 128,
    "use_fused_overlap": true,
    "host_backend": "mooncake"
  }
}
```

并且 `kv_load_failure_policy` 必须为 `"fail"`。构造函数里做硬校验（`mooncake_connector.py:2301-2323`）：

```python
if kv_role == "kv_producer" and offload.enabled:
    raise ValueError("Blockwise DSA Prefill requires sparse KV offload disabled")
if kv_role == "kv_consumer":
    if kv_transfer_config.kv_load_failure_policy != "fail":
        raise ValueError("DSA Host offload requires kv_load_failure_policy='fail'")
    if not offload.enabled:           raise ValueError("Blockwise DSA Decode requires sparse KV offload enabled")
    if not offload.use_fused_overlap: raise ValueError("Blockwise DSA Decode requires fused overlap")
    if offload.host_backend != "mooncake": raise ValueError("Blockwise DSA Decode requires host_backend='mooncake'")
```

`dsa_pd_offload=false` 时，原有非 DSA 的 Mooncake 路径**完全不变**。

---

## 5. 关键数据结构：membership map 与控制字段

membership map 的 `int16` 布局（manager 第 65–87 行；这些常量在 #15642 已存在，PR2 只是让它们"跨内存域"——同一份格式同时被 CPU 与 NPU 两侧使用）：

![membership map 行布局（int16，行宽 16400）](plantuml/img/05-membership-map.png)

> **图 10** · membership map 行布局（int16，行宽 16400）—— 融合算子与 CPU planner 的跨内存域契约。源文件：`plantuml/05-membership-map.puml`

`plan_start = CONTROL_OFFSET - topk`（计划区起始列 = 控制区起点减去 topk），因此**计划缓冲区尾部紧邻控制区**；`plan_width = REQUIRER_COLUMNS - plan_start = topk + 8`（计划区宽度 = topk 个格 + 8 个控制格）。

**Mooncake 路径与 MemFabric 路径的 plan 落点差异（PR2 关键细节）**：

```python
if self.tp_rank == 0:
    if planner_membership_map is None:
        planner_membership_map = selection_membership_map        # 兜底
    elif self._requires_fused_membership_staging():
        planner_plan_start = 0            # ← Mooncake：planner buffer 是"紧凑 plan"，从第 0 列开始
    ...
else:
    planner_storage = plan_storage        # 非 0 rank 只消费，不规划
```

- **MemFabric**：`planner_membership_map is selection_membership_map`（同一块内存），plan 落在 `plan_start` 偏移处，`publish_plan()` 首行就 `return`。
- **Mooncake**：planner 写的是**紧凑的 `[num_tokens, topk+8]`** pinned 缓冲（`planner_plan_start = 0`），再由 `publish_plan()` 搬到段内 `plan_start` 偏移处，并有 `assert planner_storage.is_contiguous()`。

---

## 6. 为什么 PR2 必须存在（把"隐含假设"讲透）

| MemFabric 下的隐含假设 | Mooncake 下为什么不成立 | PR2 的对策 |
| :--- | :--- | :--- |
| host KV 是 **CPU 锁页内存**，CPU planner 与 NPU 算子**共享同一地址** | Mooncake 段暴露的是 **NPU 可寻址 tensor**，CPU 侧不能直接写 | 独立 membership 段 + 锁页内存里的 planner 蓝图 + NPU staging 两跳发布（`publish_plan`） |
| 非 0 rank 通过 **tp0（0 号卡）指针广播 + GVA** 访问同一块内存 | Mooncake **不保证各 rank 的虚拟地址相同**，只保证**段内偏移的布局一致** | `get_local_host_kv_views()` 用各 rank 自己的本地视图；`describe_local_views()` 只交换**偏移量**做一致性校验 |
| "membership map 在 CPU 上"可作为判断条件 | Mooncake 下它在 device（显存侧）上 | `is_fused_membership_storage()` 改成按**本进程的 data_ptr**（数据指针）判断 |
| D2H 写回用 `offload.sparse_copy` + 指针数组，无需索引张量 | 目标是 NPU 视图，`index_copy_` 最自然，但它需要**形状稳定**的索引 | C++ 构建描述符 + `aclrtLaunchHostFunc`（把 CPU 回调挂到流上）+ `aclmdlRI`（图实例）生命周期绑定 |
| eager（即时执行）与 graph（录图执行）逻辑差别不大 | graph 捕获后 **Python 侧无法刷新索引** | Host callback 每帧重放时刷新；描述符每步只在第 0 层生成；padding（填充）用最后一条合法项保证重复执行结果一致 |
| 写回与消费之间靠隐式顺序 | 引入 `current_kv_save_stream`（写回流），顺序不再隐式 | `wait_for_current_kv_writeback()` / `fused_plan_stream.wait_stream` 显式建立依赖（让一条流等另一条流） |

---

## 7. 为什么 PR3 必须存在（跨节点三条硬约束）

![★PR3：跨节点三条硬约束 → 对策映射](plantuml/img/09-pr3-constraints.png)

> **图 11** · ★PR3：跨节点三条硬约束 → 对策映射（对应下方表格）。源文件：`plantuml/09-pr3-constraints.puml`

| 约束 | 说明 | PR3 的对策 |
| :--- | :--- | :--- |
| **混合 destination（落点）** | D 侧 main KV 在主机内存、indexer KV 在显存，一次请求要同时搬两处；且 main 要走 D2RH 而不是 D2D | `register_buffer(ptrs, sizes, locations)`（多传 locations 声明每块内存的地址类型）；主机内存段注册为 `location="npu:<device_id>"`，显存注册为 `"*"`；分两阶段提交（indexer 先、main 后），失败时记录 `DsaTransferPhase`（失败发生在哪个阶段）|
| **P_TP ≠ D_TP（以及 PP/CP）** | 源端与目的端的 block（KV 块）粒度、归属都不同；主机内存池是**共享段**，多个 rank 写必须互不重叠 | `build_component_read` 用 `writer_rank/writer_size` 对 destination ordinal（目标块序号）取模来分区；indexer 不做 writer 过滤（每个 rank 都要全量）；`_plan_dsa_endpoints` 把 P 的 tp×pcp×pp 展开成端点；MTP 层归到最后一个 PP stage（流水线级）|
| **注册有大小上限且必须可回收** | Mooncake 单次注册 ≤ 64 GiB；进程生命周期内反复注册/注销不能泄漏内存 | `collect_bounded_register_regions`（按 allocation 分组合并、按确定规则切块）+ `split_transfer_lists_at_region_boundaries`（传输也按同样边界切分）+ `GlobalTE.unregister_buffer()`（逆序注销、失败的留着待重试）|

---

## 8. 配置项 / 环境变量汇总

### 8.1 `additional_config.sparse_kv_offload_config`

| 字段 | 默认 | 引入 | 含义 |
| :--- | :--- | :--- | :--- |
| `enabled` | `false` | #15642 | 总开关 |
| `topk_buffer_size` | `4096` | #15642 | **每一行**能常驻多少个 slot（槽位），即热缓冲容量（= `lru_resident_capacity`）。必须 ≥ `topk`，且能被 `block_size`（块大小）整除 |
| `dram_size_per_dp_GB` | `128` | #15642 | 每个 DP 组（数据并行组）可用的主机内存上限；同组 TP rank 共享这块池。#15642 起不再无条件按整值申请，而是取 `min(计划值, 上限)`（由 `plan_sparse_kv_offload_memory` 按三重限制算出：NPU 可用显存 / DRAM 上限 / 实际负载需要）|
| `keep_device_kv_cache` | `false` | #15642 | 仅调试用途（PD colocate = P/D 混部在同一节点）；为 `true` 时仍分配全量显存 KV，等于没省显存、无法提升上下文长度。开启时 NPU 限制按 host+device 整页计算，并打 warning（告警）|
| `use_fused_overlap` | `false` | #15642 | 是否启用 `FusedSparseAttentionOverlap` 融合算子路径（让"搬运未命中数据"与"注意力计算"重叠）。#15642 唯一新增的配置键；`docs/` 下**无文档** |
| `host_backend` | `"memfabric"` | **PR1** | `"memfabric"` \| `"mooncake"`；`mooncake` 要求 `use_fused_overlap=true` 且 `keep_device_kv_cache=false`（仅在 `enabled=true` 时施加）。`docs/` 下无文档 |
| `topk` | —（由模型推出） | #15642 | `hf_text_config.index_topk`，只读，不可配置 |

> 小坑：manager 里两处报错信息写的是过时段名 `kv_offload_decode_config.use_fused_overlap`，实际配置路径是 `additional_config.sparse_kv_offload_config`。

### 8.2 `kv_transfer_config`

| 字段 | 引入 | 含义 |
| :--- | :--- | :--- |
| `kv_connector_extra_config.dsa_pd_offload` | **PR3** | `true` 启用 blockwise（按 KV 块）Mooncake DSA 传输；默认 `false` 时走原有 Mooncake 路径（行为完全不变）|
| `kv_connector_extra_config.prefill.{tp_size,dp_size}` | **PR3** | P 侧拓扑（tp_size = 张量并行卡数，dp_size = 数据并行组数），必须显式给出且为正数 |
| `kv_connector_extra_config.decode.{tp_size,dp_size}` | **PR3** | D 侧拓扑（含义同上，描述 Decode 节点）|
| `kv_role` | 既有 | 角色声明：P = `kv_producer`（生产者，"我把 KV 提供出去"），D = `kv_consumer`（消费者，"我去拉 KV"）|
| `kv_load_failure_policy` | 既有（`platform.py`，默认 `"fail"`） | KV 加载失败时的策略。DSA Host offload 场景**必须**为 `"fail"`（PR3 强制校验）——即失败就报错、不允许静默降级；hybrid 模型则拒绝 `"recompute"`（重算）|
| `kv_connector_extra_config.transfer_backend` | 既有 | 传输后端选择。#15642 的 `SfaRemoteD2HConnector` **硬性要求 `"memfabric"`**（P/D 两侧都是），Mooncake 通路由 PR3 的 `MooncakeConnectorV1` 走另一条路（**两条通路并存**）|

### 8.3 环境变量

| 变量 | 引入 | 含义 |
| :--- | :--- | :--- |
| `VLLM_ASCEND_SKIP_MIGRATEPAGES` | **PR1** | `1` 时跳过全进程的 NUMA `migratepages`（NUMA = 多路 CPU 的内存就近访问架构；migratepages = 把进程的页面迁移到 NPU 所在 NUMA 节点，`cpu_binding.py` 的 `bind_memory` 里提前 return，**CPU 绑核仍生效**）。`allocate_mooncake_host_region` 在 `host_register=True` 建段成功后会自动写 `os.environ` 置位，**但这个时序晚于首次 `bind_cpus`**（见 §4.2 风险表）⇒ 要真正生效需在**启动前 export** |
| `PYTHONHASHSEED=0` | #15642 文档 | 部署前置要求（固定 Python 哈希种子，让各进程行为一致）|
| `MEMFABRIC_HYBRID_EXTEND_LIB_PATH` | #15642 文档 | MemFabric 扩展库的路径（告诉进程去哪加载 MemFabric）|

### 8.4 决定行为的内部常量（非配置）

| 常量 | 值 | 位置/含义 |
| :--- | :--- | :--- |
| `_CPU_CACHE_ALIGNMENT` | 2 MiB | host KV 分配时的对齐粒度；这样相邻层 K/V 的地址差（delta）是恒定的，`skip_topk`（跳到下一层时复用上层计划）只需加一个固定偏移 |
| `_CPU_CACHE_MAX_ALIGNMENT_OVERHEAD_PER_LAYER` | 3 × 2 MiB | 每层最坏情况下的对齐浪费（起始地址向上取整 1×，K 与 V 各自的尾部填充各 1×）|
| `_VLLM_NULL_BLOCK_COUNT` | 1 | vLLM 的空块（占位用的第 0 块），要计入 workload（负载）限制 |
| `lru_workspace_threads` | 8（硬编码） | LRU planner 用 OpenMP 并行处理时开几个线程，启动时会预热（warmup，先跑一遍建好线程池）|
| `FSA_SELECTION_MEMBERSHIP_*` | 16376 / 16384 / 8 / 16400 | membership map 布局（见 §5）|
| `DEST_BLOCK_WAIT_TIMEOUT` / `_INTERVAL` | 2.0 s / 0.001 s | `sfa_pd_rd2h/read_thread.py` 等待 D 侧目的块出现的"有界轮询"参数（最多等 2 秒，每 1 毫秒重查一次）|

---

## 9. 与 0916 问题清单的关联分析

> **本节已抽出为独立文档**：`260916_DSA精度问题定位分析.md`
>
> 抽出原因：该部分在补充了"优先看 PR2、有 fused+memfabric 参照、PR3 可能是 Mooncake 存储问题"等信息后，
> 需要从"逐条猜测"重构为"两点差分 + 可判定实验"的定位文档，篇幅与分析方式都与本文（方案解析）不同。
>
> 新文档包含：
> - 把排查空间收敛到 MemFabric ↔ Mooncake 的**4 个后端差异点**（算子本身对后端无感知，这是关键前提）；
> - 两条既有分析未覆盖的具体缺陷：**跨层 plan 复用在 MemFabric 上恒为空转、Mooncake 上才真正生效**，以及 **`num_copies==0` 时静默覆写 host 池 row 0**；
> - **问题 2（长输入 100k/1M 乱码）的独立分析线**（块映射、容量、静默丢 slot）；
> - PR3 / Mooncake 存储侧的 **connector 侧 C1–C5** 与 **存储侧 M1–M4** 分派清单；
> - 假设 → 三种运行条件下的预测 → 判别实验的对照表，以及可直接当任务分派的 9 步排查顺序。
>
> 本文继续作为方案全貌与契约参考；图 5（请求时间线）、图 6（两跳发布）、图 7（图模式写回）是该定位文档的数据流基础。

---

## 10. 关键代码位置索引

| 关注点（想理解什么） | 文件 : 行 |
| :--- | :--- |
| 融合算子 Python 调用 | `vllm_ascend/attention/sfa_kv_offload.py:865-999`（`_execute_fused_overlap_offload_decode`）|
| membership 存储判定（PR2 改动） | `vllm_ascend/attention/sfa_kv_offload.py:529-535` |
| `HostKVAllocator` 协议 | `sparse_kv_offload/sparse_kv_offload_manager.py:55-63` |
| `MemFabricHostKVAllocator` | 同文件 `:219-245` |
| 池大小规划 / DRAM 上限校验 | 同文件 `:281-360`、`:587-609` |
| `prepare_host_kv_allocation`（后端分派） | 同文件 `:611-659` |
| membership map 分配（两跳 staging） | 同文件 `:790-882` |
| 控制字段初始化 | 同文件 `:884-917` |
| `is_fused_membership_storage` | 同文件 `:919-928` |
| `register_kv_caches`（后端分支） | 同文件 `:952-1364` |
| current KV 写回入口 | 同文件 `:1366-1424` |
| 写回实现（MemFabric / Mooncake 分流） | 同文件 `:1426-1648` |
| 描述符准备（Python 侧） | 同文件 `:1532-1563` |
| `prepare_fused_overlap_external_plan` | 同文件 `:1773-1979` |
| plan 输入校验 | 同文件 `:1981-2035` |
| 当前 token 注入 top-k 缓冲 | 同文件 `:2037-2059` |
| 写回同步 | 同文件 `:2061-2067` |
| C++ 描述符构建 + graph 生命周期 | `csrc/torch_binding.cpp:2539-2712`（PR2 新增）|
| C++ 算子注册 | `csrc/torch_binding.cpp:3137-3148` |
| `MooncakeHostPool` | `sparse_kv_offload/mooncake_host_pool.py:150-266` |
| `allocate_mooncake_host_region` | 同文件 `:75-147` |
| `describe_local_views`（PR3） | 同文件 `:218-256` |
| `GlobalTE.register_buffer/unregister_buffer` | `kv_transfer/utils/mooncake_transfer_engine.py:34-90` |
| DSA 元数据 | `kv_transfer/kv_p2p/mooncake_dsa_metadata.py`（全文 248 行）|
| DSA 传输规划 | `kv_transfer/kv_p2p/mooncake_dsa_transfer.py`（全文 294 行）|
| DSA 接收执行（D2D→D2RH） | `kv_transfer/kv_p2p/mooncake_connector.py:790-932` |
| DSA 调度器 | 同文件 `:2013-2288` |
| 连接器构造与强制校验 | 同文件 `:2290-2342` |
| D 侧注册（Host+HBM 混合） | 同文件 `:3357-3411` |
| P 侧注册 | 同文件 `:3413-3452` |
| D 侧本地布局 | 同文件 `:3454-3492` |
| `register_kv_caches` 分支 | 同文件 `:3494-3593` |
| `shutdown`（排空 + unregister） | 同文件 `:4811-4825` |
| endpoint 规划 | 同文件 `:4827-4884` |
| 命令下发与完成 | 同文件 `:4886-4975` |
| P 侧 DSA 拓扑元数据 | 同文件 `:2782-2813` |
| 基线 PD 连接器（MemFabric） | `kv_transfer/kv_p2p/sfa_pd_rd2h/` |
| 用户文档 | `docs/source/user_guide/feature_guide/layerwise_and_sparse_kv_cache_offloading.md` |
| 设计文档 | `docs/source/developer_guide/Design_Documents/sfa_remote_d2h_connector.md` |

---

## 11. 名词与后端对照速查

| 概念 | MemFabric 路径（#15642 基线） | Mooncake 路径（PR1/PR2/PR3） |
| :--- | :--- | :--- |
| host KV 内存谁来分配 | 用 `offload.empty(..., pin_memory=True)` 取锁页内存，且**只有 tp0（0 号卡）**分配 | Mooncake 共享段：owner_rank=0 创建，**全体 rank 各自映射**一份本地视图 |
| 非 0 rank 怎么访问 | tp0 广播指针 + 用 GVA 还原成"非拥有视图" | 各 rank 直接用自己映射出的本地 NPU 可寻址视图 |
| 各 rank 布局一致性怎么保证 | 靠"虚拟地址本来就相同" | `describe_local_views()` 交换并校验**段内偏移量** |
| Plan（计划表）怎么生产 | CPU planner 直接写共享内存 | CPU planner 写锁页的紧凑缓冲 → NPU staging 中转 → 写进段（两跳，图 6）|
| 当前 token 的 KV 怎么写回 | `offload.sparse_copy` + 指针数组（无需索引张量） | `index_select`（按索引取行）+ `index_copy_`（按索引写行）+ C++ host callback 刷新描述符（图 7）|
| P→D 怎么搬 | `SfaRemoteD2HConnector`（RD2H = remote device to host，走 MemFabric）| `MooncakeConnectorV1(dsa_pd_offload=true)` 按块搬（Main 走 D2RH + Indexer 走 D2D）|
| 内存注册（登记给传输引擎） | 无（MemFabric 自己管）| `register_memory(ptr, size, location)`：主机内存用 `npu:<dev>`，显存用 `*` |
| 注册大小上限怎么处理 | — | 按 64 GiB 分块 + 传输也在两端边界处切分 |
| 关闭时 | 没有显式注销 | `unregister_buffer()` 逆序注销（先注册的后注销）|
| 支持范围 | 仅 TP（张量并行），不支持 CP/PP；仅 MRV1（Model Runner V1）| 支持 TP/PP/CP、P_TP ≠ D_TP；仅 MRV1 |

---

## 12. 参考资料

- PR1 Decode Mooncake host DRAM allocation：<https://github.com/vllm-project/vllm-ascend/pull/15883>
- PR2 non-memfabric fused-overlap membership staging and d2h graph writeback：<https://github.com/vllm-project/vllm-ascend/pull/16219>
- PR3 blockwise Mooncake DSA transfers：<https://github.com/vllm-project/vllm-ascend/pull/16378>
- 前置 PR fused sparse attention overlap：<https://github.com/vllm-project/vllm-ascend/pull/15642>
- 融合算子原始 PR：<https://github.com/vllm-project/vllm-ascend/pull/14278>
- 内存上限约束 PR：<https://github.com/vllm-project/vllm-ascend/pull/14921>
- Sparse KV Cache Offloading RFC：<https://github.com/vllm-project/vllm/issues/48203>
- Mooncake Shared Segment 实现：`repo/Mooncake/mooncake-integration/shared_segment.py`
- Mooncake Transfer Engine（NPU 部分）：参见 `flona-workspace/output/260916_TransferEngine-NPU部分解读.md`

---

## 附录 A：本地复现分析环境的命令备忘

```bash
cd /home/zhengqinwen/workspace/repo/vllm-ascend

# 1) 拉到四个 PR（已完成）
git fetch origin refs/pull/15883/head:pr15883 \
                refs/pull/16219/head:pr16219 \
                refs/pull/16378/head:pr16378 \
                refs/pull/15642/head:pr15642

# 2) #15642 是 release 分支 PR，diff 基准是它的父提交 ae638fecd（已核实只有 4 个 commit）
git rev-list --count ae638fecd..pr15642          # → 4
git diff ae638fecd..pr15642 --stat               # → 36 files, +15612/-252（与 GitHub API 一致）
git diff 1dd633e1c^..pr15642                     # 等价写法

# 3) PR1/PR2/PR3 的"真实 base"（栈式依赖，不要对 main 做 diff！）
#    PR1 base = 9e5fc3dcf4   (main)
#    PR2 base = 63a9e32906   (含 PR1)
#    PR3 base = 36f0872be    (含 PR1+PR2)
git log --oneline 9e5fc3dcf4..pr15883     # PR1 的 5 个 commit
git log --oneline 63a9e3290..pr16219      # PR2 的 2 个新 commit（其余 5 个来自 PR1）
git log --oneline 36f0872be..pr16378      # PR3 的 7 个新 commit

# 4) 各 PR 的真正增量（剥离上游依赖）
git diff 9e5fc3dcf4 pr15883               # PR1 自身
git diff 63a9e3290 pr16219                # PR2 自身（剥离 PR1）
git diff 36f0872be pr16378                # PR3 自身（剥离 PR1+PR2）

# 5) 结构化速查：只看每个 PR 新增的类/函数
git diff 63a9e3290 pr16219 | grep -E '^\+.*(def |class )'
git diff 36f0872be pr16378 | grep -E '^\+.*(def |class )'

# 6) 只读分析：把某个 revision 的子树导出到工作区（避免 checkout 打断工作树）
D=/home/zhengqinwen/workspace/flona-workspace/output/.dsa-src
git archive pr16219 vllm_ascend/distributed/kv_transfer/sparse_kv_offload csrc/torch_binding.cpp | tar -x -C $D/pr16219
```

> 说明：本次分析全程**只读**（`git show / archive / diff`），未修改 `repo/vllm-ascend` 工作树，未 checkout（切换）任何分支。四个本地分支 `pr15642 / pr15883 / pr16219 / pr16378` 已保留，可直接复现。

## 附录 B：一句话回答"三个 PR 分别在 1P1D 的哪里"

![速查：三个 PR 分别在 1P1D 的哪里](plantuml/img/11-appendix-quickref.png)

> **图 12** · 速查：三个 PR 分别在 1P1D 的哪里。源文件：`plantuml/11-appendix-quickref.puml`

---

## 附录 C：本次分析中已核实 vs 待验证

| 项 | 状态 |
| :--- | :--- |
| 四个 PR 的 commit 数、文件数、增删行 | ✅ 已核实（`git` + GitHub API 双向一致）|
| `fused_sparse_attention_overlap` 不在 `ascend950` 构建列表 | ✅ 已核实（`csrc/build_aclnn.sh` 第 138/192 行有、196–236 行无）|
| PR1 的 `MooncakeHostPool` 调用的是 `mooncake.shared_segment.*` | ✅ 已核实（源码）|
| PR1 head 已删除 pool 的 Transfer Engine register/unregister | ✅ 已核实（提交 `a59d9581a` 信息）|
| PR2 的两跳发布与 C++ host callback 机制 | ✅ 已核实（源码）|
| PR3 的混合 location 注册、64 GiB 边界、writer 分区 | ✅ 已核实（源码）|
| Mooncake 构建是否带 `USE_ASCEND_DIRECT`、CI 镜像版本 | ❓ 无法从代码判定（不满足时启动即抛 `RuntimeError`）|
| `offload.sparse_copy` 是否接受 HostRegister 的 device VA | ❓ 需真机 |
| DP>1 同机时段名 `sparse_kv_offload_host_pool_dp{dp}` 是否冲突 | ❓ 需真机 |
| PR2/PR3 在真机上的端到端精度（含图模式） | ❓ 需真机（PR1 自述 "CI: pending pipeline verification"）|
| 第 9 节五条问题归因假设 | ❓ 待验证推断 |

---

## 附录 D：图与源文件（PlantUML）

- **图源**：`plantuml/*.puml`（每图一文件；公共字体与配色在 `plantuml/00-common.puml`）
- **渲染产物**：`plantuml/img/*.png`（已渲染好，Markdown 直接引用）
- **重新渲染**：

  ```bash
  cd flona-workspace/output
  ./plantuml/render.sh                            # 渲染全部
  ./plantuml/render.sh 01-deployment-1p1d.puml    # 只渲染某一张
  ```

- **环境说明（已内置，无需手工配置）**：本机 `$HOME` 只读、且系统无 CJK 字体，因此
  `render.sh` 把字体（`.tools/fonts/NotoSansCJKsc-Regular.otf`）与 fontconfig 缓存
  都指向工作区内的 `output/.tools/`，统一用 `Noto Sans CJK SC` 渲染中英文与代码符号。
  换机器渲染时，只需保证存在任一覆盖 CJK 的字体，并把 `00-common.puml` 里的
  `defaultFontName` 改成该字体名即可。
- **`.tools/` 未纳入版本管理**（见 `output/.gitignore`，约 53 MB）。若在新环境重新克隆后
  `render.sh` 报找不到 jar/字体，按下面两步重建即可（脚本会自动使用它们）：

  ```bash
  cd flona-workspace/output
  mkdir -p .tools/fonts .tools/xdg/fonts .tools/home .tools/cache
  curl -sL -o .tools/plantuml.jar \
    https://github.com/plantuml/plantuml/releases/download/v1.2024.8/plantuml-1.2024.8.jar
  curl -sL -o .tools/xdg/fonts/NotoSansCJKsc-Regular.otf \
    https://github.com/notofonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf
  # .tools/home/fonts.conf 内容见 output/.tools/home/fonts.conf（把 <dir> 指向上面的字体目录）
  ./plantuml/render.sh
  ```

**配色约定（全文档统一）**

| 颜色 | 含义 |
| :--- | :--- |
| 浅蓝 `#DAE8FC` / `#B9D7EA` | 1P（Prefill）侧 / 基线 &#35;15642 既有 |
| 浅绿 `#D5E8D4` / `#A9D18E` / `#82B366` | 1D（Decode）侧 HBM / NPU，基线或 ★PR3 目标 |
| 深橙 `#FFD9B3` | ★ 新 PR 新增或改动的组件 |
| 浅橙 `#FFE6CC` | ★PR1 / ★PR2 / ★PR3 引入的逻辑与约束 |
| 红边 `#C00000` | 关键路径（D2RH / D2D / 两跳发布 / index_copy 写回） |
| 黄 `#FFF2CC` / `#FFF9E6` | 关键共享数据结构（membership map / staging） |
| 灰 `#F2F2F2` / `#E8E8E8` | 基线 &#35;15642 既有 / 契约层（三个 PR 未改动） |

**图 ↔ 章节对照**

| 图 | 章节 | 内容 | 源文件 |
| :--- | :--- | :--- | :--- |
| 图 0 | §0.2 | 三个 PR 分工总图 | `10-pr-scope-map.puml` |
| 图 1 | §2.1 | 1P1D 物理部署总览 | `01-deployment-1p1d.puml` |
| 图 2 | §2.2 | Decode 单节点内部结构 | `02-decode-node-internal.puml` |
| 图 3 | §3.1 | 契约层 ↔ 两个后端依赖面 | `03-dependency-layers.puml` |
| 图 4 | §3.2 | 启动阶段（host 池建立） | `08-startup-activity.puml` |
| 图 5 | §3.3 | 一次请求完整时间线 T0–T7 | `04-request-timeline.puml` |
| 图 6 | §4.3① | membership plan 两跳发布 | `06-two-hop-publish.puml` |
| 图 7 | §4.3② | 图模式 current KV 写回 4 阶段 | `07-graph-writeback.puml` |
| 图 8 | §4.4① | D 的两种 cache 两个内存域 | `12-dsa-memory-domain.puml` |
| 图 9 | §4.4③ | Endpoint 规划 | `13-endpoint-plan.puml` |
| 图 10 | §5 | membership map 行布局 | `05-membership-map.puml` |
| 图 11 | §7 | PR3 三条硬约束 → 对策 | `09-pr3-constraints.puml` |
| 图 12 | 附录 B | PR ↔ 1P1D 速查 | `11-appendix-quickref.puml` |

