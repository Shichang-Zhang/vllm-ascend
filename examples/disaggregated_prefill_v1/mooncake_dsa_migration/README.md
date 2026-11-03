# Mooncake DSA 迁移验证

本目录对应 PR #20289 / #20478 的当前分支适配。目标是 Prefill TP8/DCP8、Decode TP2、MTP3、LI C8、Mooncake Host offload。代码与 CPU 测试已迁移；CANN 编译、NPU、跨节点、准确率和性能结果需在真实机器上补齐。

参考部署脚本为 `/home/sigh/proj/mayi/sig-ai-inference/scripts`，没有改动或执行这些外部脚本。P 端继续使用包含 AscendStoreConnector 和 MooncakeConnectorV1 的 MultiConnector；D 端必须使用直接 MooncakeConnectorV1、`dsa_pd_offload=true`、`kv_load_failure_policy=fail`、PCP1/DCP1、V1 runner。

## 本地回归

仓库根目录的 CPU 环境可使用固定的 [相关测试清单](local_tests.txt)：

```bash
mapfile -t MIGRATION_TESTS < examples/disaggregated_prefill_v1/mooncake_dsa_migration/local_tests.txt
VLLM_VERSION=0.26.0 .venv/bin/python -m pytest -q "${MIGRATION_TESTS[@]}"
```

本机该清单结果为870 passed、2 failed；两项 scheduler 失败已在未修改的 HEAD 方法上复现，详见根目录 [迁移进度](../../../MIGRATION_PROGRESS_PR_20289_20478.md)。全部 tests 还存在依赖/导入收集错误，因此不声称全套测试通过。CPU kernel 内容测试需要 g++；其同步拷贝模拟不能证明 NPU 执行时序。

## 配置与升级顺序

以下文件是完整的 Decode `--additional-config`，保留参考脚本的其他字段。MTP3 的 Copy-SFA 最小 hot budget 是8192；原4096只用于 fused-overlap 对照。

| 文件 | 模式 | 对照 |
| --- | --- | --- |
| [a_overlap.json](a_overlap.json) | 原 fused-overlap，hot4096 | 原部署基线 A |
| [b_copy_sfa.json](b_copy_sfa.json) | Copy-SFA，hot8192，新开关全关 | 迁移基线 B |
| [c_delayed.json](c_delayed.json) | B + delayed D2H | 与 B 比较 #20289 |
| [d_shared_inline.json](d_shared_inline.json) | B + shared-only inline | 与 B 比较混合布局 |
| [e_combined_inline.json](e_combined_inline.json) | delayed + shared-only inline | 与 C/D 比较组合 |
| [f_shared_side_stream.json](f_shared_side_stream.json) | shared-only side stream | 与 D 比较异步 onboarding |
| [g_combined_side_stream.json](g_combined_side_stream.json) | 全部组合 | 与 E/F 比较组合 |

先升级并重启所有 Prefill scheduler/workers，使请求元数据及 GET_META 都支持 DSA v2，再升级 Decode。新 shared-only Decode 会明确拒绝旧 P 协议；不能仅替换 D 端。新 P 对旧 D 的普通布局保持旧 aliases；多 Main-group 新布局需两端同时使用本迁移版本。

在部署脚本副本中，将原 `ADDITIONAL_CONFIG` heredoc 替换为所选 JSON 的读取，例如：

```bash
ADDITIONAL_CONFIG="$(cat /vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/mooncake_dsa_migration/g_combined_side_stream.json)"
```

保留原 `--additional-config "${ADDITIONAL_CONFIG}"`、DP/TP、Mooncake、MTP3 和 target graph 参数。上面的仓库路径按真实容器修改。每次更换配置先排空请求并停止对应实例，然后重启；布局和写回协议不支持热切换。

## 构建与身份记录

使用与目标镜像匹配的 vLLM、PyTorch、torch-npu、CANN 和 Mooncake。适配起点为 Ascend `4cac524d9d19fb71fee4940c782884ed0773782f`，本地 vLLM 源码为 `d02df748bf9efd99022f1a062597dc3cb3808485`。`VLLM_VERSION=0.26.0` 只解决本地源码版本元数据，不能证明另一份 vLLM 源码 ABI 兼容。

在目标容器的迁移源码目录构建，例如：

```bash
COMPILE_CUSTOM_KERNELS=1 python -m pip install --no-build-isolation -e .
git rev-parse HEAD
git diff --stat
npu-smi info
python -m pip show vllm vllm-ascend torch torch-npu mooncake-transfer-engine
```

保存镜像 digest、实际源码 diff、CANN/driver/firmware、模型 revision、所有 P/D 启动命令和解析后配置。本地 LIM copy ABI 已调整：10个 tensor 参数，真实可变目的 KV tensors，8项容量参数；扩展与 custom-op library 必须一起重建。算子当前仅加入 A3 `ascend910_93` 构建列表。

初始化会加载 lazy extension 并检查 ABI；schema 缺失或非 A3 会明确记录 inline fallback。已存在但 ABI 错误、ACLNN 缺失、zero-active warmup 或执行失败会报错，不会静默切换。若做 side-stream 性能比较，先确认没有 fallback 日志，并在 trace 中确认实际执行了 `LimCopyDescriptors`。

## 算子与图验证

在 NPU 环境运行：

```bash
python -m pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/test_lim_copy_descriptors.py
python -m pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/test_lim_mtp_index_reuse.py
```

第一文件验证真实 kernel 的 eager/graph 内容、动态 plan/source table、batch缩小、两owner/三member的流依赖及 capture size1/3切换。其源缓存是 NPU GM，只验证算子与流，不能代替下面的 Mooncake VMM 端到端验证。第二文件验证 MTP reuse。

Target owner 每轮执行 LIM并保留自身完整上下文 LI；member copy 与这次 LI/owner attention 可重叠。Draft groups 保留 inline/reuse，不消费 target 的旧事件。每组 geometry 独立，graph replay 依赖图内 record/wait；不依赖 Python forward 对象检查。

## 真实 P/D 正确性

先分别验证 A/B，再按表验证独立与组合模式。保持 P DP4/TP8/DCP8、D DP16/TP2、MTP3、LI C8、block128、max-seqs8、target FULL_DECODE_ONLY、draft eager。另对 Decode 显式覆盖 async on/off 与 APC on/off，确认实际 RecomputeScheduler/AsyncRecomputeScheduler 和日志中的最终开关。

使用固定模型和 tokenizer、固定输入、greedy/temperature0、固定 seed。测试前确定输出比较规则、数值容差、并发目标、HBM上限和 SLO，写入结果记录；不能依据实验结果再修改阈值。精确 KV 拷贝要求逐元素一致。模型输出先与同配置重复运行的基线比较；发生首个 token 分歧时保存该请求、接受长度和各层中间结果进行定位。

必须覆盖：

- Prompt 长度127/128/129、8191/8192/8193，长上下文跨多个 ring 周期；短 prompt 生成到 hot 边界之外。长请求覆盖实际目标上下文上限。
- MTP多接受/全拒绝、部分接收、请求重排、batch缩小、target graph切换、max-length ignored draft。保持 input/output token预算相同。
- APC重复请求和部分prefix命中；新请求不能命中尚未完成D2H的block，全部必需group可见后才能命中。
- 接收中取消、接收后decode取消、最后一个请求结束后的空调度轮次。新请求复用物理block后不能出现旧内容或迟到完成污染。
- 测试实例上注入 P→D 传输失败及资源压力，确认 `fail` 返回错误，所有任务/TP ACK 排空后才释放；不能重用仍有DMA的Host或resident行。
- P→D所有组及non-contiguous IDs：每个D TP rank都有完整私有owner KV，Indexer含C8 key/scales；共享Host只写一次。确认实际注册大小、组顺序和每层地址。

新模式明确不支持带未排空decode lease的preemption/resume；遇到保护异常应停止该轮并定位容量/调度原因，不能删除保护来继续测性能。共享 owner 常驻会增加 HBM，必须记录可分配KV blocks和目标并发是否仍成立。

## 性能与容量

下面是长上下文负载模板，地址和模型路径必须替换成真实值；通过实际 PD proxy 发请求：

```bash
vllm bench serve \
  --backend openai --base-url "http://<PD_PROXY_HOST>:<PORT>" \
  --endpoint /v1/completions --model glm-52 --tokenizer /path/to/real/model \
  --dataset-name random --random-input-len 32768 --random-output-len 512 \
  --num-prompts 128 --max-concurrency 8 --request-rate inf \
  --temperature 0 --seed 42 --ignore-eos --num-warmups 8 \
  --percentile-metrics ttft,tpot,itl,e2el --save-result --save-detailed \
  --result-dir ./migration-results --result-filename g_combined_side_stream_run1.json
```

每个配置至少重复3次，并补充短请求、mixed-length和目标并发负载。A→B包含模式及hot budget变化，不能计作两个PR的收益。需要分别报告P→D、D2H retire/CPU TP collective、LIM/第二次LI、onboarding、attention、TTFT/TPOT/吞吐、HBM峰值与可用KV容量。

按当前 vLLM profiler 配置在服务端启用分析，再用 `vllm bench serve --profile` 收集。同一次 trace 内核对 plan-ready→side-copy→member-wait，以及copy与owner compute的时间重叠；不能仅凭使用side stream声称获得重叠。CPU TP collective和resident owner第二次LI都可能抵消收益。没有超过基线波动的收益时只报告正确性适配，保持新功能默认关闭。

结果记录至少包含：配置文件/hash、模型/软件/硬件身份、async/APC、请求集、成功/错误/超时、准确率或首个分歧、MTP acceptance、p50/p99 TTFT/TPOT、吞吐、HBM、KV blocks、压力测试的释放/残留情况、profile路径及3次重复结果。

## 回退

排空并停止实例，使用 B 或 A 文件重启。若 DMA 或事件状态未知，按故障终止路径排空/清理；不要运行中切换开关。硬件验证未完成前，本地测试结果不构成生产启用或性能收益证明。
