# Qwen3 训推一致性

训练绘图脚本导出的所有 PNG/SVG（含总览图、原始曲线图）统一使用宽:高 **6:4**。
单项图为 6×4 英寸，普通总览图为 12×8、formal 总览图为 18×12 英寸。
已有运行的图片需要用更新后的绘图脚本重新生成；数据和坐标数值保持原样。

## Qwen3-30B-A3B：原生 vLLM 对照训练（不启用训推一致）

独立的[原生配置](../gsm8k/configs/qwen3_30b_a3b_gsm8k_native.yaml)不包含 `consistency`，
不使用 PIK、Level 3、V4 Attention 或严格 log-prob 相等检查。
训练使用普通 HyperParallel MoE 模块，推理由 vLLM 原生 Qwen3-MoE 实现执行。
vLLM 的 HTTP/IPC 权重发布插件仍用于接收训练权重，不替换原生推理计算。
本入口复用空闲节点筛选，当前节点必须空闲，再从候选中选出三个空闲节点。

两步测试（在候选列表第一台宿主机、仓库根目录执行）：

```bash
python3 hyper_parallel/rl/examples/consistency/scripts/run_qwen3_moe_cluster.py --hosts 80.5.21.101,80.5.21.102,80.5.21.103,80.5.21.104,80.5.21.106,80.5.21.107,80.5.21.108 --master-addr 172.27.3.101 --interface data0.172 --select-nodes 4 --native-baseline --steps 2 --micro-batch-size 2 --max-prompt 1024 --max-new-tokens 1024
```

末尾加 `--dry-run` 可只显示计划；正式训练把 `--steps 2` 改为 `--steps 50`。
原始数据、模型、镜像和共享目录参数沿用集群入口默认值，也可通过 CLI 覆盖。
数据按 4B 协议从完整 7473/1319 行分别以 `Random(42)` 无放回抽取 6400/500 行，保持顺序。
每步 128 题、每题 8 回答、micro-batch 2、每 DP rank response mini-batch 128；
2 步使用前 256 题，50 步恰好覆盖全部 6400 题。

学习率 `1e-6`、KL `0.001`、weight decay `0.01`、FP32 主参数及其余算法/优化器超参数对齐 4B。
`old_log_prob_source=actor`，`collect_log_probs=false`；仍计算训练锚点和 Reference 概率，
但不收集独立采样概率，也不要求训练/推理概率一致。
训练 TP4/CP2/DP8/EP4，关闭 SP；推理每节点 native DP8/TP2，启用原生专家并行（EP16），关闭 EPLB。
推理 token budget 8192、序列上限 64，`gpu_memory_utilization=0.6`，移除固定 KV cache 字节数，
由 vLLM 自动估算缓存容量。prompt/回答上限均为 1024。

按当前原生共置支持边界启用 CPU offload，Reference 也继承此策略。
该内存策略、专家并行范围与 unified 对照不同，不能将时间差全部归因于是否启用一致性。
无需修补版 V4 wheel；不使用 unified 的专家通信开关，原生 vLLM-Ascend 内部算子由镜像实现选择。
当前仅完成 CPU 配置/启动流程验证，四节点 native IPC、通信及内存仍需 A3 实机验收。

原生完成报告为 `node0/run/native-training.json`，不生成严格一致性通过结论。
时间仍记录在 `node0/run/steps.jsonl`、`steps.csv` 和根目录 `summary.json`，可用原耗时统计方式。
训练前后评估固定 500 题；50 步时每 10 步评估。不保存训练 checkpoint、优化器状态或最终 HF 权重。
保留日志、指标和评估结果，并自动生成 reward、loss、耗时、总览和原始曲线的 PNG/SVG。
原生模式不生成一致性图，也不会把未检查的概率差异标成零。
两种模式都必须通过图片存在及非空检查才能报告整体成功，图片位于 `node0/run/plots/`，宽高比均为 6:4。

## Qwen3-30B-A3B：6400/500 训推一致，默认 2 步，支持 5/50 步

在 rank-0 宿主机、仓库根目录执行。`--hosts` 第一项必须是当前主机；
其余候选主机中选择空闲节点，总计四节点、每节点 16 个 rank：

```bash
bash hyper_parallel/rl/examples/gsm8k/scripts/run_qwen3_30b_5step.sh \
  --hosts <rank0-ip>,<peer1-ip>,<peer2-ip>,<peer3-ip> \
  --master-addr <rank0-通信网卡-ip> --interface <通信网卡名> \
  --model-root /absolute/shared/models/Qwen3-30B-A3B \
  --data-root /absolute/shared/data/full-gsm8k \
  --shared-root /absolute/shared/results --wheel /absolute/path/to/patched-v4.whl
```

末尾添加 `--dry-run` 只显示计划，不连接主机、不启动训练。
[独立配方](../gsm8k/configs/qwen3_30b_a3b_gsm8k_unified_5step.yaml)复用 MoE 的
TP4/CP2/DP8/EP4、每节点 rollout TP2/DP8/EP8，完整模型、不做 CPU offload。
每步全局 128 题，每题 8 个回答，micro-batch **2**，每 DP rank 的 response mini-batch 128。
prompt/回答上限均为 1024，`max_model_len=2048`，`max_num_batched_tokens=8192`；
`max_num_seqs=64`，`gpu_memory_utilization=0.6`，移除固定 KV cache 字节数。
两种模式均保留前缀缓存、chunked prefill，显式启用 mode 3、FULL_AND_PIECEWISE，
捕图尺寸为 `[1, 2, 4, 8, 16, 32, 64]`。

输入必须是完整官方 split 的 RL 格式 Parquet（7473/1319 行，含 `prompt`、`extra_info`）。
[抽样脚本](../gsm8k/scripts/prepare_baseline_data.py)对 train/test 分别重置 `Random(42)`，
无放回抽取 6400/500 行，保留抽取顺序，训练不 shuffle。
独立结果目录的 `baseline/data/manifest.json` 保存源文件和输出哈希、抽取索引。
默认 2 步使用抽样训练集的前 256 题；训练前和第 2 步在同一 500 题上贪心评估。
添加 `--steps 5` 可沿用 5 步测试，使用前 640 题。
两个新增模式均设置 `save_steps=0`、`save_final=false`，所有测试和 50 步运行都不保存训练权重或优化器状态。
仅保留配置、日志、指标、数据清单及评估结果；不删除以前运行已保存的模型，也不覆盖原始模型。
完整训练在同一命令末尾添加 `--steps 50`；旧脚本名和 `--baseline-5step` 参数名保留兼容。
50 步每步 128 题，恰好覆盖全部 6400 题；训练前及第 10/20/30/40/50 步评估同一 500 题。
50 步完成后不保存 checkpoint、不导出 HF 权重；本模式仍自动绘图。
覆盖检查使用实际 request-owner 的样本索引，每步核验 128 道预期题目；
`node0/run/coverage.json` 保存每步索引，结束时检查无重复、无遗漏，50 步覆盖恰好 6400 题。
50 步默认超时 86400 秒，不再为两个新增模式预检 checkpoint 磁盘容量，仍需为日志和抽样数据预留空间。
默认 2 步运行，逐步严格概率检查和自动绘图保持不变。原 `--formal-gsm8k` 的保存和导出策略保持不变。

配置生成器保留 `consistency.mode=batch_invariant`、Level 3、V4 和 MoE 模块替换。
`collect_log_probs=true` 保留独立 sampling 概率，`old_log_prob_source=actor` 重算训练锚点；
现有一致性入口每步更新前检查有效 token 和 log-prob 最大/平均差，缺失概率或差异非零均失败。
该检查覆盖采样 token，不等同于全词表 logits 检查，也不保证准确率上升。
KL 系数为 `0.001`，使用 `low_var_kl`，与完整 MoE GSM8K 训练配方一致；
训练器会构建冻结的 Reference 模型并计算参考概率，增加内存与计算开销。
算法与优化器超参数对齐原 PR 的 4B unified 配方：学习率 `1e-6`、constant、无 warmup，
weight decay `0.01`、梯度裁剪 `1.0`、Adam betas `[0.9, 0.999]`、eps `1e-8`，
`fp32_main_params=true`、`no_decay_params=[]`。GRPO clip low/high 为 `0.2`、clip C 为 `3.0`，
advantage epsilon `1e-6`、token-mean 聚合、每步 policy update epoch 为 1。
训练 seed 为 42，rollout request seed 为 null，vLLM seed 为 17；每 micro-batch 同步梯度。
这不保证跨模型或跨硬件复现相同采样结果。
MoE 保留自身的 TP4/CP2、专家实现与内存策略；4B 的三个 `2048` 动态 token budget
当前只支持 Dense TP1，不复制到 MoE。FP32 主参数和 Reference 会增加内存占用，需实机验证。
最终配置保存在 `node*/run/generated-config.yaml`，逐步一致性证据在 `node0/run/consistency.json`。

该入口与下文 `--formal-gsm8k` 的 50 步流程互斥，原 50 步数据划分及运行参数保持不变。

## Qwen3-4B

[稠密 4B 配方](configs/qwen3_4b_rl.yaml) 使用完整 Qwen3-4B 权重，两张 NPU 同置运行两步 GRPO。默认训练 TP2/SP、推理 TP2，均为 BF16、Level 3 和 V4。PIK 的 `num_leaves: auto` 在此配置解析为 2，覆盖 Attention 的 `o_proj` 和 MLP 的 `down_proj`。在已安装 Hyper-RL 和 V4 的运行容器中，从仓库根目录运行：

```bash
torchrun --standalone --local-addr=127.0.0.1 --nproc-per-node=2 \
  hyper_parallel/rl/examples/consistency/scripts/run_consistency.py \
  hyper_parallel/rl/examples/consistency/configs/qwen3_4b_rl.yaml \
  --report /results/qwen3_4b_consistency.json
```

权重和数据分别挂载到 `/models/Qwen3-4B` 和 `/data/gsm8k`。

## Qwen3-MoE

四层 Qwen3-MoE 运行两步正常 GRPO，验证不同训推并行配置下的实际采样 token log-prob 完全一致。
模型保留原始 128 专家、top-k8 和权重，只裁剪 decoder 层数。

```text
consistency/
├── configs/qwen3_moe_4l_rl.yaml      # TP/SP/DP/EP 基线
├── configs/qwen3_moe_4l_cp_rl.yaml   # 训练 CP、推理 PCP、跨 PCP EP
└── scripts/
    ├── run_qwen3_moe_docker.sh   # 本机 Docker 入口
    ├── prepare_checkpoint.py    # 裁剪、校验并复用四层权重
    └── run_consistency.py       # 正常 RL 循环 + 更新前一致性检查
```

## 本地复现

以下命令在同一个 Bash 终端中、从仓库根目录执行。实验使用本机 Docker，无需远程服务器。

### 1. 准备镜像、模型和数据

宿主机需要 Linux ARM64、可用的 Docker 和 Ascend 910 系列设备，`npu-smi info` 能正常显示设备。
运行实验需要四张空闲 NPU；构建 wheel 不需要 NPU。宿主无需安装 Torch、CANN Toolkit 等 Python/编译依赖，
工具链由镜像提供；宿主仍需安装兼容的 NPU 驱动。

```bash
image=swr.cn-east-3.myhuaweicloud.com/huawei-hyper-rl/hyper-rl:consistency-v0.30.0-cann9.2.1-arm64
docker pull "$image"
docker image inspect --format '{{.Id}} {{.Architecture}}' "$image"
npu-smi info
```

参考环境的镜像 ID 为
`sha256:f139fa4776cf71fa1ebf00c47fa1f4c313deec48267c2102a1a42d968634fb08`。
依赖版本见[运行环境](../../docs/unified_execution_guide.md#运行环境)。镜像已包含采样器的
`q.record_stream(...)` 修复，不需要再执行 `patch_sampler.py`。

准备以下文件；路径相对于仓库根目录：

| 路径                            | 内容                                                      |
| ------------------------------- | --------------------------------------------------------- |
| `../models/Qwen3-30B-A3B/`    | 完整原始模型，包括配置、tokenizer、safetensors 权重及索引 |
| `../data/gsm8k/train.parquet` | GSM8K 训练数据，与现有 GSM8K 示例使用同一格式             |
| `../data/gsm8k/test.parquet`  | GSM8K 测试数据                                            |

YAML 从 parquet 的 `prompt` 列读取提示、`extra_info` 列读取答案信息。
无需提前裁剪模型：启动器从完整权重中提取前四层及公共参数，保存到结果目录的 `Qwen3-MoE-4L/`，
不修改原始模型。再次运行时，校验裁剪配置与文件哈希后复用。

### 2. 准备 V4 补丁 wheel

基础镜像中的 V4 尚未包含 canonical rowsum 补丁，运行时需要加载 `0.5.0+hpbi1`。
**缺少 wheel 时先构建；构建成功、校验通过后，再运行第三步实验。** 以下两种方式选一种。

**已有原验收 wheel：** 按实际位置设置路径，使用对应的固定校验和：

```bash
export V4_WHEEL="$(realpath ../wheels/flash_attn_npu-0.5.0+hpbi1-cp312-cp312-linux_aarch64.whl)"
export V4_WHEEL_SHA256=5ef04a872b1be2c812bb0c36c17ba4776a4b5f9b9eabb24eab0a56a139a497c9
printf '%s  %s\n' "$V4_WHEEL_SHA256" "$V4_WHEEL" | sha256sum --check
```

**没有 wheel：** 使用仓库提供的脚本，在现有镜像中构建：

```bash
bash hyper_parallel/rl/docker/build_v4_wheel.sh ../wheels/v4-build
```

脚本下载固定提交的 FlashAttention NPU 和 CATLASS，应用仓库中的补丁，只编译 Ascend 910 的 V4。
它不修改基础镜像，不向宿主安装依赖，也不占用 NPU；临时源码与编译缓存随容器删除。
输出目录必须为空；重新构建时换一个空目录，已有成功产物可直接复用，无需再次编译。

构建成功后，输出目录包含 wheel、`SHA256SUMS`、`build.json`、`build.log` 和 `environment.txt`。
执行下面的命令，校验并选用本次产物：

```bash
wheel_dir=$(realpath ../wheels/v4-build)
(cd "$wheel_dir" && sha256sum --check SHA256SUMS)
export V4_WHEEL="$wheel_dir/flash_attn_npu-0.5.0+hpbi1-cp312-cp312-linux_aarch64.whl"
export V4_WHEEL_SHA256=$(cut -d ' ' -f 1 "$wheel_dir/SHA256SUMS")
```

重新编译可能改变 wheel 的文件哈希，因此要使用本次 `SHA256SUMS` 中的值，不要填入原验收 wheel 的哈希。
请保留 `build.json`，其中记录了源码提交、补丁及产物身份。
下载使用宿主的 HTTP/HTTPS 代理设置；已有完整 Git 源码时，也可通过 `SOURCE_DIR` 离线构建，
详见[构建说明](../../docker/README.md#构建-v4-补丁-wheel)。

### 3. 运行两步 RL 实验

保留第二步导出的 `V4_WHEEL` 和 `V4_WHEEL_SHA256`，设置本地输入、设备与结果目录：

```bash
export MODEL_ROOT="$(realpath ../models/Qwen3-30B-A3B)"
export DATA_ROOT="$(realpath ../data/gsm8k)"
export DEVICES=0,1,2,3
export RESULT_ROOT=/tmp/hyper-rl-consistency/train-sp

bash hyper_parallel/rl/examples/consistency/scripts/run_qwen3_moe_docker.sh
```

可通过 `IMAGE` 指定含所需数值算子和 vLLM 接口的本地镜像。
`DEVICES` 必须列出四张不同的空闲卡，可按本机情况修改。`RESULT_ROOT` 首次运行至少预留 10 GiB，
用于裁剪权重和实验记录；需要持久保存时改为自己的数据目录。Docker 存储还需有镜像、构建及容器运行空间。
同卡部署若在创建 rollout 通信组时出现 HCCL `EI0020` 监听端口冲突，可以给本次运行追加
`--rollout.vllm.server_hccl_if_base_port=64000 --rollout.vllm.server_hccl_npu_socket_port_range=64000-64150`。
请按本机可用端口选择，与训练进程的 HCCL 端口段错开。

启动器依次检查输入、wheel 校验和、镜像身份和设备占用，创建独立容器，安装当前 checkout 与 RL 插件，
将补丁 wheel 安装到 `/opt/hp-v4`，供训练进程和 vLLM worker 共同加载，然后裁剪权重并执行两步训练。
源码、完整模型和数据以只读方式挂载；结果保存在宿主 `RESULT_ROOT`，容器退出后保留。
固定镜像缺少当前 HP 要求的 PEFT 依赖，启动器会下载固定版本 `peft==0.21.2` 和
`accelerate==1.12.0`，以 `--no-deps` 安装并执行 `pip check`；不会升级 Torch/NPU/V4 等数值依赖。
首次启动会进行图编译与捕获，需要等待；过程日志写入 `run/train.log`。

启动器在容器内、导入 Torch 和启动 torchrun 之前计算 CPU 线程预算，结果保存为
`run/cpu-threads.json`。它读取进程 affinity（包含 cpuset 限制）、SMT 物理核心关系及
可见 cgroup v1/v2 的最紧上级 CPU 配额。开启训练 CPU offload 时，默认取可用核心的
50%（向下取整），按本机训练进程数均分，每进程最多 8 线程、至少 1 线程。
例如 192 核、8 个训练 rank 时，每 rank 分配 8 线程，总计 64 线程。
关闭 offload 时自动使用 1 线程；显式设置的 OMP/MKL 线程数不受自动分配上限约束。
CPU offload 的判断使用与训练入口相同的 YAML 和命令行覆盖。

可用下列环境变量覆盖线程预算：

| 变量 | 含义 |
| --- | --- |
| `OMP_NUM_THREADS` | 手动指定每个训练进程的线程数；未设置或 `auto` 时自动分配 |
| `MKL_NUM_THREADS` | 显式数值优先于 OMP；启动器将两个变量统一为最终训练线程数 |
| `HYPER_RL_CPU_RESERVE` | 自动分配时预留的核心预算，默认预留一半；非负整数只可进一步增加预留，不能突破 50% 预算 |
| `HYPER_RL_VLLM_CPU_THREADS` | 独立的推理 OMP/MKL 线程数，启动器默认 1，不继承较大的训练线程数 |

例如 `OMP_NUM_THREADS=8 bash hyper_parallel/rl/examples/consistency/scripts/run_qwen3_moe_docker.sh`
可保留手动训练预算（显式设置了 MKL 时以 MKL 为准）。未使用该启动器的入口可在容器内、
torchrun 之前调用 `python -m rl.utils.cpu_threads --config CONFIG --local-world-size N`：
标准输出是最终线程数，标准错误输出包含资源计算依据。将结果设置到 OMP/MKL 后再启动训练。

这是启动时的计算线程预算，不是运行中抢占空闲 CPU 或分配互斥核集合，也不改变 NUMA 绑定。
同一启动器的共享 affinity 只除以本机进程数一次；不要在已经分别绑核的 worker 中再次均分。
共享机器应通过容器限额或手动预算限制资源。自动值不保证是内存带宽约束下的性能最优值。

默认使用普通并行基线配置。两份 YAML 都显式包含完整替换配方，不通过隐藏的配置继承补充训练参数。
运行其他组合时只覆盖下面列出的拓扑字段，保持共享数值实现不变。

## 配置与验证矩阵

所有组合使用四张物理卡。训练的进程数为 `DP × TP × CP`，推理 worker 数为 `DP × TP × PCP`；
不能只改一个并行度而忽略总 worker 数和 EP 整除约束。表中的 DP2 训练指 `dp_shard=2` 的 FSDP 数据并行。

| 用例 | 配置 | 训练 | 推理 | 主要验证点 |
| --- | --- | --- | --- | --- |
| A | 普通基线 | TP4/SP/EP4/CP1 | TP2/DP2/PCP1/EP4，无 SP | 两端 TP 不同，训练开启 SP |
| B | 普通基线＋覆盖 | TP2/DP2/EP4/CP1，无 SP | TP4/DP1/PCP1/EP4/SP | FSDP 数据并行与推理 SP |
| C | CP/PCP 配置 | TP2/CP2/EP2/SP | TP2/DP1/PCP2/EP4，无 SP | 跨 CP 前向与跨 PCP 专家域 |
| D | CP/PCP 配置＋覆盖 | TP2/CP2/EP2/SP | TP2/DP2/PCP1/EP2，无 SP | 只在训练侧开启 CP |
| E | CP/PCP 配置＋覆盖 | TP2/CP2/EP2/SP | TP2/DP1/PCP2/EP2，无 SP | 相同 PCP 下不同专家组大小 |
| F | CP/PCP 配置＋覆盖 | TP2/CP2/EP2/SP | TP2/DP1/PCP2/EP4/SP，budget=512 | 推理 SP 与较大合法 batch |
| G | CP/PCP 配置＋覆盖 | TP2/DP2/CP1/EP2/SP | TP2/DP1/PCP2/EP4，无 SP | 只在推理侧开启 PCP |

两份文件是 [普通基线](configs/qwen3_moe_4l_rl.yaml) 和 [CP/PCP 配置](configs/qwen3_moe_4l_cp_rl.yaml)。
后者的 CP wrapper 声明为 `when: cp`，因此 G 中 CP1 不安装该 wrapper。
EP2 自然在 TP2 组内；EP4 跨两个 PCP 平面。两端 CP 不必相同。

两份配方的 PIK 使用 `num_leaves: auto`，按本次训练和推理的最大 TP 确定共享 G。
TP 必须为 2 的幂（包括 1）。若要在不同最大 TP 的独立实验间比较逐位结果，应显式固定同一个 G；
自动模式不保证不同 G 的实验互相逐位相同。

保留上文模型、数据、wheel 和 DEVICES 设置；每个用例使用独立 RESULT_ROOT，**依次**运行：

```bash
runner=hyper_parallel/rl/examples/consistency/scripts/run_qwen3_moe_docker.sh
base=hyper_parallel/rl/examples/consistency/configs/qwen3_moe_4l_rl.yaml
context=hyper_parallel/rl/examples/consistency/configs/qwen3_moe_4l_cp_rl.yaml

# A：默认的不同 TP/SP 组合。
CONFIG="$base" RESULT_ROOT=/tmp/hyper-rl-consistency/A bash "$runner"

# B：训练 FSDP2，推理开启 SP。
CONFIG="$base" RESULT_ROOT=/tmp/hyper-rl-consistency/B bash "$runner" \
  --train.accelerator.tp=2 --train.accelerator.dp_shard=2 \
  --train.accelerator.sequence_parallel=false \
  --rollout.vllm.tensor_parallel_size=4 --rollout.vllm.data_parallel_size=1 \
  --rollout.vllm.sequence_parallel=true

# C：训练 CP2，推理 PCP2，专家域跨 PCP。
CONFIG="$context" RESULT_ROOT=/tmp/hyper-rl-consistency/C bash "$runner"

# D：训练 CP2，推理 PCP1。
CONFIG="$context" RESULT_ROOT=/tmp/hyper-rl-consistency/D bash "$runner" \
  --rollout.vllm.context_parallel_size=1 --rollout.vllm.data_parallel_size=2 \
  --rollout.vllm.expert_parallel_size=2

# E：推理 PCP2/EP2，每个 TP2 组拥有一份完整专家集合。
CONFIG="$context" RESULT_ROOT=/tmp/hyper-rl-consistency/E bash "$runner" \
  --rollout.vllm.expert_parallel_size=2

# F：推理 SP，提高每轮调度的 token 预算。
CONFIG="$context" RESULT_ROOT=/tmp/hyper-rl-consistency/F bash "$runner" \
  --rollout.vllm.sequence_parallel=true --rollout.vllm.max_num_batched_tokens=512

# G：训练 CP1/FSDP2，推理仍为 PCP2。
CONFIG="$context" RESULT_ROOT=/tmp/hyper-rl-consistency/G bash "$runner" \
  --train.accelerator.cp=1 --train.accelerator.dp_shard=2
```

这些是可分别复现的验证组合，不表示一次启动会穷举全部拓扑。
F 提高调度上界，实际送入 MoE 的行数取决于 prompt、并发数和调度结果。

## EP 与图模式对照

两份配方都使用 `unified_ep_compute_fn`。训练使用动态 AllToAll，推理使用 vLLM-Ascend 原生专家。TP、SP、PCP 与 DP 改变源行和专家的归属。对照 eager 与图模式时，可给上述命令增加 `--rollout.vllm.enforce_eager=true` 并更换结果目录。两种模式执行同一个专家入口。[MoE 实现](../../docs/unified_execution_moe.md#推理端的原生专家计算)

## 检查结果

每步更新前比较实际 rollout log-prob 与 Actor 重算结果，要求使用同一权重版本和 token 前缀。
仅统计有效 action token，排除 padding，要求非空、有限且最大/平均差值均为 **0**。
不使用 Actor 概率回填推理结果，不修改采样 token 或 reward 来通过检查。

两步正常 RL 循环与权重发布完成后才写出 `passed: true`：

```bash
RESULT_ROOT=/tmp/hyper-rl-consistency/C
cat "$RESULT_ROOT/run/consistency.json"
rg 'CONSISTENCY_CHECK|CONSISTENCY_PASS' "$RESULT_ROOT/run/train.log"
```

用例命令中的 `RESULT_ROOT=... bash ...` 是该命令的局部环境变量；检查时请先将 RESULT_ROOT
设为要检查的目录，例如 `RESULT_ROOT=/tmp/hyper-rl-consistency/C`。
报告需要包含 `completed_steps: 2`、第 1/2 步，以及每步正的 `log_probs_valid_tokens`。

| 文件 | 内容 |
| --- | --- |
| `run/consistency.json` | 两步采样概率检查与实际训练/推理配置 |
| `run/train.log` | 图编译、推理、训练、发布及一致性检查日志 |
| `run/config.yaml` | 包含 CLI 覆盖的完整实际配置 |
| `run/environment.txt`、`run/v4-wheel.sha256` | 依赖与数值内核身份 |
| `Qwen3-MoE-4L/crop_manifest.json` | 裁剪权重的来源和校验信息 |

默认短响应可能产生全零任务奖励，因此示例的成功不能单独证明非零梯度或学习效果。
需要非零更新时使用下述测试夹具：它只提供明确的交替奖励，采样、模型计算、optimizer 和发布仍走真实路径。
梯度正确性另由完整参数对照检查，不由两步采样概率检查代替。

## 组件与非零更新检查

在已安装本仓及固定数值依赖的环境中，显式选择空闲设备：

```bash
# 真实 EP 路由、不等长及空源行、图回放与训练梯度。
RL_EP_ST_DEVICES=0,1,2,3 RL_EP_ST_SIZE=4 RL_EP_ST_MODEL=/path/to/Qwen3-MoE-4L \
  pytest -q hyper_parallel/rl/tests/st/test_unified_expert_parallel.py

# CP Attention、完整测试模型的 CP1/CP2 logits 与逐参数梯度。
RL_CP_ST_DEVICES=0,1,2,3 \
  pytest -q hyper_parallel/rl/tests/st/test_context_parallel.py -k 'attention or model'

# 跨 PCP 专家域、权重映射及分页 Attention。
RL_PCP_ST_DEVICES=0,1,2,3 RL_UNIFIED_ST_MODEL=/path/to/Qwen3-MoE-4L \
  pytest -q hyper_parallel/rl/tests/st/test_prefill_context_parallel.py -k 'mesh or attention'

# 两次非零更新及发布；该 YAML 内的模型、数据和结果路径必须对当前环境有效。
RL_PCP_ST_DEVICES=0,1,2,3 \
  RL_PCP_ST_RL_CONFIG=hyper_parallel/rl/examples/consistency/configs/qwen3_moe_4l_cp_rl.yaml \
  pytest -q hyper_parallel/rl/tests/st/test_prefill_context_parallel.py -k parallel_rl
```

执行范围和梯度容差见 [统一执行验收](../../docs/unified_execution_guide.md#支持范围与验收)。
这两份配置聚焦不同并行布局的前向一致性；跨机、大 EP、长上下文性能需要单独验收。

## A3 ???????

A3 ???????????????? V4/runtime wheels ??? ID ???
?????????????????????? token?reward ??????
A3 uses the native Ascend expert path; the former MC2/AllGather selector is removed.
?????????????? A2 fullmesh????? mc2 ???????????

?????????????????????

```bash
export IMAGE=hyper-rl:consistency-v0.30.0-cann9.2.1-a3-runtime-arm64
export MODEL_ROOT=/mnt/share/zjy/models/Qwen3-30B-A3B
export DATA_ROOT=/mnt/share/zjy/datasets/gsm8k/prepared
export RESULT_BASE=/home/z00988596/results/consistency
export DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

# ? TP/EP/SP ???CP1?PCP1?
bash run_a3.sh --layers 0 --steps 2 \
  --train-tp 4 --train-ep 4 --train-sp false \
  --rollout-tp 4 --rollout-ep 4 --rollout-sp false --transport ipc

# CP/PCP??? DP2 x TP4 x CP2??? DP2 x TP4 x PCP2?
bash run_a3.sh --layers 0 --steps 2 \
  --train-tp 4 --train-cp 2 --train-ep 4 --train-sp true \
  --rollout-tp 4 --rollout-pcp 2 --rollout-ep 4 --rollout-sp false --transport ipc
```

run_a3.sh ?? CP1/PCP1?CP ? PCP ?? 1 ????? CP ???
????????????????TP2/CP2/EP2??? TP2/PCP2/EP4?
??????? DEVICES=0,1,2,3 ? --layers 4?
CONFIG ?????????? CP>1 ???????? CP Attention ???

| ?? | ?? |
| --- | --- |
| TRAIN_CP / --train-cp | ?? CP1/2/4 |
| ROLLOUT_PCP / --rollout-pcp | ?? PCP1/2/4?DCP ??? 1 |
| GENERATION_TIMEOUT | ????? HTTP request_timeout??? 1800 ? |
| --transport auto/ipc | ????????colocated ???????????? IPC |

??????????????????????????????????????
??????????? --steps ?????????????????????
?? bash run_a3.sh --dry-run ??????????? Docker?

??????????v1.0-A3119_IPC1 ? ACL ?????
v2.0-A3119_IPC0 ??? IPC ???????????????

```bash
git fetch origin --tags
git switch --detach v1.0-A3119_IPC1
# ????? IPC ??
git switch --detach v2.0-A3119_IPC0
# ???? CP/PCP ???
git switch A3_hyper_rl
```

??????? Git ???????A3 ??????
[A3 ????](A3_OPS_REPAIR.md)?????????? [A3 ????](A3_TEST_GUIDE.md)?

## Real GSM8K learning acceptance

Use this mode to verify an actual optimizer update, weight publication, and the next rollout's exact
sampled-token log probabilities. It keeps the strict GSM8K scorer and the recipe learning rate.
No alternating or synthetic rewards are injected. Ordinary runs remain forward-consistency checks.

Select original prepared training rows and allow complete answers:

```bash
export MAX_PROMPT=512 MAX_NEW_TOKENS=1536 MAX_MODEL_LEN=2048
export PREFILL_TOKENS=128
export MAX_NUM_SEQS=4 KV_CACHE_MEMORY_BYTES=1073741824
export MICRO_BATCH_SIZE=1 RESPONSE_MINI_BATCH_SIZE=8 PROMPTS_PER_RANK=1
unset GLOBAL_PROMPT_BATCH_SIZE
export LEARNING_PROMPT_ROWS=4483,310

bash run_a3.sh --layers 0 --steps 2 --candidates 8 --transport ipc \
  --train-tp 4 --train-ep 4 --train-sp false \
  --rollout-tp 4 --rollout-ep 4 --rollout-sp false \
  --require-learning
```

These zero-based row indices refer to the checked 7473-row prepared GSM8K training set.
The selector preserves original questions and labels, records source/subset hashes, and repeats selected
records only to supply distributed batches. The subset is an acceptance fixture, not an accuracy
benchmark. Inspect the source questions if your prepared dataset uses a different row order.

`--require-learning` enables the existing learning gate for the first step and records bounded owned-weight
probes. A changed probe is positive evidence of an actual parameter update. If all sampled probes remain
unchanged, the check fails without claiming that every unsampled value was inspected.
At least one group must produce nonzero advantages, the reported gradient must be finite and positive,
and an actual weight change must be witnessed. Publication must complete before the following rollout
passes the unchanged strict zero-logp-difference gate. The second step may naturally have uniform rewards;
the required evidence is the next-policy probability check after a nonzero first update.

Inspect `run/learning-step*-rank*.json`, `run/steps.jsonl`, and `run/consistency.json`.
The first learning record must have `publication_completed: true` and `next_step_logp_verified: true`.
Natural sampling is not guaranteed to reproduce mixed rewards on every device or topology. A uniform
first group fails clearly; it is never replaced by fabricated rewards. Strict GSM8K scoring can give zero
for a mathematically correct answer whose final format does not match `#### NUMBER`.

## Full-source 50-step GSM8K run

The short preset uses the full prepared training split as a shuffled source and keeps the original
GSM8K scoring rule. It does not select only the earlier two acceptance questions.

Run from the repository root with the previously validated A3 image/model/data configuration:

```bash
git pull --ff-only origin A3_hyper_rl
# Check the new all-different topology before the longer run:
bash run_a3_gsm8k50.sh --steps 2
# Start a fresh50-step run from the original model:
bash run_a3_gsm8k50.sh
```

| Setting | Training | Rollout |
| --- | --- | --- |
| TP | 4 | 2 |
| DP | 2 | 8 |
| SP | true | false |
| EP | 4 | 8 |
| CP / PCP | CP2 | PCP1 |

The preset requires16 logical devices. All five parallel settings differ; DCP stays1.
This new topology still requires target A3 acceptance. A CPU configuration check is not device validation.

Defaults are50 steps,4 prompts per training DP rank,4 candidates per prompt, micro-batch1,
and response mini-batch16. With policy-update epochs1, this is one optimizer mini-batch per training step.
The run processes400 source prompt rows and generates1600 candidate answers. All7473 source rows remain
available to the shuffled loader;400 rows are used in this run, not the entire dataset.

For a smaller run, use `bash run_a3_gsm8k50.sh --prompts 2` (200 prompt rows and800 answers over50 steps).
`--steps` and `--candidates` can also be overridden. Workload defaults deliberately override old smoke-test
`STEPS`, `PROMPTS_PER_RANK`, `CANDIDATES`, and response mini-batch settings.
Other existing environment settings, including image identity checks, remain in effect.
The preset clears the earlier `LEARNING_PROMPT_ROWS` filter and `GLOBAL_PROMPT_BATCH_SIZE`.

Generation defaults to1024 output tokens,512 prompt tokens and4096 total-context tokens.
The measured prepared dataset has7473 rows and a maximum239 prompt tokens with the configured template.
The preset timeout defaults to28800 seconds and can be changed with `RUN_TIMEOUT`. Runtime is not guaranteed;
measure this topology and batch size before estimating the full run.

Every requested step retains the unchanged strict sampled-token logp gate. `--track-updates` records
parameter-change witnesses, gradient norms, publications and subsequent probability checks.
Uniform-reward groups and genuine zero gradients are recorded normally; nonfinite evidence still fails.
This differs from `--require-learning`, which requires a nonzero first-step update and is not enabled
by the full-source preset. Neither mode injects rewards or changes the recipe learning rate.

After training, CPU-only Matplotlib automatically writes the following files under the run directory:

- `run/plots/reward.png` and `.svg`: raw reward mean/std overlaid with trailing 5-step means.
- `run/plots/loss.png` and `.svg`: raw signed total/policy/KL losses overlaid with trailing 5-step means.
- `run/plots/reward-raw.png` and `.svg`, `loss-raw.png` and `.svg`: separate original curves.
- `run/plots/consistency.png` and `.svg`: exact maximum and mean logp differences.
- `run/plots/training-curves.png` and `.svg`: the three-panel overview with actual topology.
- `run/plots/plot-summary.json`: completion state, invalid/missing evidence and update witnesses.

Missing measurements are not replaced by zeros. Failed or partial runs are marked accordingly.
Plots of completed steps do not certify missing later steps. These are training-batch rewards, not held-out
GSM8K accuracy, and a50-step run is not by itself evidence of convergence.

To regenerate plots in a Python environment with Matplotlib and PyYAML:

```bash
python hyper_parallel/rl/examples/consistency/scripts/plot_training.py /path/to/results/run
# Optional: use a larger window and keep the default figures in a separate directory.
python hyper_parallel/rl/examples/consistency/scripts/plot_training.py /path/to/results/run \
    --smooth-window 10 --output /path/to/results/run/plots-ma10
```

Automatic post-training plotting uses the same 5-step default without launcher changes. Each mean uses
the current and preceding four recorded steps; initial steps use the available prefix. Invalid values
leave gaps throughout their window. Reward std smoothing averages recorded per-step standard deviations;
it is not a pooled standard deviation. Consistency metrics and acceptance always use exact raw values.
Use `--smooth-window 1` for raw-only rendering. No training metrics are modified.

The launcher already passes `/results/run` inside the container, mapped to `${RESULT_ROOT}/run` on the
host. `run_a3.sh` prints `RESULT_ROOT` for each unique run (default parent:
`/home/z00988596/results/consistency`). For a finished run, pass that existing run directory to the command
above; pulling new code does not automatically redraw older results. Locate retained metrics with
`find /home/z00988596/results/consistency -type f -name steps.jsonl` if using the default parent.

The preset retains the example's default `save_final: false`: logs and figures are saved, but model and
optimizer checkpoints are not. Use an explicit recipe enabling `train.checkpoint.save_final` if persisted
training weights are required; account for model and optimizer-state storage.

### Experimental MoE device-resident training

`CPU_OFFLOAD=false` opts into colocated unified
Qwen3-MoE GRPO device-residency validation. Parameters and optimizer moments stay on the training
device; existing HSDP resharding, IPC publication, residency checks and strict probability checks remain
active. This is an experimental path, not a claim of completed A3 hardware acceptance. It does not
enable `offload_during_rollout`, whose dense TP1 restriction remains. HBM must accommodate training
shards and optimizer state alongside rollout allocations; allocation failures terminate the run.

Run both five-step cases using the full-model 16-device preset:

```bash
python3 hyper_parallel/rl/examples/consistency/scripts/benchmark_offload.py
```

Reuse a successful five-step CPU-offload mode directory and run only the resident case:

```bash
python3 hyper_parallel/rl/examples/consistency/scripts/benchmark_offload.py \
    --mode device_resident --baseline /path/to/comparison/cpu_offload
```

The comparison excludes the first step. It requires successful strict consistency in both runs and
matching effective configs except for the residency switches. Generated token counts are reported
separately; differing response lengths or cache effects can affect the ratio. Failed runs print saved
rank errors and never produce a speedup ratio. Existing baseline files are read without modification.

### Multi-node HCCL environment probe

`scripts/run_hccl_cluster.py` in this example is a standalone communication diagnostic, not a
multi-node RL launcher. Run it from the first host outside Docker. Replace the example addresses and
interface with the management IPs and data interface of your allocation:

```bash
python3 hyper_parallel/rl/examples/consistency/scripts/run_hccl_cluster.py \
    --hosts 192.0.2.10,192.0.2.11 --master-addr 198.51.100.10 \
    --interface data0 --nproc-per-node 16
```

The controller requires noninteractive SSH to the other hosts, identical locally loaded images, and
idle NPUs on every host. It sends the worker source through each temporary container's environment;
other hosts need no checkout. Containers use host networking, the existing driver mount, automatic
HCCL host/device port allocation and devices `0..nproc_per_node-1`. No model or dataset is loaded.
Each rank verifies every element of 4-byte, 4-MiB and 64-MiB AllReduce results. The first iteration at
each size is warmup; three subsequent iterations measure synchronized local duration. Reported
durations are not bus bandwidth or RL throughput. The default timeout is 300 seconds per collective
and 900 seconds for controller polling, excluding startup and preflight.

Only containers with names generated for this invocation are removed on completion or interruption.
Logs and `summary.json` remain in the controller's `hccl-results/` directory. This does not modify
image-reference containers. `--dry-run` checks arguments and displays topology without contacting
hosts. A passing probe establishes communication evidence only; RL requires its own short run.

To select four hosts from a longer candidate list, append `--select-nodes 4`. The first host remains
mandatory rank zero. Other candidates are checked in list order; busy, unreachable, image-mismatched
or otherwise ineligible peers are skipped. No containers start unless four eligible hosts are found.
The selected host list is printed as `SELECTED_HOSTS` and saved in the summary. Add `--select-only`
to inspect the selection without starting containers. Selection is a snapshot, not a resource
reservation; occupancy is checked again before startup. Without `--select-nodes`, every listed host
is required, preserving the original behavior.

Concurrent rank output may place multiple JSON records on one line. The collector parses each
marked object independently; malformed or incomplete evidence never becomes a pass. To reparse
retained logs without contacting hosts or launching containers:

```bash
python3 hyper_parallel/rl/examples/consistency/scripts/run_hccl_cluster.py \
    --summarize /path/to/hccl-results/probe-directory
```

This writes `recovered-summary.json` and preserves the original logs and summary. Older runs did not
save container exit states: their worker evidence can pass while the overall `passed` field remains
false because exit status cannot be verified. Check `worker_checks_passed` and
`container_exit_status_verified` separately. New runs retain the container exit states.

### Selected-node full-model RL short run

`scripts/run_qwen3_moe_cluster.py` is the experimental synchronous Qwen3 MoE cluster entry.
Run it on the first management host, outside Docker, after pulling this branch. It selects that
host plus eligible peers in candidate order; the default selection is four hosts with 16 devices
each. The controller checks image identity, device occupancy, the data interface, model/data paths
and shared storage before launching. Selection does not reserve devices.

```bash
python3 hyper_parallel/rl/examples/consistency/scripts/run_qwen3_moe_cluster.py --hosts 192.0.2.10,192.0.2.11,192.0.2.12,192.0.2.13 --master-addr 198.51.100.10 --interface data0 --select-nodes 4 --dry-run
```

Replace addresses/interface with the allocation and remove `--dry-run` to execute. Paths default
to `/mnt/share/zjy/models/Qwen3-30B-A3B`, `/mnt/share/zjy/datasets/gsm8k/prepared`, and
`/mnt/share/zjy/rl-cluster-runs`; override them using `--model-root`, `--data-root`, and
`--shared-root`. The last directory must be writable and shared at the same path on every host.
The existing runtime image must already be loaded locally on each eligible host.

The controller snapshots this checkout's tracked working files plus the batch-invariant and
runtime wheels into a new shared run directory. All containers mount this same snapshot read-only;
other hosts need no checkout. `plan.json`, `source.tar.sha256`, selected hosts and the resolved
image ID preserve the run inputs. The known batch-invariant wheel checksum is checked before launch.

Four-host defaults:

| Setting | Value |
| --- | --- |
| Model | Full pretrained Qwen3-30B-A3B; `MODEL_LAYERS=0` |
| Training | Global TP4 / CP2 / DP8 / EP4 / EDP16 |
| Inference | Per-node TP2 / DP8 / EP8, PCP1 |
| Residency | `cpu_offload=false`; add `--cpu-offload` for the standard offload policy |
| Workload | 3 steps, 1 prompt per training DP rank, 4 responses per prompt |
| Token budgets | 512 prompt tokens, up to 1024 new tokens; override `--max-new-tokens` |
| Acceptance | Strict pre-update sampled-token consistency plus update observation |

Each node owns one loopback vLLM service. Global full-gather/broadcast still publishes one policy;
HTTP mutations, physical-device discovery and IPC handles stay within each node. Any node's control
or version-validation failure rejects all Trainers. Training TP x CP groups must fit inside a node.
The small configuration template's filename does not crop the model: the shared worker explicitly
selects the full checkpoint and skips checkpoint preparation.

The controller prints `RESULT_DIRECTORY`; inspect `node0/run/consistency.json`,
`node0/run/summary.json`, `node*/run/progress-rank*.json`, and `node*.log` there. The top-level
`summary.json` passes only when the consistency report covers all requested steps and every owned
container exits successfully. Cleanup removes only containers created by this invocation.

The A3 entries also integrate the [PR1533 execution optimizations](https://github.com/mindspore-ai/hyper-parallel/commit/c98e6591b0a360cb942dc478ba3f4155d0b84db0).
The selected-node entry retains full-model loading and node-local HTTP/IPC control. Inference uses
the native Ascend expert path; `MC2_CAPACITY`, `EP_CAPACITY` and `INFERENCE_EP_BACKEND` are removed
and must be unset for the single-node entry. PIK `num_leaves: auto` resolves jointly from both roles;
TP4 training and TP2 rollout select four leaves. This changes the plan from an explicit eight-leaf
baseline, so acceptance requires fresh sampled-token consistency within the new run.

Before Torch import, the shared worker resolves CPU threads for the effective `CPU_OFFLOAD` setting
using the local `DEVICE_COUNT`, not the global world. `OMP_NUM_THREADS` / `MKL_NUM_THREADS` accept
`auto` or explicit positive integers; automatic device-resident training uses one thread per rank.
`HYPER_RL_VLLM_CPU_THREADS` independently limits inference threads. Each node saves its detected
budget in `nodeN/run/cpu-threads.json`.

Use `--steps 5` to collect four samples after excluding the first step. Compare the same selected
hosts, model, prompt batch, candidate count, token limits and residency policy. The controller
prints `CLUSTER_TIMING` and saves `mean_seconds`, `warm_step_samples`, and generated-token totals
in the top-level summary. Missing measurements are omitted, never treated as zero. Different
generated lengths can change timings; the upstream update does not directly replace weight publication.

To inspect timings for either an old or new retained run without launching containers:

```bash
python3 hyper_parallel/rl/examples/consistency/scripts/run_qwen3_moe_cluster.py --summarize /path/to/qwen3-cluster-directory
```

#### Target-batch preflight

Before predicting a full-dataset run, measure its intended batch with the same full model and topology.
For four nodes, 136 global questions divide into 17 questions per training DP rank, with four responses
per question. Keep micro-batch 1 and set the local response mini-batch to 68, so each step contains one
optimizer update with 68 forward/backward micro-batches per DP rank:

```bash
python3 hyper_parallel/rl/examples/consistency/scripts/run_qwen3_moe_cluster.py --hosts 192.0.2.10,192.0.2.11,192.0.2.12,192.0.2.13 --master-addr 198.51.100.10 --interface data0 --select-nodes 4 --steps 5 --global-prompt-batch-size 136 --micro-batch-size 1 --response-mini-batch-size 68 --full-data --plot
```

Use allocation-specific addresses and interface. Append `--dry-run` to inspect the batch plan without
contacting hosts. `BATCH_PLAN`, `plan.json`, the generated configuration and top-level summary retain
the effective batch settings. The local mini-batch defaults to all local responses; explicit uneven
DP batches or non-divisible accumulation settings are rejected before preflight. Defaults without
these options preserve the original one-question-per-DP short run.

`--full-data` loads and shuffles the existing training parquet; a five-step pilot processes 680 question
presentations, not an entire epoch. It does not create the planned train/validation split or evaluate
the official test set. Pilot checkpoints remain disabled: the formal split-based run starts afresh
from the original pretrained model rather than reusing pilot updates.

`CLUSTER_TIMING` and `--summarize` also show the response-count-weighted mean generated length,
minimum/maximum response lengths, truncation ratio, and the recorded Trainer rank-zero allocation
peak. Missing counters remain null. This memory value excludes other training ranks and vLLM
processes and must not be interpreted as the whole-device HBM peak. Keep it alongside real-device
occupancy observations and any rank-specific OOM reports.

`--plot` uses the existing six-figure export in `node0/run/plots`: reward, loss, consistency, the
three-panel overview, raw reward and raw loss, each in PNG and SVG. These pilot plots do not replace
the additional validation and coverage figures planned for the formal run.

CPU/mocked regressions validate orchestration, not multi-node numerical parity or performance.

#### Formal 50-step GSM8K epoch

```bash
python3 hyper_parallel/rl/examples/consistency/scripts/run_qwen3_moe_cluster.py --hosts 192.0.2.10,192.0.2.11,192.0.2.12,192.0.2.13 --master-addr 198.51.100.10 --interface data0 --select-nodes 4 --formal-gsm8k
```

Replace example addresses/interface with allocated hosts. The first host remains mandatory; additional
candidates allow three idle peers to be selected. This opt-in mode retains full pretrained MoE,
TP4/CP2/DP8/EP4/EDP16 training, node-local TP2/DP8/EP8 rollout, offload=false and micro-batch 1.
Defaults are 50 steps, global batch 136 and a 24-hour timeout. Append `--dry-run` for a plan without allocation.
Ordinary short-run and single-node entries retain their behavior. Formal runs start from original
pretrained weights and do not reuse pilot updates or expose an automatic resume option.

CPU preparation enforces 7473/1319 source counts, duplicate/overlap checks and full untruncated prompt
lengths. A fixed seed splits the training source into 6726 train / 747 validation; all 1319 official
test rows remain unchanged. Source/split hashes and source indices are saved in `formal/data/manifest.json`.
Formal defaults are 512 prompt tokens, 1024 new tokens and 1536 total context tokens.
Original parquet files are read-only. Prompts exceeding 512 tokens stop before training; `--max-prompt`
explicitly raises that budget, changing the workload and requiring a new throughput estimate.

A finite loader uses every training question once: 49 batches of 136, then 62 real questions, four
responses per question. At step 50 six DP ranks hold 32 responses and two hold 28; the latter append
four zero-action-mask rows each. The final mini-batch is 32 with micro-batch 1 and one optimizer update.
Padding generates no extra responses and preserves real GRPO advantages. Actual question indices are
gathered in `node0/run/coverage.json`; missing or duplicated coverage fails formal acceptance.

Greedy validation covers all 747 held-out questions at steps 0/10/20/30/40/50. The final step-50 model
is tested once on all 1319 official test questions before inference shutdown. Test accuracy does not
select the checkpoint. Each evaluation retains count, accuracy and elapsed time in
`node0/run/validation-step*.json` or `test-step50.json`.

Checkpoints are saved at steps 10/20/30/40/50 under `formal/checkpoints/step_*`; the final HF export is
`formal/checkpoints/step_50/hf`. The launcher prints shared free space and a model-only minimum for
five checkpoints plus export; optimizer/rank state needs additional capacity. Formal acceptance also
requires checkpoint completion markers and final exported weights.

Eight figures are exported under `node0/run/plots`, each in PNG and SVG: reward, loss, consistency,
validation accuracy, step time, stage time, coverage and overview (including final test accuracy).
Reward/loss use trailing-five-step display smoothing; consistency stays unsmoothed. Raw JSON/CSV
remains available, and missing evidence never becomes zero or a pass. `summary.json` and `--summarize`
include formal acceptance, evaluation records and controller elapsed time covering preparation,
startup, evaluation, checkpoint IO and cleanup. Step timing excludes evaluation and checkpoint IO.
Formal four-node device acceptance remains separate from CPU/mocked regression checks.
