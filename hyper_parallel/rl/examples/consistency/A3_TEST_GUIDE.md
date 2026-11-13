# A3 单节点训推一致性分批测试指南

按“环境预检 → 四层模型基线 → TP/EP/SP 单项 → 组合变化 → 完整模型 → 增加步数”执行。
每个用例独占并复用全部 16 个逻辑设备，训练和推理按阶段轮流运行；不同用例顺序执行。
任一用例失败时先排查，不放宽零误差门禁。

## 1. 硬件与验证范围

目标为单节点、8 张物理卡，每卡两个 die，共 16 个独立逻辑设备，每个 die 有 64 GB 显存。
进程数和 TP/DP/FSDP 网格按逻辑设备数计算，显存预算按每个逻辑设备 64 GB 计算。
先用 `npu-smi info -m` 确认逻辑编号，下面以 0–15 为例。

本指南默认测试 GSM8K 单轮生成、GRPO 和统一执行路径，不接入 Codex。CP、PP 固定为 1。
SP 指 TP 内的 sequence parallel，与 CP 是不同的配置。
下方组合是目标 A3 的待验收矩阵；其他硬件或四层模型的通过结果不能代替 A3 完整模型验收。
每次比较的是同一策略版本下，实际采样 action token 的 rollout logprob 与 Actor 重算 logprob。
这项验收不等同于全 logits、梯度逐位一致或学习收益验收。

## 2. 拉取代码与依赖

首次拉取：

```bash
mkdir -p /home/z00988596/A3_ARL
cd /home/z00988596/A3_ARL

git clone --single-branch -b A3_hyper_rl \
  https://github.com/Zjunyuan2026/hyper-parallel.git

cd hyper-parallel
git log -1 --oneline
sha256sum --check wheels/SHA256SUMS
```

如果已经克隆，进入现有仓库，确认工作区干净后更新：

```bash
git status --short --branch
git switch A3_hyper_rl
git pull --ff-only origin A3_hyper_rl
sha256sum --check wheels/SHA256SUMS
```

有本地修改时先保留修改再更新，不使用 `reset --hard` 清除它们。
三个 wheel 均应显示 `OK`。它们是普通 Git 文件，无需 Git LFS。
启动器默认读取仓库内的 V4 补丁和 PEFT/Accelerate wheel，在临时容器内安装并运行 `pip check`。
目标机需要 Docker、配套驱动和常用 Bash 工具；宿主无需安装训练用的 Python 包。

网络代理应使用执行 Git 命令那台机器可访问的地址。其他机器的 `127.0.0.1` 不是 A3 的代理地址。

## 3. 检查镜像、设备和模型

```bash
docker image inspect \
  hyper-rl:consistency-v0.30.0-cann9.2.1-arm64 \
  --format '{{.Id}} {{.Architecture}}'

npu-smi info
npu-smi info -m
free -h
df -h /home/z00988596

ls /home/z00988596/models/Qwen3-30B-A3B/config.json
ls /home/z00988596/models/Qwen3-30B-A3B/model.safetensors.index.json
```

镜像应为 `arm64`，完整 Image ID 为：

```text
sha256:f139fa4776cf71fa1ebf00c47fa1f4c313deec48267c2102a1a42d968634fb08
```

模型目录还需要 tokenizer、完整权重分片等文件。
检查 16 个有效 `Chip Logic ID`，不要把 Mcu 行计入设备数。
如果 `Ascend910_9382` 在基础 `torch.ones` 上报 `EZ1013` / `OnesLike` 配置缺失，
先按 [A3 算子包修复](A3_OPS_REPAIR.md) 构建并验证派生镜像，再继续本指南。
启动器会将逻辑编号映射到物理 NPU，并保守拒绝使用忙的物理 NPU。
CPU offload 会消耗主机内存；模型、优化器和权重发布需要的内存不能只用总显存判断。
预留模型裁剪、结果、图编译及 Docker 运行所需磁盘空间。

## 4. 准备 GSM8K Parquet，只做一次

原始文件位置为：

```text
/home/z00988596/datasets/gsm8k/grade_school_math/data/train.jsonl
/home/z00988596/datasets/gsm8k/grade_school_math/data/test.jsonl
```

在仓库根目录执行：

```bash
docker run --rm \
  -v "$PWD:/workspace:ro" \
  -v /home/z00988596/datasets/gsm8k/grade_school_math/data:/raw:ro \
  -v /home/z00988596/datasets/gsm8k:/data \
  --entrypoint python \
  hyper-rl:consistency-v0.30.0-cann9.2.1-arm64 \
  /workspace/hyper_parallel/rl/examples/consistency/scripts/prepare_gsm8k.py \
  --source /raw --output /data/prepared

ls -lh /home/z00988596/datasets/gsm8k/prepared/
```

应生成 `train.parquet`、`test.parquet`、`manifest.json`。脚本只转换普通 train/test，保留原始文件，
答案只用于评分，不进入 prompt。输出目录非空时会拒绝覆盖。
如果数据已经正确生成，跳过转换；需要重新转换时选择新的输出目录并相应设置 `DATA_ROOT`。

## 5. 设置公共参数和测试函数

以下命令在同一个 Bash 终端中、从仓库根目录执行：

```bash
cd /home/z00988596/A3_ARL/hyper-parallel

export PROFILE="$PWD/hyper_parallel/rl/examples/consistency/configs/a3.env"
export RUNNER=internal
export DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export STEPS=2
export MODEL_LAYERS=4
export MAX_MODEL_LEN=4096
export MAX_PROMPT=512
export MAX_NEW_TOKENS=1024
export EP_CAPACITY=4096

run_case() {
    local name=$1
    local train_tp=$2 train_ep=$3 train_sp=$4
    local rollout_tp=$5 rollout_ep=$6 rollout_sp=$7
    local out="/home/z00988596/results/consistency/$(date +%Y%m%d-%H%M%S)-${name}"

    TRAIN_TP="$train_tp" TRAIN_EP="$train_ep" TRAIN_SP="$train_sp" \
    ROLLOUT_TP="$rollout_tp" ROLLOUT_EP="$rollout_ep" ROLLOUT_SP="$rollout_sp" \
    RESULT_ROOT="$out" \
      bash hyper_parallel/rl/examples/consistency/scripts/run_qwen3_moe_docker.sh \
      || {
          echo "测试失败，停止后续测试。日志目录：$out/run"
          return 1
      }

    echo "结果目录：$out/run"
    cat "$out/run/summary.json"
    grep 'STEP_SUMMARY' "$out/run/train.log"
}
```

若实际逻辑编号不是 0–15，修改 `DEVICES`，按数字升序且不重复。
每次运行自动使用新的结果目录，首次裁剪会生成独立权重，不改变原始模型。

参数顺序：`用例名 训练TP 训练EP 训练SP 推理TP 推理EP 推理SP`。
训练 FSDP 自动为 `设备数/TRAIN_TP`，推理 DP 自动为 `设备数/ROLLOUT_TP`；EP 不额外乘设备数。
`MODEL_LAYERS=4` 为四层模型，`MODEL_LAYERS=0` 为完整模型，`STEPS` 为本次从零开始的步数。
不要单独 `source a3.env`；通过启动器的 `PROFILE` 加载，使仓库内 wheel 路径正确解析。

## 6. 第一批：四层模型，两步基线

```bash
run_case smoke-A 4 4 false 4 4 false
```

先检查模型加载、GSM8K 单轮生成与评分、V4、训练、发布和清理流程。第一次有冷启动、编译和图捕获开销。
必须满足第 11 节的通过标准，再运行下一批。

## 7. 第二批：分别改变 TP、EP、SP

```bash
run_case smoke-B-TP 4 4 false 2 4 false &&
run_case smoke-C-EP 4 4 false 4 8 false &&
run_case smoke-D-train-SP 4 4 true 4 4 false &&
run_case smoke-E-rollout-SP 4 4 false 4 4 true
```

`&&` 保证任一失败时停止本批后续用例。不同批次之间也应人工检查结果后再继续。

| 用例 | 训练 TP/FSDP | 训练 EP/SP | 推理 TP/DP | 推理 EP/SP | 检查重点 |
| --- | --- | --- | --- | --- | --- |
| A | 4/4 | 4/关 | 4/4 | 4/关 | 相同布局基线 |
| B | 4/4 | 4/关 | 2/8 | 4/关 | 改变 TP |
| C | 4/4 | 4/关 | 4/4 | 8/关 | 改变 EP |
| D | 4/4 | 4/开 | 4/4 | 4/关 | 训练开启 SP |
| E | 4/4 | 4/关 | 4/4 | 4/开 | 推理开启 SP |
| F | 4/4 | 4/开 | 2/8 | 8/关 | 多项组合变化 |
| G | 2/8 | 8/关 | 4/4 | 4/开 | 反向组合变化 |

## 8. 第三批：组合变化

单项测试全部通过后：

```bash
run_case smoke-F-combined 4 4 true 2 8 false &&
run_case smoke-G-reverse 2 8 false 4 4 true
```

## 9. 第四批：完整 Qwen3-30B-A3B

四层测试完成后，切换完整模型并重新从基线开始：

```bash
export MODEL_LAYERS=0
export STEPS=2

run_case full-A 4 4 false 4 4 false
```

基线通过后，再运行单项：

```bash
run_case full-B-TP 4 4 false 2 4 false &&
run_case full-C-EP 4 4 false 4 8 false &&
run_case full-D-train-SP 4 4 true 4 4 false &&
run_case full-E-rollout-SP 4 4 false 4 4 true
```

单项通过后，再运行组合：

```bash
run_case full-F-combined 4 4 true 2 8 false &&
run_case full-G-reverse 2 8 false 4 4 true
```

完整模型的显存、主机内存和通信负载与四层模型不同，每一组都需要重新验收。

## 10. 第五批：增加步数

完整模型两步通过后，对已经通过且需要重点验证的组合增加步数：

```bash
export STEPS=10
run_case full-F-10steps 4 4 true 2 8 false
```

还可将 `STEPS` 改为其他正整数，每次使用新的结果目录。默认不保存可恢复的训练 checkpoint，
增加 `STEPS` 是启动新实验，不是接着上一目录续训。

## 11. 每个用例的通过标准

必须同时满足：

1. 启动命令退出码为 0。
2. `run/summary.json` 和 `run/consistency.json` 均有 `passed: true`。
3. `completed_steps` 等于本次设置的 `STEPS`，每个更新前都有检查记录。
4. `rollout_probs_diff_valid=1`，有效 action token 数大于 0，比较结果有限。
5. `rollout_log_probs_diff_max` 和 `rollout_log_probs_diff_mean` 每步均为 0。

一致性检查失败会在优化器更新前报错；缺少概率证据不能当作零误差。
四层模型可能得到零奖励、零 loss 和零梯度。这种结果只用于流程和前向一致性验收，不能证明学习收益。

## 12. 查看结果与性能

将以下路径替换成测试函数打印的实际目录：

```bash
export CASE_RESULT=/home/z00988596/results/consistency/实际运行目录名

cat "$CASE_RESULT/run/summary.json"
cat "$CASE_RESULT/run/consistency.json"
grep -E 'STEP_SUMMARY|CONSISTENCY_CHECK|CONSISTENCY_PASS|RUN_FAILED' \
  "$CASE_RESULT/run/train.log"
```

| 目的 | 指标或文件 |
| --- | --- |
| 有效比较证据 | `training/rollout_probs_diff_valid`、`training/log_probs_valid_tokens` |
| 训推误差 | `training/rollout_log_probs_diff_max`、`training/rollout_log_probs_diff_mean` |
| 奖励 | `reward/mean`、`reward/std`、`reward/min`、`reward/max` |
| 梯度和 loss | `train/gradient_norm`、`train/total_loss`、`train/policy_loss`、`train/kl_loss` |
| 每步耗时 | `perf/time_per_step` |
| Agent/采样耗时 | `timing_s/gen`，包括其范围内的服务等待和工具执行 |
| 更新与发布耗时 | `timing_s/update_actor`、`timing_s/weight_sync` |
| 逐步完整数据 | `run/steps.csv`、`run/steps.jsonl` |
| 实际生效配置 | `run/config.yaml` |
| 软件和输入身份 | `run/environment.txt`、源码提交、wheel 校验记录 |
| 实际运行模式和长度 | `run/generated-config.yaml`、日志 `TASK_READY` |

字段缺失时检查原始日志，不补成零值。首步包含冷启动，应与后续步分开分析。
`summary.json` 给出排除首步的平均耗时及样本数；两步实验只有一个后续步，不能据此判断稳态性能。
阶段事件是各 rank 的主机墙钟区间，不是设备 kernel 时间；rank0 的时间不是额外全 rank 最大值测量。
改变训练 TP 会改变自动推导的 FSDP/DP 和全局 batch，不直接据此对不同用例做吞吐排名。
若要做严格的跨布局数值对照，还需固定权重、输入 token、mask、全局工作量并进行轨迹重放。

## 13. 失败时的排查顺序

1. 查看 `launcher.log`、`launcher-status.txt`、`container-status.txt`，确定是否在训练前失败。
2. 查看 `summary.json`、`failure-rank*.json`，保留第一条原始异常。
3. 查看 `progress-rank*.json` 和 `events-rank*.jsonl`，定位模型构建、采样、重算、更新或发布阶段。
4. 查看 `train.log` 和 `run/torchrun/` 中各 rank 的完整日志。
5. 对照 `config.yaml`、`npu-mapping.txt`、`npu-before.txt`、主机内存和磁盘记录。

| 现象 | 优先检查 |
| --- | --- |
| 找不到输入 | 模型目录、Parquet、仓库内 wheel、`a3.env` 的宿主路径 |
| 镜像或 SHA256 不匹配 | 镜像完整 ID 和 `wheels/SHA256SUMS`，保留严格校验 |
| 设备映射/占用错误 | `npu-smi info -m` 的有效逻辑编号及同物理 NPU 的其他 die |
| `pip check` 失败 | wheel 是否完整、是否使用指定镜像，查看实际依赖错误 |
| Context budget exceeded | 实际 prompt、工具历史、生成预算及 `MAX_MODEL_LEN` |
| EP 容量不足 | `EP_CAPACITY` 是否覆盖训练序列和推理 prefill 行数 |
| OOM 或进程被杀 | 区分 NPU 与主机内存不足，检查模型深度、上下文、batch、KV cache、EP 容量 |
| logprob 非零误差 | 同一策略版本、V4 补丁、Level 3 状态、实际 TP/EP/SP 和原始 token 证据 |

提高上下文时同步检查 `EP_CAPACITY` 和显存需求；更大的固定容量也增加计算、通信与缓冲开销。
一次只调整一个影响因素，保留失败目录，用新的结果目录重试。
进程被强杀时可能来不及写 Python summary，应结合退出码和 torchrun 日志；文件缺失不代表通过。

## 相关入口

- [一致性示例与运行环境说明](README.md)
- [A3 路径配置](configs/a3.env)
- [Docker 启动器](scripts/run_qwen3_moe_docker.sh)
- [逐步一致性检查和结果输出](scripts/run_consistency.py)
- [统一执行的支持边界](../../rl/unified_execution/README.md)
