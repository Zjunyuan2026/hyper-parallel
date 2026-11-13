# A3 基础算子包修复

适用于 `Ascend910_9382` 在 BF16 `torch.ones` 上报 `561103` / `EZ1013`，
提示缺少 `aclnnInplaceOne_1_OnesLikeAiCore` 配置的情况。
原始固定镜像的安装记录为 `Ascend-cann-910b-ops 9.2.1`，A3 需要对应的算子包。
修复应先通过基础算子测试，再恢复四层模型及后续并行测试。

此流程创建新的派生镜像，保留原镜像和宿主机环境。构建不需要占用 NPU。
安装包约 2.5 GB，构建还需要解包及算子文件空间。安装器使用 `--quiet`，按安装帮助含义接受其 EULA。
`--force` 仅作用于派生镜像内的同版本算子替换，不在宿主机执行安装。

## 1. 准备安装包并更新代码

安装包应放在：

```text
/home/z00988596/installers/Ascend-cann-A3-ops_9.2.1_linux-aarch64.run
```

已有该文件时不重复下载。官方下载地址：

```bash
mkdir -p /home/z00988596/installers
cd /home/z00988596/installers
wget --header="Referer: https://www.hiascend.com/" \
  "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.2.1/Ascend-cann-A3-ops_9.2.1_linux-aarch64.run"
```

随后进入仓库更新。工作区有修改时先保留修改，不强制清除：

```bash
cd /home/z00988596/A3_ARL/hyper-parallel
git status --short --branch
git pull --ff-only origin A3_hyper_rl
```

## 2. 构建派生镜像

基础镜像必须已经在本机。使用 BuildKit 将安装包只读挂载到构建步骤，避免将安装包保留在最终镜像层中：

```bash
cd /home/z00988596/A3_ARL/hyper-parallel

DOCKER_BUILDKIT=1 docker build \
  -f hyper_parallel/rl/docker/Dockerfile.a3-ops \
  -t hyper-rl:consistency-v0.30.0-cann9.2.1-a3-ops-arm64 \
  /home/z00988596/installers
```

Dockerfile 先执行安装包完整性检查，然后以 `/usr/local/Ascend` 为安装根目录。
原镜像的记录为 `Install_Path=/usr/local/Ascend`、`Toolkit_InstallPath=/usr/local/Ascend/cann-9.2.1`，
不能把后者误当成安装根目录而产生额外嵌套。
构建检查 A3 包名、9.2.1 版本、CANN 软链和 `ascend910_93` 的 OnesLike 内核配置，保存安装记录及包 SHA256。
构建成功只证明安装与静态检查成功，不代表 A3 算子、V4 或训推一致性已验收。

## 3. 保存新镜像身份并做单设备验证

同一个 Bash 终端中执行。以下假设逻辑设备 0 空闲；实际使用前检查占用，并按需修改可见设备：

```bash
export IMAGE=hyper-rl:consistency-v0.30.0-cann9.2.1-a3-ops-arm64
docker image inspect "$IMAGE" --format '{{.Id}}' \
  > /home/z00988596/installers/a3-ops-image-id.txt
export EXPECTED_IMAGE_ID="$(cat /home/z00988596/installers/a3-ops-image-id.txt)"

npu-smi info
docker run --rm --privileged \
  -e ASCEND_RT_VISIBLE_DEVICES=0 \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v "$PWD:/workspace:ro" \
  --entrypoint /bin/bash "$IMAGE" -c '
    source /usr/local/Ascend/cann/set_env.sh &&
    python /workspace/hyper_parallel/rl/examples/consistency/scripts/probe_npu_runtime.py
  '
```

应看到 `ONES_OK` 和 `RUNTIME_PROBE_PASS`，后者确认 Level 3 下 BF16 矩阵乘法结果正确。
如果仍报错，保留完整输出，暂不运行 16 进程测试。不要通过关闭 Level 3、伪造 SoC 或复制其他 SoC 的 JSON 绕过错误。

## 4. 恢复四层模型两步基线

单设备检查通过后，沿用同一终端中的 `IMAGE`、`EXPECTED_IMAGE_ID`：

```bash
export PROFILE="$PWD/hyper_parallel/rl/examples/consistency/configs/a3.env"
export DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export MODEL_LAYERS=4
export STEPS=2
export MAX_MODEL_LEN=4096
export EP_CAPACITY=4096

TRAIN_TP=4 TRAIN_EP=4 TRAIN_SP=false \
ROLLOUT_TP=4 ROLLOUT_EP=4 ROLLOUT_SP=false \
RESULT_ROOT="/home/z00988596/results/consistency/$(date +%Y%m%d-%H%M%S)-a3-ops-baseline" \
  bash hyper_parallel/rl/examples/consistency/scripts/run_qwen3_moe_docker.sh
```

模型和已准备的 Parquet 路径仍由 `a3.env` 指定，仓库内 wheel 默认不变。
启动器保留完整 Image ID 校验：默认接受原镜像，派生镜像需要显式提供构建后保存的 `EXPECTED_IMAGE_ID`。
新终端需重新设置 `IMAGE` 并从保存的文件加载 `EXPECTED_IMAGE_ID`。

基础 `ones` / `mm` 通过后，仍需在真实流程中检验 V4 补丁和其他算子的硬件兼容性。
若后续出现 V4 或其他算子错误，按其实际错误继续诊断，不将本页的基础算子通过当作整个环境通过。
两步基线通过后，继续 [A3 分批测试指南](A3_TEST_GUIDE.md) 的单项、组合及完整模型测试，
期间保留新镜像的两个环境变量。

## 参考

- [官方 vLLM Ascend 环境安装说明](https://docs.vllm.ai/projects/ascend/en/main/getting_started/installation.html)
- [派生镜像 Dockerfile](../../docker/Dockerfile.a3-ops)
- [基础算子检查脚本](scripts/probe_npu_runtime.py)
