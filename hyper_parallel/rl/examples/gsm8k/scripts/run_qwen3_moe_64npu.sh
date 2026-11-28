#!/usr/bin/env bash
# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

# Run inside the prepared Ascend environment. Each invocation owns a new result directory.
set -euo pipefail
mode=${1:-}
if (( $# )); then shift; fi
steps=50
if (( $# )) && [[ $1 != --* ]]; then
    steps=$1
    shift
fi
if [[ ! $mode =~ ^(native|unified)$ || ! $steps =~ ^[1-9][0-9]*$ ]]; then
    echo "Usage: $0 {native|unified} [positive-step-count] [--section.field=value ...]" >&2
    exit 2
fi

scripts=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd -- "$scripts/../../../../.." && pwd)
source "$scripts/configure_qwen3_shared_paths.sh"
config="$scripts/../configs/qwen3_30b_a3b_gsm8k_${mode}_64npu.yaml"
model=$(realpath -- "$GSM8K_MODEL")
data=$(realpath -- "$GSM8K_DATA")
node_rank=${NODE_RANK:?Set NODE_RANK to 0, 1, 2 or 3}
master_addr=${MASTER_ADDR:?Set MASTER_ADDR to the rank-zero data address}
run_id=${GSM8K_RUN_ID:?Use the same unique GSM8K_RUN_ID on all four nodes}
[[ $node_rank =~ ^[0-3]$ && $run_id =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
results_root=$(realpath -m -- "$GSM8K_RESULTS")
result="$results_root/$run_id/node$node_rank/run"
for file in "$model/config.json" "$data/train.parquet" "$data/test.parquet"; do
    [[ -r "$file" ]] || { echo "Missing input: $file" >&2; exit 1; }
done
python - "$data" <<'PYDATA'
import sys
from pathlib import Path
import pandas as pd
import matplotlib
import yaml
root = Path(sys.argv[1])
for split, expected in (("train", 6400), ("test", 500)):
    actual = len(pd.read_parquet(root / f"{split}.parquet"))
    if actual != expected:
        raise SystemExit(f"Expected prepared {expected}-row {split} split, got {actual}")
PYDATA
command -v npu-smi >/dev/null
mkdir -p -- "$(dirname -- "$result")"
mkdir -- "$result"
exec > >(tee "$result/train.log") 2>&1
echo "Run directory: $result"
cd "$repo"

export PYTHONPATH="$repo/hyper_parallel/rl:$repo:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_HOST_IP=127.0.0.1 GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-data0.172} VLLM_SERVER_DEV_MODE=1
export VLLM_ASCEND_ENABLE_NZ=0 VLLM_BATCH_INVARIANT=0
export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-data0.172}
export PYTHONFAULTHANDLER=1
export HCCL_IF_BASE_PORT=${HCCL_IF_BASE_PORT:-64600}
export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-64600-64799}
export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-2}
export HYPER_RL_VLLM_CPU_THREADS=${HYPER_RL_VLLM_CPU_THREADS:-1}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
export HYPER_RL_FAILURE_DIR="$result"
if [[ "$mode" == unified ]]; then export HCCL_DETERMINISTIC=true; fi

# Use wall-clock timestamps to align device samples with evaluation, rollout and optimizer phases.
sample_resources() {
    while true; do
        date --iso-8601=seconds
        awk '/^(MemTotal|MemAvailable|SwapFree):/' /proc/meminfo
        head -n 1 /proc/stat
        for path in /sys/fs/cgroup/memory.{current,peak,max,events}; do
            if [[ -r "$path" ]]; then printf '%s %s\n' "$path" "$(< "$path")"; fi
        done
        npu-smi info || echo "npu-smi sample failed"
        sleep 10
    done
}
finish() {
    local status=$?
    if [[ -n ${monitor_pid:-} ]]; then
        kill "$monitor_pid" 2>/dev/null || true
        wait "$monitor_pid" 2>/dev/null || true
    fi
    printf '%s\n' "$status" > "$result/exit_code"
    date --iso-8601=seconds >> "$result/run-info.txt"
}
trap finish EXIT
{
    date --iso-8601=seconds
    if [[ -n ${GSM8K_SOURCE_REVISION:-} ]]; then printf '%s\n' "$GSM8K_SOURCE_REVISION"; else git rev-parse HEAD; fi
    printf 'mode=%s steps=%s devices=%s model=%s data=%s\n' "$mode" "$steps" "$ASCEND_RT_VISIBLE_DEVICES" "$model" "$data"
    for name in GSM8K_SHARE_ROOT GSM8K_IMAGES GSM8K_CACHE_ROOT XDG_CACHE_HOME HF_HOME HF_HUB_CACHE \
        HF_DATASETS_CACHE PIP_CACHE_DIR TORCH_HOME TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR \
        VLLM_CACHE_ROOT ASCEND_CACHE_PATH WANDB_CACHE_DIR; do
        printf '%s=%s\n' "$name" "${!name}"
    done
    sha256sum "$config" "$model/config.json" "$data/train.parquet" "$data/test.parquet"
    df -h "$result"
    python - "$model/config.json" <<'PY'
import json
import sys
from importlib.metadata import version
from packaging.version import Version
import torch
import torch_npu

with open(sys.argv[1]) as f:
    model = json.load(f)
print("model:", model["model_type"], "layers:", model["num_hidden_layers"], "experts:", model.get("num_experts"))
if (model["model_type"], model["num_hidden_layers"], model.get("num_experts")) != ("qwen3_moe", 48, 128):
    raise SystemExit("Expected the full Qwen3-30B-A3B checkpoint")
for name in ("torch", "torch-npu", "vllm", "vllm-ascend"):
    print(name, version(name))
for name in ("vllm", "vllm-ascend"):
    if Version(version(name)).release[:2] != (0, 30):
        raise SystemExit(f"The PR's 30B recipe requires matching vLLM/vLLM-Ascend 0.30; got {name}={version(name)}")
count = torch.npu.device_count()
print("Visible NPUs:", count)
if count != 16:
    raise SystemExit(f"Expected 16 visible NPUs, got {count}")
PY
} > "$result/run-info.txt"
save_final=false
if (( steps > 3 )); then save_final=true; fi
# Paths and checkpoint cadence follow the run directory and selected step count.
overrides=(
    "$@"
    "--model.weights_path=$model" "--model.tokenizer_path=$model"
    "--data.train_path=$data/train.parquet" "--data.test_path=$data/test.parquet"
    "--train.max_steps=$steps" "--train.checkpoint.save_final=$save_final"
    "--train.checkpoint.output_dir=$results_root/$run_id/checkpoints"
)
printf 'override=%s\n' "${overrides[@]}" >> "$result/run-info.txt"
threads=$(python -m rl.utils.cpu_threads --config "$config" --local-world-size 16 --report "$result/cpu-threads.json" "${overrides[@]}")
export OMP_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads"
sample_resources > "$result/resources.log" 2>&1 &
monitor_pid=$!
python -m torch.distributed.run --nnodes=4 --node-rank="$node_rank" --nproc-per-node=16 \
    --master-addr="$master_addr" --master-port="${MASTER_PORT:-29751}" \
    hyper_parallel/rl/examples/gsm8k/scripts/run_qwen3_moe_64npu.py "$config" "${overrides[@]}"
