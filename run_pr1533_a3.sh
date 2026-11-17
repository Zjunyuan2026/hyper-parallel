#!/usr/bin/env bash
# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export IMAGE=${IMAGE:-hyper-rl:consistency-v0.30.0-cann9.2.1-a3-runtime-arm64}
export MODEL_ROOT=${MODEL_ROOT:-/mnt/share/zjy/models/Qwen3-30B-A3B}
export DATA_ROOT=${DATA_ROOT:-/mnt/share/zjy/datasets/gsm8k/prepared}
export DEVICES=${DEVICES:-0,1,2,3}
[[ $# == 0 ]] || { echo "Usage: DEVICES=0,1,2,3 bash run_pr1533_a3.sh" >&2; exit 2; }
export RESULT_ROOT=${RESULT_BASE:-/home/z00988596/results/consistency}/pr1533-4dev-4layer-$(date +%Y%m%d-%H%M%S)-$$
if [[ -z ${EXPECTED_IMAGE_ID:-} ]]; then
    image_id_file=${IMAGE_ID_FILE:-/mnt/share/zjy/images/a3-runtime.id}
    [[ -r $image_id_file ]] || { echo "Missing trusted image ID: $image_id_file" >&2; exit 1; }
    EXPECTED_IMAGE_ID=$(tr -d '\r\n' < "$image_id_file")
fi
[[ $EXPECTED_IMAGE_ID =~ ^sha256:[0-9a-f]{64}$ ]] || { echo "Invalid image ID" >&2; exit 1; }
export EXPECTED_IMAGE_ID
export V4_WHEEL=${V4_WHEEL:-$PWD/wheels/flash_attn_npu-0.5.0+hpbi1-cp312-cp312-linux_aarch64.whl}
export V4_WHEEL_SHA256=${V4_WHEEL_SHA256:-5ef04a872b1be2c812bb0c36c17ba4776a4b5f9b9eabb24eab0a56a139a497c9}
export RUNTIME_WHEELS_DIR=$PWD/wheels/runtime
export CONFIG=$PWD/hyper_parallel/rl/examples/consistency/configs/qwen3_moe_4l_rl.yaml
mkdir -p "$RESULT_ROOT"
echo "PR_HEAD=d19b6b18c9d0bb13478ee3f0d52a19a9bb8119aa" | tee "$RESULT_ROOT/baseline.txt"
echo "RESULT_ROOT=$RESULT_ROOT"
set +e
bash hyper_parallel/rl/examples/consistency/scripts/run_pr1533_docker.sh 2>&1 | tee "$RESULT_ROOT/console.log"
rc=${PIPESTATUS[0]}
printf '%s\n' "$rc" > "$RESULT_ROOT/exit-code.txt"
exit "$rc"
