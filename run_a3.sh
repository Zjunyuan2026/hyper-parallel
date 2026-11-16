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

# Convenience entry for the shared-storage A3 GSM8K consistency recipe.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash run_a3.sh [--steps N] [--tp N] [--ep N] [--sp true|false]
                      [--layers N] [--dry-run]

Defaults: full Qwen3-30B-A3B, 2 steps, TP4/EP4/SP=false, all 16 logical devices.
--tp/--ep/--sp set both training and rollout; --layers 0 selects the full model.
--dry-run prints settings without starting Docker or accessing devices.

Examples:
  bash run_a3.sh
  bash run_a3.sh --steps 10 --tp 2 --ep 8 --sp true
  TRAIN_TP=4 ROLLOUT_TP=2 ROLLOUT_EP=8 bash run_a3.sh

Environment overrides: SHARED_ROOT, MODEL_ROOT, DATA_ROOT, IMAGE,
EXPECTED_IMAGE_ID, IMAGE_ID_FILE, DEVICES, RESULT_ROOT, MODEL_LAYERS, STEPS,
TRAIN_TP/EP/SP, ROLLOUT_TP/EP/SP, MAX_PROMPT, MAX_NEW_TOKENS, MAX_MODEL_LEN,
EP_CAPACITY, PREFILL_TOKENS, PROMPTS_PER_RANK, CANDIDATES.
Default shared root: /mnt/share/zjy. Results stay on this host.
EOF
}

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export SHARED_ROOT=${SHARED_ROOT:-/mnt/share/zjy}
export IMAGE=${IMAGE:-hyper-rl:consistency-v0.30.0-cann9.2.1-a3-runtime-arm64}
export MODEL_ROOT=${MODEL_ROOT:-${SHARED_ROOT}/models/Qwen3-30B-A3B}
export DATA_ROOT=${DATA_ROOT:-${SHARED_ROOT}/datasets/gsm8k/prepared}
export DEVICES=${DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
export MODEL_LAYERS=${MODEL_LAYERS:-0}
export STEPS=${STEPS:-2}
export TRAIN_TP=${TRAIN_TP:-4} TRAIN_EP=${TRAIN_EP:-4} TRAIN_SP=${TRAIN_SP:-false}
export ROLLOUT_TP=${ROLLOUT_TP:-4} ROLLOUT_EP=${ROLLOUT_EP:-4} ROLLOUT_SP=${ROLLOUT_SP:-false}
export MAX_PROMPT=${MAX_PROMPT:-512} MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096} EP_CAPACITY=${EP_CAPACITY:-4096}
export PROMPTS_PER_RANK=${PROMPTS_PER_RANK:-1} CANDIDATES=${CANDIDATES:-2}
# This entry always runs direct GSM8K, without inheriting a Codex runner/profile.
export RUNNER=internal
unset PROFILE
dry_run=false
while (( $# )); do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --dry-run) dry_run=true; shift ;;
        --steps|--tp|--ep|--sp|--layers)
            (( $# >= 2 )) || { printf 'Missing value for %s\n' "$1" >&2; exit 2; }
            case "$1" in
                --steps) export STEPS=$2 ;;
                --tp) export TRAIN_TP=$2 ROLLOUT_TP=$2 ;;
                --ep) export TRAIN_EP=$2 ROLLOUT_EP=$2 ;;
                --sp) export TRAIN_SP=$2 ROLLOUT_SP=$2 ;;
                --layers) export MODEL_LAYERS=$2 ;;
            esac
            shift 2 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done
for name in STEPS TRAIN_TP TRAIN_EP ROLLOUT_TP ROLLOUT_EP; do
    [[ ${!name} =~ ^[1-9][0-9]*$ ]] || { printf '%s must be positive\n' "$name" >&2; exit 2; }
done
[[ $MODEL_LAYERS =~ ^[0-9]+$ ]] || { printf 'MODEL_LAYERS must be nonnegative\n' >&2; exit 2; }
for name in TRAIN_SP ROLLOUT_SP; do
    [[ ${!name} == true || ${!name} == false ]] || {
        printf '%s must be true or false\n' "$name" >&2; exit 2;
    }
done
export RESULT_ROOT=${RESULT_ROOT:-/home/z00988596/results/consistency/$(hostname)/$(date +%Y%m%d-%H%M%S)-gsm8k-$$}
image_id_file=${IMAGE_ID_FILE:-${SHARED_ROOT}/images/a3-runtime.id}
for name in IMAGE MODEL_ROOT DATA_ROOT DEVICES RUNNER MODEL_LAYERS STEPS \
    TRAIN_TP TRAIN_EP TRAIN_SP ROLLOUT_TP ROLLOUT_EP ROLLOUT_SP \
    MAX_PROMPT MAX_NEW_TOKENS MAX_MODEL_LEN EP_CAPACITY PROMPTS_PER_RANK CANDIDATES RESULT_ROOT; do
    printf '%s=%q\n' "$name" "${!name}"
done
printf 'IMAGE_ID_FILE=%q\n' "$image_id_file"
if $dry_run; then exit 0; fi
if [[ -z ${EXPECTED_IMAGE_ID:-} ]]; then
    [[ -r $image_id_file ]] || {
        printf 'Missing image ID file: %s; set IMAGE_ID_FILE or EXPECTED_IMAGE_ID\n' "$image_id_file" >&2
        exit 1
    }
    EXPECTED_IMAGE_ID=$(cat -- "$image_id_file")
    export EXPECTED_IMAGE_ID
fi
# Reuse the existing launcher: image/device guards, read-only inputs and reports.
exec bash "${repo_root}/hyper_parallel/rl/examples/consistency/scripts/run_qwen3_moe_docker.sh"
