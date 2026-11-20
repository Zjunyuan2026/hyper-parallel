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
                      [--layers N] [--train-tp N] [--rollout-tp N]
                      [--train-ep N] [--rollout-ep N] [--train-sp BOOL] [--rollout-sp BOOL]
                      [--micro-batch N] [--prompts N] [--candidates N]
                      [--transport auto|ipc] [--train-cp N] [--rollout-pcp N] [--require-learning] [--full-data] [--track-updates] [--plot] [--dry-run]

Defaults: full Qwen3-30B-A3B, 2 steps, TP4/EP4/SP=false, all 16 logical devices.
Each candidate generates at most 32 tokens for consistency validation.
--tp/--ep/--sp set both training and rollout; --layers 0 selects the full model.
--dry-run prints settings without starting Docker or accessing devices.

Examples:
  bash run_a3.sh
  bash run_a3.sh --steps 10 --tp 2 --ep 8 --sp true
  TRAIN_TP=4 ROLLOUT_TP=2 ROLLOUT_EP=8 bash run_a3.sh

Environment overrides: SHARED_ROOT, MODEL_ROOT, DATA_ROOT, IMAGE,
EXPECTED_IMAGE_ID, IMAGE_ID_FILE, DEVICES, RESULT_ROOT, MODEL_LAYERS, STEPS,
TRAIN_TP/EP/SP, ROLLOUT_TP/EP/SP, TRAIN_CP, ROLLOUT_PCP, MAX_PROMPT, MAX_NEW_TOKENS, MAX_MODEL_LEN,
EP_CAPACITY, PREFILL_TOKENS, PROMPTS_PER_RANK, CANDIDATES.
RESULT_BASE (or inherited RESULT_ROOT) is a parent directory, not a reusable run.
Every invocation creates a unique child directory, including host and topology.
CPU_OFFLOAD (true|false), MICRO_BATCH_SIZE, RESPONSE_MINI_BATCH_SIZE, GLOBAL_PROMPT_BATCH_SIZE,
MAX_NUM_SEQS, KV_CACHE_MEMORY_BYTES, WEIGHT_SYNC_TRANSPORT, WEIGHT_BUCKET_MB,
GENERATION_TIMEOUT (HTTP request timeout; default 1800),
MC2_CAPACITY (1..256), INFERENCE_EP_BACKEND (auto|mc2|allgather),
REQUIRE_LEARNING, FULL_DATASET, TRACK_UPDATES, PLOT_TRAINING (true|false),
LEARNING_PROMPT_ROWS (original GSM8K row indices).
--full-data loads and shuffles the complete training split; 50 steps need not cover one epoch.
--track-updates records actual updates without requiring a nonzero first step.
--plot writes reward, loss and strict logp-difference plots to run/plots.
are also supported. GLOBAL_PROMPT_BATCH_SIZE keeps workload fixed across TP.
Missing images load from IMAGE_ARCHIVE (default shared images/a3-runtime.tar.gz).
A stopped image-reference container is retained; no NPU is used by that container.
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
export CPU_OFFLOAD=${CPU_OFFLOAD:-true}
export STEPS=${STEPS:-2}
export TRAIN_TP=${TRAIN_TP:-4} TRAIN_EP=${TRAIN_EP:-4} TRAIN_SP=${TRAIN_SP:-false}
export ROLLOUT_TP=${ROLLOUT_TP:-4} ROLLOUT_EP=${ROLLOUT_EP:-4} ROLLOUT_SP=${ROLLOUT_SP:-false}
export TRAIN_CP=${TRAIN_CP:-1} ROLLOUT_PCP=${ROLLOUT_PCP:-1}
export MAX_PROMPT=${MAX_PROMPT:-128} MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-32}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-256} EP_CAPACITY=${EP_CAPACITY:-256}
export PREFILL_TOKENS=${PREFILL_TOKENS:-128}
export MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1} MAX_NUM_SEQS=${MAX_NUM_SEQS:-2}
export GENERATION_TIMEOUT=${GENERATION_TIMEOUT:-1800}
export WEIGHT_SYNC_TRANSPORT=${WEIGHT_SYNC_TRANSPORT:-auto}
export PROMPTS_PER_RANK=${PROMPTS_PER_RANK:-1} CANDIDATES=${CANDIDATES:-2}
# This entry always runs direct GSM8K, without inheriting a Codex runner/profile.
export REQUIRE_LEARNING=${REQUIRE_LEARNING:-false}
export FULL_DATASET=${FULL_DATASET:-false} TRACK_UPDATES=${TRACK_UPDATES:-false}
export PLOT_TRAINING=${PLOT_TRAINING:-false}
export RUNNER=internal
unset PROFILE
dry_run=false
while (( $# )); do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --dry-run) dry_run=true; shift ;;
        --require-learning) export REQUIRE_LEARNING=true; shift ;;
        --full-data) export FULL_DATASET=true; shift ;;
        --track-updates) export TRACK_UPDATES=true; shift ;;
        --plot) export PLOT_TRAINING=true; shift ;;
        --train-cp|--rollout-pcp|--train-tp|--train-ep|--train-sp|--rollout-tp|--rollout-ep|--rollout-sp|--micro-batch|--prompts|--candidates|--transport)
            (( $# >= 2 )) || { printf 'Missing value for %s\n' "$1" >&2; exit 2; }
            case "$1" in
                --train-cp) export TRAIN_CP=$2 ;; --rollout-pcp) export ROLLOUT_PCP=$2 ;;
                --train-tp) export TRAIN_TP=$2 ;; --train-ep) export TRAIN_EP=$2 ;;
                --train-sp) export TRAIN_SP=$2 ;; --rollout-tp) export ROLLOUT_TP=$2 ;;
                --rollout-ep) export ROLLOUT_EP=$2 ;; --rollout-sp) export ROLLOUT_SP=$2 ;;
                --micro-batch) export MICRO_BATCH_SIZE=$2 ;; --prompts) export PROMPTS_PER_RANK=$2 ;;
                --candidates) export CANDIDATES=$2 ;; --transport) export WEIGHT_SYNC_TRANSPORT=$2 ;;
            esac
            shift 2 ;;
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
for name in STEPS TRAIN_TP TRAIN_CP TRAIN_EP ROLLOUT_TP ROLLOUT_PCP ROLLOUT_EP GENERATION_TIMEOUT; do
    [[ ${!name} =~ ^[1-9][0-9]*$ ]] || { printf '%s must be positive\n' "$name" >&2; exit 2; }
done
[[ $MODEL_LAYERS =~ ^[0-9]+$ ]] || { printf 'MODEL_LAYERS must be nonnegative\n' >&2; exit 2; }
for name in CPU_OFFLOAD \
    TRAIN_SP ROLLOUT_SP REQUIRE_LEARNING FULL_DATASET TRACK_UPDATES PLOT_TRAINING; do
    [[ ${!name} == true || ${!name} == false ]] || {
        printf '%s must be true or false\n' "$name" >&2; exit 2;
    }
done
if [[ $WEIGHT_SYNC_TRANSPORT != auto && $WEIGHT_SYNC_TRANSPORT != ipc ]]; then
    printf 'Colocated HCCL weight transfer is unsupported: Actor and rollout share a device; use --transport ipc\n' >&2
    exit 2
fi
if [[ $FULL_DATASET == true && -n "${LEARNING_PROMPT_ROWS:-}" ]]; then
    printf 'Full dataset mode requires LEARNING_PROMPT_ROWS to be unset\n' >&2
    exit 2
fi
result_base=${RESULT_BASE:-${RESULT_ROOT:-/home/z00988596/results/consistency}}
run_name="$(hostname)-$(date +%Y%m%d-%H%M%S)-tp${TRAIN_TP}-ep${TRAIN_EP}-cp${TRAIN_CP}-rtp${ROLLOUT_TP}-pcp${ROLLOUT_PCP}-rep${ROLLOUT_EP}"
export RESULT_ROOT="${result_base}/${run_name}-$$"
if ! $dry_run; then
    mkdir -p -- "$result_base"
    RESULT_ROOT=$(mktemp -d "${result_base}/${run_name}-XXXXXX")
    export RESULT_ROOT
fi
image_id_file=${IMAGE_ID_FILE:-${SHARED_ROOT}/images/a3-runtime.id}
for name in IMAGE MODEL_ROOT DATA_ROOT DEVICES RUNNER MODEL_LAYERS \
    CPU_OFFLOAD STEPS \
    TRAIN_TP TRAIN_CP TRAIN_EP TRAIN_SP ROLLOUT_TP ROLLOUT_PCP ROLLOUT_EP ROLLOUT_SP \
    MAX_PROMPT MAX_NEW_TOKENS MAX_MODEL_LEN EP_CAPACITY PROMPTS_PER_RANK CANDIDATES MICRO_BATCH_SIZE MAX_NUM_SEQS \
    WEIGHT_SYNC_TRANSPORT GENERATION_TIMEOUT PREFILL_TOKENS REQUIRE_LEARNING FULL_DATASET TRACK_UPDATES PLOT_TRAINING RESULT_ROOT; do
    printf '%s=%q\n' "$name" "${!name}"
done
for name in RESPONSE_MINI_BATCH_SIZE GLOBAL_PROMPT_BATCH_SIZE KV_CACHE_MEMORY_BYTES RUN_TIMEOUT \
    MC2_CAPACITY INFERENCE_EP_BACKEND; do
    if [[ -v "$name" ]]; then printf "%s=%q\n" "$name" "${!name}"; fi
done
printf 'IMAGE_ID_FILE=%q\n' "$image_id_file"
if $dry_run; then exit 0; fi
if [[ -z ${EXPECTED_IMAGE_ID:-} ]]; then
    [[ -r $image_id_file ]] || {
        printf 'Missing image ID file: %s; set IMAGE_ID_FILE or EXPECTED_IMAGE_ID\n' "$image_id_file" >&2
        exit 1
    }
    EXPECTED_IMAGE_ID=$(tr -d '\r\n' < "$image_id_file")
    export EXPECTED_IMAGE_ID
fi
[[ $EXPECTED_IMAGE_ID =~ ^sha256:[0-9a-f]{64}$ ]] || { echo 'Invalid expected image ID' >&2; exit 1; }
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    archive=${IMAGE_ARCHIVE:-${SHARED_ROOT}/images/a3-runtime.tar.gz}
    [[ -r $archive ]] || { printf 'Missing image and archive: %s\n' "$archive" >&2; exit 1; }
    docker load -i "$archive"
fi
actual=$(docker image inspect "$IMAGE" --format '{{.Id}}')
[[ $actual == "$EXPECTED_IMAGE_ID" ]] || { echo 'Image ID differs from expected ID' >&2; exit 1; }
keep="hyper-rl-image-keep-${actual#sha256:}"
if docker container inspect "$keep" >/dev/null 2>&1; then
    [[ $(docker container inspect "$keep" --format '{{.Image}}') == "$actual" ]] || {
        echo 'Image-reference container points to another image' >&2; exit 1;
    }
else
    docker create --name "$keep" --entrypoint /bin/true "$actual" >/dev/null
fi
printf 'IMAGE_REFERENCE_CONTAINER=%s\n' "$keep"
# Reuse the existing launcher: image/device guards, read-only inputs and reports.
exec bash "${repo_root}/hyper_parallel/rl/examples/consistency/scripts/run_qwen3_moe_docker.sh"
