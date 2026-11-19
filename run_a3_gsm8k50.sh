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


# Full-source GSM8K preset: 50 steps, 400 prompt rows and four candidates per prompt.
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export DEVICES=${DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
IFS=, read -r -a selected_devices <<< "$DEVICES"
(( ${#selected_devices[@]} == 16 )) || { printf "This preset requires16 logical devices\n" >&2; exit 2; }
unset LEARNING_PROMPT_ROWS GLOBAL_PROMPT_BATCH_SIZE
export REQUIRE_LEARNING=false
export RUN_TIMEOUT=${RUN_TIMEOUT:-28800}
steps=50
prompts=4
candidates=4
forward=()
while (( $# )); do
    case "$1" in
        --steps|--prompts|--candidates)
            (( $# >= 2 )) || { printf "Missing value for %s\n" "$1" >&2; exit 2; }
            case "$1" in
                --steps) steps=$2 ;; --prompts) prompts=$2 ;; --candidates) candidates=$2 ;;
            esac
            shift 2 ;;
        --help|-h)
            printf "Usage: bash run_a3_gsm8k50.sh [--steps N] [--prompts N] [--candidates N] [--dry-run]\n"
            exit 0 ;;
        *) forward+=("$1"); shift ;;
    esac
done
export STEPS=$steps PROMPTS_PER_RANK=$prompts CANDIDATES=$candidates
export MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
[[ $PROMPTS_PER_RANK =~ ^[1-9][0-9]*$ && $CANDIDATES =~ ^[1-9][0-9]*$ ]] || {
    printf "PROMPTS_PER_RANK and CANDIDATES must be positive integers\n" >&2
    exit 2
}
export RESPONSE_MINI_BATCH_SIZE=$(( PROMPTS_PER_RANK * CANDIDATES ))
export MAX_PROMPT=${MAX_PROMPT:-512} MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
export MAX_NUM_SEQS=${MAX_NUM_SEQS:-2} MC2_CAPACITY=${MC2_CAPACITY:-256}
export INFERENCE_EP_BACKEND=${INFERENCE_EP_BACKEND:-auto}
exec bash "${repo_root}/run_a3.sh" --layers 0 --steps "$STEPS" \
    --train-tp 4 --train-cp 2 --train-ep 4 --train-sp true \
    --rollout-tp 2 --rollout-pcp 1 --rollout-ep 8 --rollout-sp false \
    --prompts "$PROMPTS_PER_RANK" --candidates "$CANDIDATES" --transport ipc \
    --full-data --track-updates --plot "${forward[@]}"
