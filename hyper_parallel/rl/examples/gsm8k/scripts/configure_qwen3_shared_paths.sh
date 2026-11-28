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

# Source on the host or in a container with the shared root mounted at the same path.
gsm8k_shared_path() {
    case "$1" in
        /mnt/share/zjy) printf '%s\n' "$GSM8K_SHARE_ROOT" ;;
        /mnt/share/zjy/*) printf '%s/%s\n' "$GSM8K_SHARE_ROOT" "${1#/mnt/share/zjy/}" ;;
        *) printf '%s\n' "$1" ;;
    esac
}

configure_gsm8k_shared_paths() {
    export GSM8K_SHARE_ROOT="${GSM8K_SHARE_ROOT:-/mnt/share/zjy1}"
    GSM8K_SHARE_ROOT="${GSM8K_SHARE_ROOT%/}"
    if [[ $GSM8K_SHARE_ROOT != /* || ! -d $GSM8K_SHARE_ROOT ]]; then
        echo "Shared directory is unavailable: $GSM8K_SHARE_ROOT; mount it before launching" >&2
        return 1
    fi
    export GSM8K_MODEL="$(gsm8k_shared_path "${GSM8K_MODEL:-$GSM8K_SHARE_ROOT/models/Qwen3-30B-A3B}")"
    export GSM8K_DATA="$(gsm8k_shared_path "${GSM8K_DATA:-$GSM8K_SHARE_ROOT/datasets/gsm8k-6400-500}")"
    export GSM8K_RESULTS="$(gsm8k_shared_path "${GSM8K_RESULTS:-$GSM8K_SHARE_ROOT/rl-cluster-runs}")"
    export GSM8K_IMAGES="$(gsm8k_shared_path "${GSM8K_IMAGES:-$GSM8K_SHARE_ROOT/images}")"
    export GSM8K_CACHE_ROOT="$(gsm8k_shared_path "${GSM8K_CACHE_ROOT:-$GSM8K_SHARE_ROOT/cache}")"
    if [[ -n ${GSM8K_COMPARE_WITH:-} ]]; then
        export GSM8K_COMPARE_WITH="$(gsm8k_shared_path "$GSM8K_COMPARE_WITH")"
    fi
    local node_cache="$GSM8K_CACHE_ROOT/${GSM8K_CACHE_NODE:-$(hostname -s)}"
    export XDG_CACHE_HOME="$node_cache/xdg"
    export HF_HOME="$GSM8K_CACHE_ROOT/huggingface"
    export HF_HUB_CACHE="$HF_HOME/hub" HF_DATASETS_CACHE="$HF_HOME/datasets"
    export PIP_CACHE_DIR="$GSM8K_CACHE_ROOT/pip"
    export TORCH_HOME="$node_cache/torch"
    export TORCHINDUCTOR_CACHE_DIR="$node_cache/torchinductor"
    export TRITON_CACHE_DIR="$node_cache/triton"
    export VLLM_CACHE_ROOT="$node_cache/vllm"
    export ASCEND_CACHE_PATH="$node_cache/ascend"
    export WANDB_CACHE_DIR="$GSM8K_CACHE_ROOT/wandb"
    mkdir -p -- "$XDG_CACHE_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$PIP_CACHE_DIR" \
        "$TORCH_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT" \
        "$ASCEND_CACHE_PATH" "$WANDB_CACHE_DIR"
}

configure_gsm8k_shared_paths
