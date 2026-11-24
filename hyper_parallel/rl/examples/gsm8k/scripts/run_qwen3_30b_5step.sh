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
scripts=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd -- "$scripts/../../../../.." && pwd)
exec python "$repo/hyper_parallel/rl/examples/consistency/scripts/run_qwen3_moe_cluster.py" \
    --baseline-5step --steps 5 --select-nodes 4 \
    --global-prompt-batch-size 128 --micro-batch-size 2 --response-mini-batch-size 128 \
    --max-prompt 1024 --max-new-tokens 1024 "$@"
