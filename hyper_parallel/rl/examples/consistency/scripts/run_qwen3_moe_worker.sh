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
set -eo pipefail
phase=environment
trap 'rc=$?; printf "CONTAINER_EXIT phase=%s code=%s\n" "$phase" "$rc"; printf "phase=%s\nexit_code=%s\n" "$phase" "$rc" > /results/run/container-status.txt; chown -R "${RESULT_UID}:${RESULT_GID}" /results' EXIT
printf "PHASE environment %s\n" "$(date -Is)"
source /usr/local/Ascend/cann/set_env.sh
set -u
mkdir -p /results/run
rm -f /results/run/consistency.json
printf "%s\n" "${V4_WHEEL_SHA256}" > /results/run/v4-wheel.sha256
unset VLLM_PLUGINS
export PYTHONPATH=/opt/hp-v4:/workspace/hyper_parallel/rl:/workspace:${PYTHONPATH:-}
python -m pip install --no-deps --no-compile --target /opt/hp-v4 /wheels/*.whl
# Keep editable metadata in the container; source remains the mounted checkout.
mkdir /tmp/hp-install
cp setup.py requirements.txt README.md MANIFEST.in /tmp/hp-install/
ln -s /workspace/hyper_parallel /tmp/hp-install/hyper_parallel
python -m pip install --no-deps --no-build-isolation -e /tmp/hp-install
bash hyper_parallel/rl/docker/install_runtime.sh
if [[ -f /runtime-wheels/peft-0.21.2-py3-none-any.whl ]]; then
    sha256sum /runtime-wheels/*.whl > /results/run/runtime-wheels.sha256
    python -m pip install --no-index --no-deps /runtime-wheels/*.whl
fi
python -m pip check
python -m pip freeze > /results/run/environment.txt
if [[ "${RUNNER:-internal}" == codex ]]; then
    [[ "$(codex --version)" == "codex-cli 0.152.1" ]]
    export SHELL=/bin/bash
fi
if [[ "${PLOT_TRAINING:-false}" == true ]]; then
    python -c "from matplotlib.figure import Figure"
fi
phase=checkpoint
printf "PHASE checkpoint %s\n" "$(date -Is)"
export RUN_MODEL_PATH=/models/Qwen3-30B-A3B
if (( MODEL_LAYERS > 0 )); then
    export RUN_MODEL_PATH=/results/Qwen3-MoE-${MODEL_LAYERS}L
    python hyper_parallel/rl/examples/consistency/scripts/prepare_checkpoint.py \
        --source /models/Qwen3-30B-A3B --output "$RUN_MODEL_PATH" --layers "$MODEL_LAYERS" --reuse
fi
phase=configuration
if [[ "${REQUIRE_LEARNING:-false}" == true && -n "${LEARNING_PROMPT_ROWS:-}" ]]; then
    python hyper_parallel/rl/examples/consistency/scripts/select_learning_prompts.py \
        --source /data/gsm8k/train.parquet --rows "$LEARNING_PROMPT_ROWS" \
        --output /results/run/learning-data/train.parquet --minimum-rows "$(( DEVICE_COUNT * 2 ))"
fi
world=$(( DEVICE_COUNT * ${NNODES:-1} ))
python hyper_parallel/rl/examples/consistency/scripts/configure_run.py \
    /run-config.yaml /results/run/generated-config.yaml --world "$world"
phase=train
printf "PHASE train %s\n" "$(date -Is)"
launch=(--standalone --local-addr=127.0.0.1)
if (( ${NNODES:-1} > 1 )); then
    [[ ${MODEL_LAYERS} == 0 ]] || { echo 'Cluster runs require the full model' >&2; exit 1; }
    launch=(--nnodes="$NNODES" --node_rank="$NODE_RANK" --master_addr="$MASTER_ADDR"
            --master_port="$MASTER_PORT" --local-addr="$NODE_ADDR" --max_restarts=0)
fi
set +e
timeout --signal=TERM --kill-after=60s "$RUN_TIMEOUT" \
    python -m torch.distributed.run "${launch[@]}" --nproc_per_node="$DEVICE_COUNT" \
    --log-dir /results/run/torchrun --redirects 3 --tee 3 \
    hyper_parallel/rl/examples/consistency/scripts/run_consistency.py /results/run/generated-config.yaml \
    --report /results/run/consistency.json "$@" 2>&1 | tee /results/run/train.log
train_status=("${PIPESTATUS[@]}")
set -e
printf "TRAIN_EXIT code=%s\n" "${train_status[0]}"
plot_code=0
if [[ "${PLOT_TRAINING:-false}" == true && -s /results/run/steps.jsonl ]]; then
    phase=plot
    set +e
    python hyper_parallel/rl/examples/consistency/scripts/plot_training.py /results/run \
        > /results/run/plot.log 2>&1
    plot_code=$?
    set -e
    cat /results/run/plot.log
    printf "PLOT_EXIT code=%s\n" "$plot_code"
fi
if (( train_status[0] != 0 )); then phase=train; exit "${train_status[0]}"; fi
if (( train_status[1] != 0 )); then phase=train_log; exit "${train_status[1]}"; fi
exit "$plot_code"
