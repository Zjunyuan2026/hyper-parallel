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
if [[ "${NATIVE_BASELINE:-false}" != true ]]; then
    printf "%s\n" "${V4_WHEEL_SHA256}" > /results/run/v4-wheel.sha256
fi
unset VLLM_PLUGINS
export PYTHONPATH=/workspace/hyper_parallel/rl:/workspace:${PYTHONPATH:-}
if [[ "${NATIVE_BASELINE:-false}" != true ]]; then
    export PYTHONPATH=/opt/hp-v4:$PYTHONPATH
fi
# Resolve before Torch import, using the effective offload override and local ranks only.
cpu_overrides=()
if [[ -v CPU_OFFLOAD ]]; then cpu_overrides+=("--train.accelerator.cpu_offload=${CPU_OFFLOAD}"); fi
for option in "$@"; do
    if [[ $option == --train.accelerator.cpu_offload=* ]]; then cpu_overrides+=("$option"); fi
done
cpu_threads=$(python -m rl.utils.cpu_threads --config /run-config.yaml \
    --local-world-size "$DEVICE_COUNT" --report /results/run/cpu-threads.json "${cpu_overrides[@]}")
export OMP_NUM_THREADS="$cpu_threads" MKL_NUM_THREADS="$cpu_threads"
if [[ "${NATIVE_BASELINE:-false}" != true ]]; then
    python -m pip install --no-deps --no-compile --target /opt/hp-v4 /wheels/*.whl
fi
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
training_script=hyper_parallel/rl/examples/consistency/scripts/run_consistency.py
training_report=/results/run/consistency.json
if [[ "${NATIVE_BASELINE:-false}" == true ]]; then
    training_script=hyper_parallel/rl/examples/gsm8k/scripts/run_qwen3_30b_native.py
    training_report=/results/run/native-training.json
fi
timeout --signal=TERM --kill-after=60s "$RUN_TIMEOUT" \
    python -m torch.distributed.run "${launch[@]}" --nproc_per_node="$DEVICE_COUNT" \
    --log-dir /results/run/torchrun --redirects 3 --tee 3 \
    "$training_script" /results/run/generated-config.yaml \
    --report "$training_report" "$@" 2>&1 | tee /results/run/train.log
train_status=("${PIPESTATUS[@]}")
set -e
printf "TRAIN_EXIT code=%s\n" "${train_status[0]}"
plot_code=0
if [[ "${PLOT_TRAINING:-false}" == true && -s /results/run/steps.jsonl ]]; then
    phase=plot
    set +e
    plot_script=plot_training.py
    if [[ "${FORMAL_GSM8K:-false}" == true ]]; then plot_script=plot_full_gsm8k.py; fi
    python "hyper_parallel/rl/examples/consistency/scripts/${plot_script}" /results/run \
        > /results/run/plot.log 2>&1
    plot_code=$?
    set -e
    cat /results/run/plot.log
    printf "PLOT_EXIT code=%s\n" "$plot_code"
fi
if (( train_status[0] != 0 )); then phase=train; exit "${train_status[0]}"; fi
if (( train_status[1] != 0 )); then phase=train_log; exit "${train_status[1]}"; fi
exit "$plot_code"
