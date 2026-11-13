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

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/../../../../.." && pwd)
workspace_root=$(dirname -- "${repo_root}")
if [[ -n "${PROFILE:-}" ]]; then
    source "${PROFILE}"
fi
image=${IMAGE:-swr.cn-east-3.myhuaweicloud.com/huawei-hyper-rl/hyper-rl:consistency-v0.30.0-cann9.2.1-arm64}
model_root=${MODEL_ROOT:-${workspace_root}/models/Qwen3-30B-A3B}
data_root=${DATA_ROOT:-${workspace_root}/data/gsm8k}
result_root=${RESULT_ROOT:-/tmp/hyper-rl-consistency}
v4_wheel=${V4_WHEEL:-${repo_root}/wheels/flash_attn_npu-0.5.0+hpbi1-cp312-cp312-linux_aarch64.whl}
v4_wheel_sha256=${V4_WHEEL_SHA256:-5ef04a872b1be2c812bb0c36c17ba4776a4b5f9b9eabb24eab0a56a139a497c9}
config=${CONFIG:-${script_dir}/../configs/qwen3_moe_4l_rl.yaml}
devices=${DEVICES:-0,1,2,3}
[[ "${devices}" =~ ^[0-9]+(,[0-9]+)*$ ]] || { printf 'DEVICES must list NPU IDs\n' >&2; exit 1; }
IFS=',' read -r -a device_list <<<"${devices}"
device_count=${#device_list[@]}
[[ $(printf '%s\n' "${device_list[@]}" | sort -u | wc -l) -eq "${device_count}" ]] || {
    printf 'DEVICES must contain different NPU IDs\n' >&2; exit 1;
}
[[ "$(printf '%s\n' "${device_list[@]}" | sort -n | paste -sd, -)" == "${devices}" ]] || {
    printf 'DEVICES must be in ascending numeric order\n' >&2; exit 1;
}
model_layers=${MODEL_LAYERS:-4}
[[ "${model_layers}" =~ ^[0-9]+$ ]] || { printf 'MODEL_LAYERS must be 0 (full) or positive\n' >&2; exit 1; }
mkdir -p "${result_root}/run"
[[ ! -e "${result_root}/run/launcher.log" ]] || { printf 'Use a new RESULT_ROOT for each run\n' >&2; exit 1; }
exec > >(tee "${result_root}/run/launcher.log") 2>&1
stage=preflight
trap 'rc=$?; printf "LAUNCHER_EXIT stage=%s code=%s time=%s\n" "${stage}" "${rc}" "$(date -Is)"; printf "stage=%s\nexit_code=%s\n" "${stage}" "${rc}" > "${result_root}/run/launcher-status.txt"' EXIT
printf 'PREFLIGHT time=%s host=%s devices=%s image=%s\n' "$(date -Is)" "$(hostname)" "${devices}" "${image}"
git -C "${repo_root}" rev-parse HEAD > "${result_root}/run/source-commit.txt"
git -C "${repo_root}" status --short > "${result_root}/run/source-status.txt"
sha256sum "${script_dir}"/*.py "${script_dir}"/*.sh "${config}" > "${result_root}/run/source-files.sha256"
free -h > "${result_root}/run/host-memory.txt"
df -h "${result_root}" > "${result_root}/run/host-disk.txt"
for path in "${model_root}/config.json" "${data_root}/train.parquet" "${data_root}/test.parquet" "${v4_wheel}" "${config}"; do
    [[ -f "${path}" ]] || { printf 'Missing input: %s\n' "${path}" >&2; exit 1; }
done
[[ "${v4_wheel_sha256}" =~ ^[0-9a-f]{64}$ ]] || {
    printf 'V4_WHEEL_SHA256 must be a SHA256 digest\n' >&2; exit 1;
}
[[ $(sha256sum "${v4_wheel}" | cut -d ' ' -f 1) == "${v4_wheel_sha256}" ]] || {
    printf 'V4 wheel differs from the expected SHA256\n' >&2; exit 1;
}
image_id=$(docker image inspect "${image}" --format '{{.Id}}')
[[ "${image_id}" == sha256:f139fa4776cf71fa1ebf00c47fa1f4c313deec48267c2102a1a42d968634fb08 ]] || {
    printf 'Image identity differs from the validated environment: %s\n' "${image_id}" >&2; exit 1;
}
npu_info=$(npu-smi info)
printf '%s\n' "${npu_info}" > "${result_root}/run/npu-before.txt"
device_mapping=$(npu-smi info -m)
printf '%s\n' "${device_mapping}" > "${result_root}/run/npu-mapping.txt"
for device in "${device_list[@]}"; do
    physical_id=$(printf '%s\n' "${device_mapping}" | awk -v logical="${device}" \
        '$1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+$/ && $3 ~ /^[0-9]+$/ && $3 == logical {print $1}')
    [[ "${physical_id}" =~ ^[0-9]+$ ]] || {
        printf 'Logical device %s has no unique mapping in npu-smi info -m\n' "${device}" >&2; exit 1;
    }
    [[ "${npu_info}" == *"No running processes found in NPU ${physical_id} "* ]] || {
        printf 'Physical NPU %s for logical device %s is not idle; select idle cards\n' "${physical_id}" "${device}" >&2; exit 1;
    }
    printf 'DEVICE_MAP logical=%s physical_npu=%s\n' "${device}" "${physical_id}"
done
mkdir -p "${result_root}"
result_root=$(cd -- "${result_root}" && pwd)
if [[ ! -d "${result_root}/Qwen3-MoE-4L" ]] && (( $(df -Pk "${result_root}" | awk 'NR==2 {print $4}') < 10 * 1024 * 1024 )); then
    printf 'RESULT_ROOT needs at least 10 GiB free for the cropped checkpoint and run\n' >&2; exit 1
fi
config=$(realpath "${config}")
v4_wheel=$(realpath "${v4_wheel}")
container="hyper-rl-consistency-$$"
portable_options=()
extra_wheels=()
extra_wheels_dir=${EXTRA_WHEELS_DIR:-${repo_root}/wheels/runtime}
if [[ -d "${extra_wheels_dir}" ]]; then
    extra_wheels_dir=$(realpath "${extra_wheels_dir}")
    extra_wheels=(-v "${extra_wheels_dir}:/runtime-wheels:ro")
fi
for option in TRAIN_TP TRAIN_EP TRAIN_SP ROLLOUT_TP ROLLOUT_EP ROLLOUT_SP STEPS RUNNER \
    MODEL_INIT_THREADS PROMPTS_PER_RANK CANDIDATES MAX_PROMPT MAX_MODEL_LEN MAX_NEW_TOKENS EP_CAPACITY PREFILL_TOKENS; do
    if [[ -v "${option}" ]]; then portable_options+=(-e "${option}=${!option}"); fi
done
printf 'RUN_PLAN world=%s devices=%s layers=%s config=%s results=%s\n' "${device_count}" "${devices}" "${model_layers}" "${config}" "${result_root}"
stage=container
docker run --rm --name "${container}" --privileged --network=host --shm-size="${SHM_SIZE:-64g}" \
    "${portable_options[@]}" "${extra_wheels[@]}" -e "DEVICE_COUNT=${device_count}" -e "MODEL_LAYERS=${model_layers}" \
    -e "TZ=${TZ:-Asia/Shanghai}" \
    -e "RUN_TIMEOUT=${RUN_TIMEOUT:-14400}" -e PYTHONUNBUFFERED=1 \
    -e "ASCEND_RT_VISIBLE_DEVICES=${devices}" -e PYTHONDONTWRITEBYTECODE=1 \
    -e HCCL_DETERMINISTIC=true -e OMP_NUM_THREADS=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -e VLLM_HOST_IP=127.0.0.1 -e GLOO_SOCKET_IFNAME=lo \
    -e "HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-lo}" \
    -e "HCCL_IF_BASE_PORT=${HCCL_IF_BASE_PORT:-62400}" \
    -e "HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-62400-62550}" \
    -e "RESULT_UID=$(id -u)" -e "RESULT_GID=$(id -g)" \
    -e "V4_WHEEL_SHA256=${v4_wheel_sha256}" \
    -v /usr/local/dcmi:/usr/local/dcmi:ro \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
    -v "${repo_root}:/workspace:ro" -v "${model_root}:/models/Qwen3-30B-A3B:ro" \
    -v "${data_root}:/data/gsm8k:ro" -v "${result_root}:/results" \
    -v "${v4_wheel}:/wheels/$(basename -- "${v4_wheel}"):ro" -v "${config}:/run-config.yaml:ro" \
    -w /workspace "${image}" /bin/bash -c '
        set -eo pipefail
        phase=environment
        trap '\''rc=$?; printf "CONTAINER_EXIT phase=%s code=%s\n" "$phase" "$rc"; printf "phase=%s\nexit_code=%s\n" "$phase" "$rc" > /results/run/container-status.txt; chown -R "${RESULT_UID}:${RESULT_GID}" /results'\'' EXIT
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
        if [[ "${RUNNER:-codex}" == codex ]]; then
            [[ "$(codex --version)" == "codex-cli 0.152.1" ]]
            export SHELL=/bin/bash
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
        python hyper_parallel/rl/examples/consistency/scripts/configure_run.py \
            /run-config.yaml /results/run/generated-config.yaml --world "$DEVICE_COUNT"
        phase=train
        printf "PHASE train %s\n" "$(date -Is)"
        timeout --signal=TERM --kill-after=60s "$RUN_TIMEOUT" \
            python -m torch.distributed.run --standalone --local-addr=127.0.0.1 --nproc_per_node="$DEVICE_COUNT" \
            --log-dir /results/run/torchrun --redirects 3 --tee 3 \
            hyper_parallel/rl/examples/consistency/scripts/run_consistency.py /results/run/generated-config.yaml \
            --report /results/run/consistency.json "$@" 2>&1 | tee /results/run/train.log
    ' hyper-rl-consistency "$@"
