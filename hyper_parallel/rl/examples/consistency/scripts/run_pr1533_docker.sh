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
image=${IMAGE:?Set the verified A3 image}
model_root=${MODEL_ROOT:-${workspace_root}/models/Qwen3-30B-A3B}
data_root=${DATA_ROOT:-${workspace_root}/data/gsm8k}
result_root=${RESULT_ROOT:-/tmp/hyper-rl-consistency}
v4_wheel=${V4_WHEEL:-${workspace_root}/wheels/flash_attn_npu-0.5.0+hpbi1-cp312-cp312-linux_aarch64.whl}
v4_wheel_sha256=${V4_WHEEL_SHA256:-5ef04a872b1be2c812bb0c36c17ba4776a4b5f9b9eabb24eab0a56a139a497c9}
config=${CONFIG:-${script_dir}/../configs/qwen3_moe_4l_rl.yaml}
devices=${DEVICES:-0,1,2,3}
[[ "${devices}" =~ ^[0-9]+(,[0-9]+){3}$ ]] || { printf 'DEVICES must list four NPU IDs\n' >&2; exit 1; }
IFS=',' read -r -a device_list <<<"${devices}"
[[ $(printf '%s\n' "${device_list[@]}" | sort -u | wc -l) -eq 4 ]] || {
    printf 'DEVICES must contain four different NPU IDs\n' >&2; exit 1;
}
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
[[ "${image_id}" == ${EXPECTED_IMAGE_ID:?Set the verified A3 image ID} ]] || {
    printf 'Image identity differs from the validated environment: %s\n' "${image_id}" >&2; exit 1;
}
mkdir -p "${result_root}/run"
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
printf 'Local 4-NPU RL: devices=%s results=%s config=%s\n' "${devices}" "${result_root}" "${config}"
docker run --rm --name "${container}" --privileged --network=host --shm-size=64g \
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
    -v "${RUNTIME_WHEELS_DIR:?Set RUNTIME_WHEELS_DIR}:/runtime-wheels:ro" \
    -w /workspace "${image_id}" /bin/bash -c '
        set -eo pipefail
        source /usr/local/Ascend/cann/set_env.sh
        set -u
        trap "chown -R ${RESULT_UID}:${RESULT_GID} /results" EXIT
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
        sha256sum /runtime-wheels/*.whl > /results/run/runtime-wheels.sha256
        python -m pip install --no-index --no-deps /runtime-wheels/*.whl
        python -m pip check
        python -m pip freeze > /results/run/environment.txt
        python hyper_parallel/rl/examples/consistency/scripts/prepare_checkpoint.py \
                --source /models/Qwen3-30B-A3B --output /results/Qwen3-MoE-4L --layers 4 --reuse
        timeout --signal=TERM --kill-after=60s 7200s \
            python -m torch.distributed.run --standalone --local-addr=127.0.0.1 --nproc_per_node=4 \
            hyper_parallel/rl/examples/consistency/scripts/run_pr1533_consistency.py /run-config.yaml \
            --report /results/run/consistency.json "$@" 2>&1 | tee /results/run/train.log
    ' hyper-rl-consistency "$@"
