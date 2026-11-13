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
# Preserve the Docker target even when BASH_ENV changes SOC_VERSION before this script starts.
readonly hp_a3_build_soc="${HP_A3_TARGET_SOC:-${SOC_VERSION:-}}"
printf 'A3_BUILD_INPUT target=%q shell_soc=%q BASH_ENV=%q\n' \
    "$hp_a3_build_soc" "${SOC_VERSION:-}" "${BASH_ENV:-}"
if [[ ! "$hp_a3_build_soc" =~ ^ascend910_93[0-9][0-9]$ ]]; then
    printf 'Invalid build SOC_VERSION=%q; expected an A3 target such as ascend910_9382\n' \
        "$hp_a3_build_soc" >&2
    exit 1
fi
source /usr/local/Ascend/cann/set_env.sh
if [[ -f /usr/local/Ascend/nnal/atb/set_env.sh ]]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi
set -u
printf 'A3_BUILD_TARGET requested=%q environment_after_source=%q\n' \
    "$hp_a3_build_soc" "${SOC_VERSION:-}"
export SOC_VERSION="$hp_a3_build_soc"
# CANN/ATB are already loaded; child build shells must inherit this target unchanged.
export BASH_ENV=/dev/null

source_dir=/vllm-workspace/vllm-ascend
record_dir=/opt/a3-vllm-build
[[ "$(realpath "$source_dir")" == /vllm-workspace/vllm-ascend ]]
[[ "${MAX_JOBS:-8}" =~ ^[1-9][0-9]*$ ]] || {
    printf 'MAX_JOBS must be positive\n' >&2; exit 1;
}
test -f "$source_dir/setup.py"
test -d "$source_dir/csrc/third_party/catlass/include"
grep -qx 'package_name=Ascend-cann-A3-ops' \
    /usr/local/Ascend/cann-9.2.1/aarch64-linux/ascend_ops_install.info
mkdir -p "$record_dir"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH:-}
export COMPILE_CUSTOM_KERNELS=1
export MAX_JOBS=${MAX_JOBS:-8}
# Use the official A3 build switch only during compilation, not at runtime.
export VLLM_BATCH_INVARIANT=1
export SETUPTOOLS_SCM_PRETEND_VERSION
SETUPTOOLS_SCM_PRETEND_VERSION=$(python -c 'from importlib.metadata import version; print(version("vllm-ascend"))')
python - "$record_dir" <<'PY'
import json
from importlib.metadata import version
from pathlib import Path
import sys
names = ('torch', 'torch-npu', 'vllm', 'vllm-ascend', 'transformers', 'triton-ascend')
Path(sys.argv[1], 'versions-before.json').write_text(json.dumps({name: version(name) for name in names}, indent=2))
PY
printf 'SOC_VERSION=%s\nMAX_JOBS=%s\n' "$SOC_VERSION" "$MAX_JOBS" > "$record_dir/target.txt"
git -C "$source_dir" rev-parse HEAD > "$record_dir/source-commit.txt"
git -C "$source_dir" diff > "$record_dir/source-before.patch"

# These are generated A2 outputs in the derived image, never host directories.
rm -rf /vllm-workspace/vllm-ascend/build /vllm-workspace/vllm-ascend/csrc/build \
    /vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom
find "$source_dir/vllm_ascend" -maxdepth 1 -type f -name '*.so' -delete
cd "$source_dir"
python -m pip install -v --no-build-isolation --no-deps -e . 2>&1 | tee "$record_dir/build.log"

python - "$record_dir" "$source_dir" <<'PY'
import ast
import json
from importlib.metadata import version
from pathlib import Path
import sys
record, source = map(Path, sys.argv[1:])
before = json.loads((record / 'versions-before.json').read_text())
after = {name: version(name) for name in before}
(record / 'versions-after.json').write_text(json.dumps(after, indent=2))
if before != after:
    raise RuntimeError(f'Package versions changed: before={before}, after={after}')
build_info = source / 'vllm_ascend/_build_info.py'
tree = ast.parse(build_info.read_text())
values = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
          if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
if values.get('__device_type__') != 'A3':
    raise RuntimeError(f'Expected an A3 build, got {values}')
native = list((source / 'vllm_ascend').glob('vllm_ascend_C*.so'))
if not native or not (source / 'vllm_ascend/libvllm_ascend_kernels.so').is_file():
    raise RuntimeError('Rebuilt native libraries are missing')
(record / 'build-info.py').write_text(build_info.read_text())
print('A3_BUILD_METADATA_PASS; hardware kernels still require target-device tests', flush=True)
PY
find "$source_dir/vllm_ascend" -maxdepth 1 -type f -name '*.so' \
    -exec sha256sum {} + > "$record_dir/native-sha256.txt"
python -m pip check
