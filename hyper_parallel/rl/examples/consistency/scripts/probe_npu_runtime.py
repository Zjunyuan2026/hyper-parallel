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
"""Check Level 3 BF16 operators and the vLLM-Ascend hardware build type."""

import json
import os

import torch
import torch_npu

from vllm_ascend.device.device_config import check_ascend_device_type, get_ascend_device_type


def main() -> None:
    """Fail on missing operators before launching the distributed training job."""
    print("RUNTIME_PROBE_START", json.dumps({
        "torch": str(torch.__version__), "torch_npu": str(torch_npu.__version__),
        "opp": os.environ.get("ASCEND_OPP_PATH"),
        "custom_opp": os.environ.get("ASCEND_CUSTOM_OPP_PATH"),
        "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
    }), flush=True)
    torch_npu.npu.set_device(0)
    print("DEVICE", torch_npu.npu.get_device_name(0), flush=True)
    torch_npu.npu.set_deterministic_level(3)
    torch.utils.deterministic.fill_uninitialized_memory = False
    left = torch.ones((1, 16), dtype=torch.bfloat16, device="npu:0")
    right = torch.ones((16, 16), dtype=torch.bfloat16, device="npu:0")
    torch_npu.npu.synchronize()
    print("ONES_OK", flush=True)
    result = torch.mm(left, right)
    torch_npu.npu.synchronize()
    expected = torch.full((1, 16), 16, dtype=torch.bfloat16)
    torch.testing.assert_close(result.cpu(), expected, atol=0, rtol=0)
    print("MM_OK", flush=True)
    print("VLLM_ASCEND_BUILD_DEVICE", str(get_ascend_device_type()), flush=True)
    check_ascend_device_type()
    print("RUNTIME_PROBE_PASS: BF16 ones/mm Level 3 and vLLM-Ascend device match", flush=True)


if __name__ == "__main__":
    main()
