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
"""Torchrun worker for bounded HCCL correctness and local-duration measurements."""

from datetime import timedelta
import json
import os
import time

import torch
import torch.distributed as dist
import torch_npu  # pylint: disable=unused-import


def main() -> None:
    """Verify every element after each reduction and report completed-device timing."""
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.npu.set_device(local_rank)
    dist.init_process_group("hccl", timeout=timedelta(seconds=300))
    results = []
    expected = world * (world + 1) / 2
    try:
        for elements in (1, 1048576, 16777216):
            tensor = torch.empty(elements, device=f"npu:{local_rank}", dtype=torch.float32)
            elapsed = []
            for iteration in range(4):
                tensor.fill_(rank + 1)
                torch.npu.synchronize()
                dist.barrier()
                started = time.perf_counter()
                dist.all_reduce(tensor)
                torch.npu.synchronize()
                seconds = time.perf_counter() - started
                if not bool(torch.all(tensor == expected).item()):
                    raise RuntimeError(f"AllReduce mismatch: rank={rank}, elements={elements}, expected={expected}")
                if iteration:
                    elapsed.append(seconds)
            results.append({"bytes": tensor.numel() * tensor.element_size(),
                            "mean_seconds": sum(elapsed) / len(elapsed)})
            del tensor
        dist.barrier()
    finally:
        dist.destroy_process_group()
    print("HCCL_PROBE_RESULT " + json.dumps({"rank": rank, "local_rank": local_rank,
                                            "world_size": world, "passed": True, "measurements": results}),
          flush=True)


if __name__ == "__main__":
    main()
