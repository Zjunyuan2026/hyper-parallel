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
"""Framework-free launcher for actual CPU/DDP updates and sampling transport."""
from pathlib import Path

import pytest

from tests.common.distributed_launcher import torchrun_case


@pytest.mark.parametrize("mode", ["off", "metrics_only", "decoupled"])
def test_colocate_grpo_gloo(tmp_path, monkeypatch, mode):
    """Check coordinator/collective order and checkpoint recovery on two CPU ranks."""
    monkeypatch.setenv("USE_TF", "0")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("HYPER_RL_COLOCATE_ST_CORRECTION", mode)
    monkeypatch.setenv("HYPER_RL_COLOCATE_ST_OUTPUT", str(tmp_path / "checkpoints"))
    worker = Path(__file__).with_name("_colocate_train.py").resolve()
    monkeypatch.chdir(tmp_path)
    torchrun_case(str(worker), "test_colocate_worker", num_proc=2)
