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
"""Framework-free distributed correction launchers; all model imports stay in workers."""
from pathlib import Path

from tests.common.distributed_launcher import torchrun_case


def test_correction_gradient_and_finite_failure(tmp_path, monkeypatch):
    """Compare distributed weighting to a global batch, including an empty local rank."""
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.chdir(tmp_path)
    torchrun_case(str(Path(__file__).with_name("_rollout_correction.py").resolve()),
                  "test_correction_ddp", num_proc=2)


def test_rejected_groups_refill_without_update(tmp_path, monkeypatch):
    """Exercise real producer IPC, lease rejection and same-policy refill."""
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("HYPER_RL_COLOCATE_ST_OUTPUT", str(tmp_path / "checkpoints"))
    monkeypatch.chdir(tmp_path)
    torchrun_case(str(Path(__file__).with_name("_rollout_correction.py").resolve()),
                  "test_rejection_refill", num_proc=2)
