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
"""Framework-free launcher using the repository's standard distributed ST helper."""
from pathlib import Path

from tests.common.distributed_launcher import torchrun_case


def test_async_grpo_gloo(tmp_path, monkeypatch):
    """Exercise real Actor/DDP updates and spawned production protocol on two CPU ranks."""
    worker = Path(__file__).with_name("_async_train.py").resolve()
    monkeypatch.setenv("USE_TF", "0")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("HYPER_RL_ASYNC_ST_OUTPUT", str(tmp_path / "checkpoints"))
    monkeypatch.chdir(tmp_path)
    torchrun_case(str(worker), "test_async_worker", num_proc=2)
