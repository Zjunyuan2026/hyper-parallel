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
"""CPU-only native control-path tests without importing Torch or creating process groups."""

import ast
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[4]


def load_function(path: str, name: str, bindings: dict):
    """Compile the actual function with mocked optional runtime dependencies."""
    tree = ast.parse((ROOT / path).read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    future = ast.parse("from __future__ import annotations").body
    module = ast.fix_missing_locations(ast.Module(body=[*future, function], type_ignores=[]))
    exec(compile(module, path, "exec"), bindings)
    return bindings[name]


class NativeBaselineTests(unittest.TestCase):
    """Verify native admission and reporting without weakening unified probability gates."""

    def test_node_local_native_ep_allows_device_residency(self) -> None:
        """Native resident mode requires node-local expert parallelism; other guards remain enforced."""
        bindings = {
            "torch": SimpleNamespace(accelerator=SimpleNamespace(
                current_accelerator=lambda: SimpleNamespace(type="npu"))),
            "optional_mapping": lambda mapping, key: mapping.get(key, {}),
            "_validate_vllm_basics": lambda _config: ("colocated", 8, 2),
            "_trainer_world_size": lambda _config: 64,
            "_validate_colocated_vllm": Mock(), "_validate_disjoint_vllm": Mock(),
            "_validate_vllm_limits": Mock(), "_validate_dense_parallelism": Mock(),
            "resolve_vllm_model": lambda model, _implementation: model,
            "resolve_weight_sync_config": Mock(),
        }
        validate = load_function("hyper_parallel/rl/rl/config/validate.py", "_validate_vllm", bindings)
        rollout = {"vllm": {"node_local_world_size": 16, "model_implementation": "native",
                            "enable_expert_parallel": True}}
        model = SimpleNamespace(family="qwen3_moe")
        validate(rollout, {"cpu_offload": True, "tp": 4}, model)
        bindings["_validate_colocated_vllm"].assert_called_once()
        validate(rollout, {"cpu_offload": False, "tp": 4}, model)
        rollout["vllm"]["enable_expert_parallel"] = False
        with self.assertRaisesRegex(ValueError, "device-resident"):
            validate(rollout, {"cpu_offload": False, "tp": 4}, model)
        with self.assertRaisesRegex(ValueError, "Node-local"):
            validate(rollout, {"cpu_offload": True}, SimpleNamespace(family="qwen3"))
        rollout["vllm"]["model_implementation"] = "unified"
        validate(rollout, {"cpu_offload": False, "tp": 4}, model)

    def test_native_training_report_does_not_assert_probability_parity(self) -> None:
        """A completed ordinary run is distinct from a strict consistency pass."""
        class Observer:
            def __init__(self, _tracker, _directory):
                self.rows = []

        class Recorder:
            def __init__(self, _directory):
                pass

            def call(self, _phase, callback):
                return callback()

        trainer = SimpleNamespace(train_dataloader=SimpleNamespace(dataset=[None] * 6400),
                                  state=SimpleNamespace(global_step=0), _tracker=None)

        def train():
            trainer.state.global_step = 2
            trainer._tracker.rows = [{"training/rollout_log_probs_diff_max": 1.0}] * 2

        trainer.train = train
        run = load_function("hyper_parallel/rl/examples/gsm8k/scripts/run_qwen3_30b_native.py", "run", {
            "json": json, "time": time, "os": os, "PhaseRecorder": Recorder,
            "MetricsObserver": Observer, "build_trainer": lambda _config: trainer,
        })
        config = {"rollout": {"vllm": {"model_implementation": "native"}}, "train": {"max_steps": 2}}
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {"RANK": "0"}):
            report = Path(temporary) / "native-training.json"
            run(config, report)
            result = json.loads(report.read_text())
            self.assertTrue(result["passed"])
            self.assertFalse(result["strict_consistency"])
            self.assertEqual(result["completed_steps"], 2)
            with self.assertRaisesRegex(ValueError, "no consistency"):
                run({**config, "consistency": {}}, report)
            trainer.train = lambda: None
            trainer.state.global_step = 1
            with self.assertRaises(RuntimeError):
                run(config, report)
