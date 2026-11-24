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
"""Audit full-epoch coverage using the production sampler and actual baseline observer."""

import ast
import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]


def load_class(path: str, name: str, bindings: dict):
    """Load a CPU class with explicit stand-ins for optional device dependencies."""
    tree = ast.parse((ROOT / path).read_text())
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
    future = ast.parse("from __future__ import annotations").body
    module = ast.fix_missing_locations(ast.Module(body=[*future, node], type_ignores=[]))
    exec(compile(module, path, "exec"), bindings)
    return bindings[name]


class BaselineCoverageTests(unittest.TestCase):
    """Accept real DP sample order, and reject duplicates and incomplete epochs."""

    def test_pilot_and_full_epoch_coverage(self) -> None:
        for steps in (2, 5, 50):
            with self.subTest(steps=steps):
                self._exercise_coverage(steps)

    def _exercise_coverage(self, steps: int) -> None:
        sampler = load_class("hyper_parallel/rl/rl/data/prompts.py", "_DistributedPromptSampler", {"random": random})
        shards = [list(sampler(6400, rank=rank, world_size=8, seed=42, shuffle=False)) for rank in range(8)]
        trainer = SimpleNamespace(state=SimpleNamespace(max_steps=steps, global_step=0),
                                  train_dataloader=SimpleNamespace(dataset=[None] * 6400),
                                  rollout_engine=SimpleNamespace(is_request_owner=True),
                                  _generate_training_rollout=lambda _batch, _timings: "generated",
                                  _run_rank_synchronized=lambda _operation, callback: callback())

        def gather(output, _local):
            start = trainer.state.global_step * 16
            for rank in range(8):
                output[rank * 8] = shards[rank][start:start + 16]

        dist = SimpleNamespace(get_rank=lambda: 0, get_world_size=lambda: 64, all_gather_object=gather)
        coverage = load_class("hyper_parallel/rl/examples/consistency/scripts/run_consistency.py",
                              "BaselineCoverage", {"dist": dist, "json": json})
        with tempfile.TemporaryDirectory() as temporary:
            observer = coverage(trainer, Path(temporary))
            for step in range(steps):
                trainer.state.global_step = step
                self.assertEqual(trainer._generate_training_rollout({"sample_indices": []}, {}), "generated")
            observer.finish()
            records = json.loads((Path(temporary) / "coverage.json").read_text())
            self.assertEqual(len(records), steps)
            self.assertEqual(len({index for row in records for index in row["indices"]}), steps * 128)
            trainer.state.global_step = 0
            shards[1][0] = shards[0][0]
            with self.assertRaisesRegex(ValueError, "coverage"):
                trainer._generate_training_rollout({"sample_indices": []}, {})
            observer.records.pop()
            with self.assertRaisesRegex(ValueError, "incomplete"):
                observer.finish()
