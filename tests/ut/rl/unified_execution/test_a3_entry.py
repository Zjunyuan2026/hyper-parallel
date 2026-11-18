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
"""CPU-only regressions for the public A3 shell entry."""
import os
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[4]


class A3EntryTests(unittest.TestCase):
    """Inherited run paths must not cause repeated runs to overwrite results."""

    def test_repeated_inherited_directory_creates_distinct_children(self) -> None:
        env = {**os.environ, "RESULT_ROOT": "/tmp/prior-a3-run"}
        env.pop("RESULT_BASE", None)
        paths = []
        for _ in range(2):
            result = subprocess.run(["bash", str(ROOT / "run_a3.sh"), "--dry-run"],
                                    env=env, capture_output=True, text=True, check=True)
            paths.append(next(line for line in result.stdout.splitlines() if line.startswith("RESULT_ROOT=")))
        self.assertNotEqual(paths[0], paths[1])
        self.assertTrue(all(path.startswith("RESULT_ROOT=/tmp/prior-a3-run/") for path in paths))

    def test_independent_train_rollout_options(self) -> None:
        result = subprocess.run(["bash", str(ROOT / "run_a3.sh"), "--dry-run", "--train-tp", "2",
                                 "--rollout-tp", "4", "--train-ep", "8", "--transport", "ipc"],
                                capture_output=True, text=True, check=True)
        self.assertIn("TRAIN_TP=2\n", result.stdout)
        self.assertIn("ROLLOUT_TP=4\n", result.stdout)
        self.assertIn("TRAIN_EP=8\n", result.stdout)

    def test_real_learning_check_is_explicit(self) -> None:
        """Expose opt-in real reward validation without starting a container."""
        result = subprocess.run(["bash", str(ROOT / "run_a3.sh"), "--require-learning", "--dry-run",
                                 "--steps", "2", "--candidates", "8"],
                                capture_output=True, text=True, check=True)
        self.assertIn("REQUIRE_LEARNING=true\n", result.stdout)
        self.assertIn("CANDIDATES=8\n", result.stdout)

    def test_colocated_hccl_is_rejected_before_docker(self) -> None:
        """Keep the verified IPC route and reject accidental transport changes."""
        result = subprocess.run(["bash", str(ROOT / "run_a3.sh"), "--transport", "hccl", "--dry-run"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("use --transport ipc", result.stderr)
