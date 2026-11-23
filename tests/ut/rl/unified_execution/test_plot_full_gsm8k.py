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
"""Check formal chart exports and rejection of incomplete acceptance evidence."""

import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[4] / "hyper_parallel/rl/examples/consistency/scripts"
with patch.object(sys, "path", [str(SCRIPTS), *sys.path]):
    SPEC = importlib.util.spec_from_file_location("formal_plot", SCRIPTS / "plot_full_gsm8k.py")
    PLOT = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(PLOT)


class FormalPlotTests(unittest.TestCase):
    """Eight charts require both numerical consistency and formal completion evidence."""

    def test_exports_all_figures_without_claiming_missing_test_results(self) -> None:
        """A complete short log cannot certify training coverage, validation and final test."""
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = {"step": 1, "training/rollout_log_probs_diff_max": 0,
                   "training/rollout_log_probs_diff_mean": 0, "training/log_probs_valid_tokens": 10,
                   "training/rollout_probs_diff_valid": 1, "training/actor_log_probs_mean": -1,
                   "training/rollout_log_probs_mean": -1, "perf/time_per_step": 300,
                   "train/total_loss": -0.1, "reward/mean": 0.5}
            (root / "steps.jsonl").write_text(json.dumps(row) + "\n")
            (root / "summary.json").write_text(json.dumps({"passed": True, "requested_steps": 1}))
            result = PLOT.render(root, root / "plots")
            self.assertFalse(result["passed"])
            self.assertEqual(len(list((root / "plots").glob("*.png"))), 8)
            self.assertEqual(len(list((root / "plots").glob("*.svg"))), 8)
            self.assertIn("No complete test evidence", (root / "plots/overview.svg").read_text(encoding="utf-8"))
