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
import struct
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from xml.etree import ElementTree

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
            sys.modules["plot_training"].render(root, root / "ordinary")
            for folder, expected in ((root / "plots", 8), (root / "ordinary", 6)):
                self.assertEqual(len(list(folder.glob("*.png"))), expected)
                self.assertEqual(len(list(folder.glob("*.svg"))), expected)
                for path in folder.glob("*.png"):
                    width, height = struct.unpack(">II", path.read_bytes()[16:24])
                    self.assertEqual(width * 4, height * 6)
                for path in folder.glob("*.svg"):
                    attributes = ElementTree.parse(path).getroot().attrib
                    self.assertAlmostEqual(float(attributes["width"][:-2]) / float(attributes["height"][:-2]), 1.5)
