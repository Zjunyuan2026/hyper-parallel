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
"""Validate comparison figures against synthetic CPU metrics, never real experiment claims."""

import json
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from matplotlib.image import imread

SCRIPTS = Path(__file__).resolve().parents[4] / "hyper_parallel/rl/examples/consistency/scripts"
with patch.object(sys, "path", [str(SCRIPTS), *sys.path]):
    RENDER = runpy.run_path(str(SCRIPTS / "plot_comparison.py"))["render"]


def fixture(root: Path, native: bool) -> None:
    """Write complete 50-step fixtures with independently checkable time sums."""
    directory = root / "node0/run"
    directory.mkdir(parents=True)
    (root / "summary.json").write_text(json.dumps({"passed": True,
                                                  "elapsed_launch_to_collection_seconds": 1200}), encoding="utf-8")
    rows = []
    for step in range(1, 51):
        row = {"step": step, "reward/mean": 0.5, "reward/std": 0.1,
               "timing_s/step": 12, "timing_s/gen": 1, "timing_s/old_log_prob": 2,
               "timing_s/ref": 2, "timing_s/update_actor": 5, "timing_s/weight_sync": 1}
        if not native:
            row.update({"training/rollout_log_probs_diff_max": 0,
                        "training/rollout_probs_diff_valid": 1, "training/log_probs_valid_tokens": 100})
        rows.append(row)
    (directory / "steps.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    evaluations = [{"step": step, "validation/accuracy": (250 + step) / 500,
                    "validation/correct": 250 + step, "validation/total": 500} for step in range(0, 51, 10)]
    (directory / "evaluations.jsonl").write_text("\n".join(json.dumps(r) for r in evaluations), encoding="utf-8")


class ComparisonPlotTests(unittest.TestCase):
    """Keep absent probabilities distinct from real zero and use full-run time sums."""

    def test_four_panels_and_missing_native_logp(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root / "consistency", False)
            fixture(root / "native", True)
            output = root / "plots"
            report = RENDER(root / "consistency", root / "native", output)
            self.assertEqual(len(list(output.glob("*.png"))), 4)
            self.assertEqual(len(list(output.glob("*.svg"))), 4)
            for path in output.glob("*.png"):
                pixels = imread(path)
                self.assertAlmostEqual(pixels.shape[1] / pixels.shape[0], 1.5)
            self.assertEqual(report["missing_logp"], [{"label": "NO-Consistency", "steps": list(range(1, 51))}])
            self.assertEqual(report["runs"]["Consistency"]["seconds"]["Run total"], 1200)
            self.assertEqual(report["runs"]["Consistency"]["seconds"]["Training steps"], 600)
            self.assertEqual(report["runs"]["Consistency"]["seconds"]["Actor update"], 250)
            svg = (output / "test-accuracy-comparison.svg").read_text(encoding="utf-8")
            self.assertIn("60.00%", svg)
            self.assertIn("300/500", svg)
            (root / "native/summary.json").write_text('{"passed":false}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "successful 50-step"):
                RENDER(root / "consistency", root / "native", output)
