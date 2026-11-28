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
"""CPU reporting integration: preserve evaluation metrics and export real figures."""

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[4]
EXAMPLES = ROOT / "hyper_parallel/rl/examples"


def load_script(name, path):
    """Import the device-free example utilities from their actual locations."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReportingTests(unittest.TestCase):
    def test_observer_merges_final_evaluation_without_overwriting_training(self):
        module = load_script("gsm8k64", EXAMPLES / "gsm8k/scripts/run_qwen3_moe_64npu.py")
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {"RANK": "0"}):
            directory = Path(temporary)
            tracker = Mock()
            observer = module.MetricsObserver(tracker, directory)
            observer.log({"validation/accuracy": 0.4}, step=0)
            observer.log({"reward/mean": 0.5}, step=1)
            observer.log({"validation/accuracy": 0.6}, step=1)
            rows = [json.loads(line) for line in (directory / "steps.jsonl").read_text().splitlines()]
            self.assertEqual(rows, [{"step": 1, "reward/mean": 0.5, "validation/accuracy": 0.6}])
            self.assertEqual(len((directory / "evaluations.jsonl").read_text().splitlines()), 2)
            self.assertEqual(tracker.log.call_count, 3)

    def test_four_comparison_figures_keep_recorded_logp_and_time_scope(self):
        plotting = EXAMPLES / "consistency/scripts"
        with patch.object(sys, "path", [str(plotting), *sys.path]):
            module = load_script("comparison64", plotting / "plot_comparison.py")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = []
            for mode in ("unified", "native"):
                run = root / mode
                directory = run / "node0/run"
                directory.mkdir(parents=True)
                config = yaml.safe_load((EXAMPLES / "gsm8k/configs" /
                    f"qwen3_30b_a3b_gsm8k_{mode}_64npu.yaml").read_text())
                (directory / "config.yaml").write_text(yaml.safe_dump(config))
                rows = []
                for step in range(1, 51):
                    row = {"step": step, "reward/mean": step / 100, "reward/std": 0.2,
                           "training/rollout_log_probs_diff_mean": 0 if mode == "unified" else 0.01,
                           "training/rollout_probs_diff_valid": 1, "training/log_probs_valid_tokens": 100,
                           "timing_s/gen": 10, "timing_s/update_actor": 20, "timing_s/weight_sync": 5}
                    if step % 10 == 0:
                        row.update({"validation/accuracy": 0.5, "validation/correct": 250,
                                    "validation/total": 500})
                    rows.append(row)
                (directory / "steps.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
                (run / "summary.json").write_text(json.dumps({"passed": True,
                    "elapsed_launch_to_collection_seconds": 2000, "scope": "test lifecycle"}))
                runs.append(run)
            output = root / "plots"
            report = module.render(*runs, output)
            self.assertEqual(report["missing_logp"], [])
            self.assertEqual(report["runs"]["NO-Consistency"]["seconds"]["Rollout"], 500)
            self.assertEqual(report["run_total_scope"]["Consistency"], "test lifecycle")
            self.assertIn("TP4 DP8 EP8 CP2", report["display_annotation"])
            self.assertEqual(len(list(output.glob("*.png"))), 4)
            self.assertEqual(len(list(output.glob("*.svg"))), 4)
            for path in output.glob("*.png"):
                with Image.open(path) as figure:
                    self.assertEqual(figure.size, (1080, 720))
