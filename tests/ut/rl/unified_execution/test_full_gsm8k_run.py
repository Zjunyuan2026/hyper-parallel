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
"""CPU regressions for complete-source sampling, all-different layouts and honest training plots."""

import json
import math
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import yaml

from hyper_parallel.rl.examples.consistency.scripts import configure_run, plot_training

ROOT = Path(__file__).resolve().parents[4]
RECIPE = ROOT / "hyper_parallel/rl/examples/consistency/configs/qwen3_moe_4l_cp_rl.yaml"


def metrics(step: int) -> dict:
    """Return explicitly synthetic complete test evidence, not a reported training result."""
    return {"step": step, "reward/mean": 0.5, "reward/std": 0.5, "train/total_loss": -0.01 * step,
            "training/rollout_probs_diff_valid": 1, "training/log_probs_valid_tokens": 32,
            "training/actor_log_probs_mean": -1, "training/rollout_log_probs_mean": -1,
            "training/rollout_log_probs_diff_max": 0, "training/rollout_log_probs_diff_mean": 0}


class FullSourceRunTests(unittest.TestCase):
    """Keep all source rows available and derive the requested sixteen-device topology."""

    def test_all_five_parallel_settings_differ_and_full_data_is_not_capped(self) -> None:
        """Use400 prompt rows over50 steps without filtering the source down to400 rows."""
        settings = {"RUN_MODEL_PATH": "/models/full", "FULL_DATASET": "true", "STEPS": "50",
                    "TRAIN_TP": "4", "TRAIN_CP": "2", "TRAIN_EP": "4", "TRAIN_SP": "true",
                    "ROLLOUT_TP": "2", "ROLLOUT_PCP": "1", "ROLLOUT_EP": "8", "ROLLOUT_SP": "false",
                    "PROMPTS_PER_RANK": "4", "CANDIDATES": "4", "RESPONSE_MINI_BATCH_SIZE": "16"}
        with patch.dict(os.environ, settings, clear=True):
            config = configure_run.configure(yaml.safe_load(RECIPE.read_text()), 16)
        train, rollout = config["train"]["accelerator"], config["rollout"]["vllm"]
        for left, right in ((train["tp"], rollout["tensor_parallel_size"]),
                            (train["dp_shard"], rollout["data_parallel_size"]),
                            (train["ep"], rollout["expert_parallel_size"]),
                            (train["cp"], rollout["context_parallel_size"]),
                            (train["sequence_parallel"], rollout["sequence_parallel"])):
            self.assertNotEqual(left, right)
        self.assertIsNone(config["data"]["max_train_samples"])
        self.assertTrue(config["data"]["shuffle"])
        self.assertEqual(config["data"]["train_path"], "/data/gsm8k/train.parquet")
        self.assertEqual(config["train"]["prompt_batch_size"] * train["dp_shard"] * 50, 400)
        self.assertEqual(config["train"]["response_mini_batch_size"], 16)
        self.assertEqual(config["train"]["policy_update_epochs"], 1)
        self.assertFalse(config["train"]["learning_gate"]["enabled"])

    def test_full_source_rejects_inherited_two_question_filter(self) -> None:
        """Never call a filtered two-question run full-data training."""
        with patch.dict(os.environ, {"RUN_MODEL_PATH": "/models/full", "FULL_DATASET": "true",
                                     "LEARNING_PROMPT_ROWS": "4483,310"}, clear=True):
            with self.assertRaisesRegex(ValueError, "LEARNING_PROMPT_ROWS"):
                configure_run.configure(yaml.safe_load(RECIPE.read_text()), 16)

    def test_preset_overrides_old_small_batch_and_can_lower_workload(self) -> None:
        """Avoid inheriting the prior one-prompt/two-candidate smoke-test workload."""
        env = {**os.environ, "PROMPTS_PER_RANK": "1", "CANDIDATES": "2", "STEPS": "2",
               "GLOBAL_PROMPT_BATCH_SIZE": "2", "LEARNING_PROMPT_ROWS": "4483,310",
               "DEVICES": ",".join(str(i) for i in range(16))}
        for arguments, prompts, mini in (([], 4, 16), (["--prompts", "2"], 2, 8)):
            result = subprocess.run(["bash", str(ROOT / "run_a3_gsm8k50.sh"), "--dry-run", *arguments],
                                    env=env, text=True, capture_output=True, check=True)
            for expected in ("STEPS=50", "CANDIDATES=4", "FULL_DATASET=true", "TRACK_UPDATES=true",
                             "PLOT_TRAINING=true", "TRAIN_TP=4", "ROLLOUT_TP=2",
                             f"PROMPTS_PER_RANK={prompts}", f"RESPONSE_MINI_BATCH_SIZE={mini}"):
                self.assertIn(expected + "\n", result.stdout)
            self.assertNotIn("GLOBAL_PROMPT_BATCH_SIZE=", result.stdout)

    def test_preset_rejects_an_unexpected_device_count(self) -> None:
        """The advertised DP2/DP8 layout requires all sixteen selected devices."""
        result = subprocess.run(["bash", str(ROOT / "run_a3_gsm8k50.sh"), "--dry-run"],
                                env={**os.environ, "DEVICES": "0,1,2,3"},
                                text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 2)


class TrainingPlotTests(unittest.TestCase):
    """Missing steps and missing probabilities must not look like a successful zero-error run."""

    def test_missing_values_remain_missing(self) -> None:
        """Keep absent/null/non-numeric metrics distinct from measured zero."""
        for row in ({}, {"value": None}, {"value": "invalid"}):
            self.assertTrue(math.isnan(plot_training.number(row, "value")))
        self.assertEqual(plot_training.number({"value": 0}, "value"), 0)

    def test_complete_fifty_step_prefix_and_partial_run_are_distinct(self) -> None:
        """Exercise50-step report logic without claiming a real50-step device run."""
        with TemporaryDirectory() as directory:
            path = Path(directory)
            rows = [metrics(step) for step in range(1, 51)]
            (path / "summary.json").write_text(json.dumps({"passed": True, "requested_steps": 50}))
            self.assertTrue(plot_training.run_status(path, rows)["accepted"])
            partial = plot_training.run_status(path, rows[:3])
            self.assertFalse(partial["completed_run"])
            self.assertTrue(partial["strict_zero_for_all_recorded_steps"])
            self.assertIn("PARTIAL", partial["status"])

    def test_missing_nonfinite_or_nonzero_evidence_is_rejected(self) -> None:
        """An existing passed flag cannot override invalid metric evidence."""
        with TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "summary.json").write_text(json.dumps({"passed": True, "requested_steps": 1}))
            for key, value in (("training/log_probs_valid_tokens", float("inf")),
                               ("training/rollout_log_probs_diff_max", 1e-9),
                               ("training/actor_log_probs_mean", float("nan")),
                               ("training/rollout_probs_diff_valid", None)):
                row = metrics(1)
                row[key] = value
                result = plot_training.run_status(path, [row])
                self.assertFalse(result["accepted"])
                self.assertEqual(result["invalid_or_nonzero_steps"], [1])

    def test_corrupt_or_noncontiguous_metric_file_is_rejected(self) -> None:
        """Do not silently skip an interrupted JSON row or fill absent steps with zeros."""
        with TemporaryDirectory() as directory:
            path = Path(directory) / "steps.jsonl"
            for text in ("", '{"step":', json.dumps(metrics(2))):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    plot_training.read_rows(path)

    def test_png_svg_export_marks_partial_run(self) -> None:
        """Render actual files with a conspicuous incomplete-run label."""
        with TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "steps.jsonl").write_text("\n".join(json.dumps(metrics(i)) for i in range(1, 4)) + "\n")
            (path / "summary.json").write_text(json.dumps({"passed": False, "requested_steps": 50}))
            output = path / "plots"
            result = plot_training.render(path, output)
            self.assertFalse(result["accepted"])
            for name in ("reward", "loss", "consistency", "training-curves"):
                self.assertTrue((output / f"{name}.png").read_bytes().startswith(b"\x89PNG"))
                self.assertIn("PARTIAL", (output / f"{name}.svg").read_text())
