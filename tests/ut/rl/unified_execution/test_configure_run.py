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
"""Check portable GSM8K configuration without importing the device runtime."""

import os
from pathlib import Path
import runpy
import unittest
from unittest.mock import patch

import yaml


ROOT = Path(__file__).resolve().parents[4] / "hyper_parallel/rl/examples/consistency"
# This is a standalone configuration CLI; its contract needs no Torch installation.
CONFIGURE = runpy.run_path(str(ROOT / "scripts/configure_run.py"))["configure"]


class ConfigureRunTests(unittest.TestCase):
    """Keep task selection and token/transport budgets independent of topology."""

    def test_single_turn_removes_external_harness_settings(self) -> None:
        """A reused Codex recipe must not retain a harness or tool parser."""
        config = yaml.safe_load((ROOT / "configs/qwen3_moe_4l_rl.yaml").read_text())
        config["agentic"] = {"runner": "codex", "codex": {"session_root": "/old"}}
        with patch.dict(os.environ, {"RUN_MODEL_PATH": "/models/Qwen3-30B-A3B"}, clear=True):
            result = CONFIGURE(config, 16)
        self.assertEqual(result["agentic"]["runner"], "internal")
        self.assertEqual(result["agentic"]["interaction_mode"], "single_turn")
        self.assertEqual(result["agentic"]["max_turns"], 1)
        self.assertNotIn("codex", result["agentic"])
        self.assertFalse(result["rollout"]["vllm"]["enable_auto_tool_choice"])
        self.assertIsNone(result["rollout"]["vllm"]["tool_call_parser"])
        self.assertEqual(result["data"]["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(result["model"]["registry_name"], "qwen3_30b_a3b")

    def test_internal_lengths_and_ep_capacity_follow_launcher(self) -> None:
        """Direct generation honors length overrides and all selected devices."""
        env = {"RUN_MODEL_PATH": "/models/Qwen3-30B-A3B", "RUNNER": "internal", "STEPS": "5",
               "TRAIN_TP": "2", "TRAIN_EP": "8", "ROLLOUT_TP": "4", "ROLLOUT_EP": "4",
               "MAX_PROMPT": "768", "MAX_NEW_TOKENS": "2048", "MAX_MODEL_LEN": "4096",
               "EP_CAPACITY": "4096", "PREFILL_TOKENS": "1024"}
        with patch.dict(os.environ, env, clear=True):
            config = CONFIGURE(yaml.safe_load((ROOT / "configs/qwen3_moe_4l_rl.yaml").read_text()), 16)
        self.assertEqual(config["train"]["max_steps"], 5)
        self.assertEqual(config["train"]["accelerator"]["dp_shard"], 8)
        self.assertEqual(config["train"]["accelerator"]["edp_shard"], 2)
        self.assertEqual(config["rollout"]["vllm"]["data_parallel_size"], 4)
        self.assertEqual(config["rollout"]["vllm"]["max_model_len"], 4096)
        self.assertEqual(config["data"]["max_prompt_length"], 768)
        self.assertEqual(config["rollout"]["max_new_tokens"], 2048)
        capacities = [rule["replace"]["implementation"]["capacity"] for rule in config["train"]["plan_overrides"]
                      if rule.get("replace", {}).get("implementation", {}).get("_target_", "").endswith(
                          "fixed_capacity_ep_compute_fn")]
        self.assertEqual(capacities, [4096])

    def test_invalid_budgets_fail_before_model_startup(self) -> None:
        """Reject requests that exceed context or the fixed EP transport capacity."""
        for overrides in ({"MAX_PROMPT": "512", "MAX_NEW_TOKENS": "1024", "MAX_MODEL_LEN": "1024"},
                          {"EP_CAPACITY": "2048", "MAX_MODEL_LEN": "4096"},
                          {"EP_CAPACITY": "4096", "PREFILL_TOKENS": "8192"}):
            with self.subTest(overrides=overrides):
                with patch.dict(os.environ, {"RUN_MODEL_PATH": "/models/full", **overrides}, clear=True):
                    with self.assertRaises(ValueError):
                        CONFIGURE(yaml.safe_load((ROOT / "configs/qwen3_moe_4l_rl.yaml").read_text()), 16)

    def test_codex_requires_explicit_selection(self) -> None:
        """Preserve the prior external-harness recipe when explicitly requested."""
        with patch.dict(os.environ, {"RUN_MODEL_PATH": "/models/full", "RUNNER": "codex"}, clear=True):
            config = CONFIGURE(yaml.safe_load((ROOT / "configs/qwen3_moe_4l_rl.yaml").read_text()), 16)
        self.assertEqual(config["agentic"]["runner"], "codex")
        self.assertIn("codex", config["agentic"])
        self.assertTrue(config["rollout"]["vllm"]["enable_auto_tool_choice"])


if __name__ == "__main__":
    unittest.main()
