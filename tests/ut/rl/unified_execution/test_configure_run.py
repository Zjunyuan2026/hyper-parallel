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
import runpy
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[4] / "hyper_parallel/rl/examples/consistency"
# This is a standalone configuration CLI; its contract needs no Torch installation.
CONFIGURE = runpy.run_path(str(ROOT / "scripts/configure_run.py"))["configure"]


class ConfigureRunTests(unittest.TestCase):
    """Keep task selection and token/transport budgets independent of topology."""

    def test_baseline_memory_override_preserves_training_and_cache_mode(self) -> None:
        """Both profiles accept explicit memory budgets without changing default 0.4."""
        common = {"RUN_MODEL_PATH": "/models/Qwen3-30B-A3B", "TRAIN_TP": "4", "TRAIN_CP": "2",
                  "TRAIN_EP": "4", "ROLLOUT_TP": "2", "ROLLOUT_PCP": "1", "ROLLOUT_EP": "16",
                  "ROLLOUT_LOCAL_WORLD_SIZE": "16", "GLOBAL_PROMPT_BATCH_SIZE": "128",
                  "MICRO_BATCH_SIZE": "2", "RESPONSE_MINI_BATCH_SIZE": "128", "CANDIDATES": "8",
                  "STEPS": "2", "MAX_PROMPT": "1024", "MAX_NEW_TOKENS": "1024", "MAX_MODEL_LEN": "2048"}
        for flag, filename in (("BASELINE_5STEP", "qwen3_30b_a3b_gsm8k_unified_5step.yaml"),
                               ("NATIVE_BASELINE", "qwen3_30b_a3b_gsm8k_native.yaml")):
            recipe = yaml.safe_load((ROOT.parent / "gsm8k/configs" / filename).read_text(encoding="utf-8"))
            with patch.dict(os.environ, {**common, flag: "true"}, clear=True):
                default = CONFIGURE(yaml.safe_load(yaml.safe_dump(recipe)), 64)
            for value in ("0.5", "0.35"):
                with self.subTest(flag=flag, value=value), patch.dict(
                    os.environ, {**common, flag: "true", "GPU_MEMORY_UTILIZATION": value}, clear=True,
                ):
                    actual = CONFIGURE(yaml.safe_load(yaml.safe_dump(recipe)), 64)
                    inference = actual["rollout"]["vllm"]
                    self.assertEqual(inference["gpu_memory_utilization"], float(value))
                    self.assertNotIn("kv_cache_memory_bytes", inference)
                    inference["gpu_memory_utilization"] = 0.4
                    self.assertEqual(actual, default)
            for value in ("0", "-0.1", "1.1", "nan", "inf"):
                with self.subTest(flag=flag, invalid=value), patch.dict(
                    os.environ, {**common, flag: "true", "GPU_MEMORY_UTILIZATION": value}, clear=True,
                ), self.assertRaisesRegex(ValueError, "GPU_MEMORY_UTILIZATION"):
                    CONFIGURE(yaml.safe_load(yaml.safe_dump(recipe)), 64)

    def test_native_baseline_removes_unified_execution(self) -> None:
        """Native runs use ordinary MoE training, no strict probability gate and fixed ordered data."""
        environment = {"RUN_MODEL_PATH": "/models/Qwen3-30B-A3B", "TRAIN_TP": "4", "TRAIN_CP": "2",
                       "TRAIN_EP": "4", "TRAIN_SP": "false", "ROLLOUT_TP": "2", "ROLLOUT_PCP": "1",
                       "ROLLOUT_EP": "8", "ROLLOUT_LOCAL_WORLD_SIZE": "16", "GLOBAL_PROMPT_BATCH_SIZE": "128",
                       "MICRO_BATCH_SIZE": "2", "RESPONSE_MINI_BATCH_SIZE": "128", "CANDIDATES": "8",
                       "NATIVE_BASELINE": "true", "STEPS": "2", "CPU_OFFLOAD": "true",
                       "MAX_PROMPT": "1024", "MAX_NEW_TOKENS": "1024", "MAX_MODEL_LEN": "2048",
                       "MAX_NUM_SEQS": "4", "KV_CACHE_MEMORY_BYTES": str(2 * 1024**3)}
        recipe = ROOT.parent / "gsm8k/configs/qwen3_30b_a3b_gsm8k_native.yaml"
        with patch.dict(os.environ, environment, clear=True):
            config = CONFIGURE(yaml.safe_load(recipe.read_text()), 64)
        self.assertNotIn("consistency", config)
        self.assertEqual(config["train"]["checkpoint"]["save_steps"], 0)
        self.assertFalse(config["train"]["checkpoint"]["save_final"])
        self.assertNotIn("expert_parallel_size", config["rollout"]["vllm"])
        self.assertNotIn("unified_execution", config["rollout"]["vllm"])
        self.assertEqual(config["rollout"]["vllm"]["model_implementation"], "native")
        self.assertTrue(config["rollout"]["collect_log_probs"])
        self.assertFalse(config["train"]["accelerator"]["cpu_offload"])
        self.assertFalse(config["train"]["accelerator"]["sequence_parallel"])
        self.assertEqual(config["train"]["micro_batch_size"], 2)
        self.assertEqual(config["train"]["optimizer"]["weight_decay"], 0.01)
        self.assertEqual(config["algorithm"]["kl_coef"], 0.001)
        self.assertEqual(config["data"]["max_train_samples"], 6400)
        self.assertFalse(config["data"]["shuffle"])
        self.assertNotIn("unified_execution", str(config["train"]["plan_overrides"]))
        self.assertEqual(config["rollout"]["vllm"]["max_num_batched_tokens"], 8192)
        self.assertEqual(config["rollout"]["vllm"]["max_num_seqs"], 64)
        self.assertEqual(config["rollout"]["vllm"]["gpu_memory_utilization"], 0.4)
        self.assertNotIn("kv_cache_memory_bytes", config["rollout"]["vllm"])
        self.assertEqual(config["rollout"]["vllm"]["compilation_config"]["cudagraph_capture_sizes"][-1], 64)
        with patch.dict(os.environ, {**environment, "STEPS": "50"}, clear=True):
            full = CONFIGURE(yaml.safe_load(recipe.read_text()), 64)
        self.assertEqual(full["train"]["max_steps"], 50)
        self.assertEqual(full["train"]["prompt_batch_size"] * 8 * 50, 6400)
        with patch.dict(os.environ, {**environment, "FORMAL_GSM8K": "true"}, clear=True), self.assertRaises(ValueError):
            CONFIGURE(yaml.safe_load(recipe.read_text()), 64)

    def test_baseline_pilot_keeps_independent_sampling_probabilities(self) -> None:
        """The 4B sampling protocol changes data, not the MoE arithmetic or probability gate."""
        environment = {"RUN_MODEL_PATH": "/models/Qwen3-30B-A3B", "TRAIN_TP": "4", "TRAIN_CP": "2",
                       "TRAIN_EP": "4", "ROLLOUT_TP": "2", "ROLLOUT_PCP": "1", "ROLLOUT_EP": "8",
                       "ROLLOUT_LOCAL_WORLD_SIZE": "16", "GLOBAL_PROMPT_BATCH_SIZE": "128",
                       "MICRO_BATCH_SIZE": "2", "RESPONSE_MINI_BATCH_SIZE": "128", "CANDIDATES": "8",
                       "BASELINE_5STEP": "true", "STEPS": "5", "CPU_OFFLOAD": "false",
                       "MAX_PROMPT": "1024", "MAX_NEW_TOKENS": "1024", "MAX_MODEL_LEN": "2048"}
        with patch.dict(os.environ, environment, clear=True):
            config = CONFIGURE(yaml.safe_load((ROOT / "configs/qwen3_moe_4l_cp_rl.yaml").read_text()), 64)
        self.assertEqual(config["data"]["max_train_samples"], 6400)
        self.assertFalse(config["data"]["shuffle"])
        self.assertEqual(config["evaluation"]["max_samples"], 500)
        self.assertEqual(config["train"]["micro_batch_size"], 2)
        self.assertEqual(config["train"]["prompt_batch_size"] * 8, 128)
        self.assertEqual(config["rollout"]["vllm"]["max_num_batched_tokens"], 8192)
        self.assertEqual(config["rollout"]["vllm"]["max_num_seqs"], 64)
        self.assertEqual(config["rollout"]["vllm"]["gpu_memory_utilization"], 0.4)
        self.assertNotIn("kv_cache_memory_bytes", config["rollout"]["vllm"])
        self.assertEqual(config["rollout"]["vllm"]["compilation_config"]["cudagraph_capture_sizes"][-1], 64)
        self.assertTrue(config["rollout"]["collect_log_probs"])
        self.assertEqual(config["train"]["old_log_prob_source"], "actor")
        self.assertEqual(config["consistency"]["mode"], "batch_invariant")
        self.assertEqual(config["train"]["accelerator"]["cp"], 2)
        self.assertEqual(config["rollout"]["vllm"]["expert_parallel_size"], 16)
        self.assertFalse(config["train"]["accelerator"]["reference_cpu_offload"])
        self.assertFalse(config["train"]["accelerator"]["offload_during_rollout"])
        self.assertEqual(config["train"]["checkpoint"]["save_steps"], 0)
        self.assertFalse(config["train"]["checkpoint"]["save_final"])
        self.assertEqual(config["algorithm"]["kl_coef"], 0.001)
        self.assertEqual(config["algorithm"]["kl_type"], "low_var_kl")
        self.assertEqual(config["train"]["optimizer"], {
            "lr": 1e-6, "lr_min": 0.0, "lr_decay_style": "constant", "lr_warmup_ratio": 0.0,
            "weight_decay": 0.01, "max_grad_norm": 1.0, "eps": 1e-8, "betas": [0.9, 0.999],
            "foreach": None, "fp32_main_params": True, "no_decay_params": [],
        })
        self.assertEqual(config["algorithm"], {
            "name": "grpo", "advantage_epsilon": 1e-6, "clip_ratio_low": 0.2, "clip_ratio_high": 0.2,
            "clip_ratio_c": 3.0, "kl_coef": 0.001, "kl_type": "low_var_kl", "loss_aggregation": "token-mean",
        })
        self.assertEqual(config["train"]["seed"], 42)
        self.assertTrue(config["train"]["gradient_sync_each_micro_batch"])
        self.assertIsNone(config["rollout"]["seed"])
        self.assertEqual(config["rollout"]["vllm"]["seed"], 17)
        with patch.dict(os.environ, {**environment, "STEPS": "50"}, clear=True):
            full = CONFIGURE(yaml.safe_load((ROOT / "configs/qwen3_moe_4l_cp_rl.yaml").read_text()), 64)
        self.assertEqual(full["train"]["max_steps"], 50)
        self.assertEqual(full["evaluation"]["every_n_steps"], 10)
        self.assertEqual(full["train"]["checkpoint"]["save_steps"], 0)
        self.assertFalse(full["train"]["checkpoint"]["save_final"])
        self.assertEqual(full["train"]["prompt_batch_size"] * 8 * 50, 6400)
        self.assertEqual(full["algorithm"], config["algorithm"])
        self.assertEqual(full["train"]["optimizer"], config["train"]["optimizer"])
        recipe = ROOT.parent / "gsm8k/configs/qwen3_30b_a3b_gsm8k_unified_5step.yaml"
        with patch.dict(os.environ, environment, clear=True):
            materialized = CONFIGURE(yaml.safe_load(recipe.read_text()), 64)
        for section in ("model", "data", "rollout", "train", "consistency", "evaluation", "algorithm"):
            self.assertEqual(materialized[section], config[section])
        with (patch.dict(os.environ, {**environment, "MICRO_BATCH_SIZE": "1"}, clear=True),
              self.assertRaises(ValueError)):
            CONFIGURE(yaml.safe_load((ROOT / "configs/qwen3_moe_4l_cp_rl.yaml").read_text()), 64)

    def test_formal_data_evaluation_and_checkpoint_settings(self) -> None:
        """Formal mode holds out validation and enables complete greedy evaluation and saves."""
        environment = {"RUN_MODEL_PATH": "/models/Qwen3-30B-A3B", "TRAIN_TP": "4", "TRAIN_CP": "2",
                       "TRAIN_EP": "4", "ROLLOUT_TP": "2", "ROLLOUT_PCP": "1", "ROLLOUT_EP": "8",
                       "ROLLOUT_LOCAL_WORLD_SIZE": "16", "GLOBAL_PROMPT_BATCH_SIZE": "136",
                       "MICRO_BATCH_SIZE": "1", "RESPONSE_MINI_BATCH_SIZE": "68", "CANDIDATES": "4",
                       "FORMAL_GSM8K": "true", "STEPS": "50", "CPU_OFFLOAD": "false",
                       "MAX_PROMPT": "512", "MAX_NEW_TOKENS": "1024", "MAX_MODEL_LEN": "1536"}
        with patch.dict(os.environ, environment, clear=True):
            config = CONFIGURE(yaml.safe_load((ROOT / "configs/qwen3_moe_4l_cp_rl.yaml").read_text()), 64)
        self.assertEqual(config["data"]["test_path"], "/formal-run/data/validation.parquet")
        self.assertEqual(config["evaluation"]["every_n_steps"], 10)
        self.assertEqual(config["data"]["max_prompt_length"], 512)
        self.assertEqual(config["rollout"]["max_new_tokens"], 1024)
        self.assertEqual(config["rollout"]["vllm"]["max_model_len"], 1536)
        self.assertIsNone(config["evaluation"]["max_samples"])
        self.assertFalse(config["evaluation"]["do_sample"])
        self.assertEqual(config["train"]["checkpoint"]["save_steps"], 10)
        self.assertTrue(config["train"]["checkpoint"]["save_final"])
        self.assertEqual(config["algorithm"]["kl_coef"], 0.0)
        self.assertEqual(config["train"]["optimizer"]["weight_decay"], 0.0)
        self.assertFalse(config["train"]["optimizer"].get("fp32_main_params", False))

    def test_target_batch_preflight_uses_complete_local_mini_batches(self) -> None:
        """Large prompt batches do not silently multiply the number of optimizer updates."""
        environment = {"RUN_MODEL_PATH": "/models/Qwen3-30B-A3B", "TRAIN_TP": "4", "TRAIN_CP": "2",
                       "TRAIN_EP": "4", "ROLLOUT_TP": "2", "ROLLOUT_PCP": "1", "ROLLOUT_EP": "8",
                       "ROLLOUT_LOCAL_WORLD_SIZE": "16", "GLOBAL_PROMPT_BATCH_SIZE": "136",
                       "MICRO_BATCH_SIZE": "1", "RESPONSE_MINI_BATCH_SIZE": "68", "CANDIDATES": "4",
                       "FULL_DATASET": "true", "STEPS": "5", "CPU_OFFLOAD": "false"}
        with patch.dict(os.environ, environment, clear=True):
            config = CONFIGURE(yaml.safe_load((ROOT / "configs/qwen3_moe_4l_cp_rl.yaml").read_text()), 64)
        train = config["train"]
        self.assertEqual((train["prompt_batch_size"], train["response_mini_batch_size"], train["micro_batch_size"]),
                         (17, 68, 1))
        self.assertEqual(train["max_steps"], 5)
        self.assertEqual(train["policy_update_epochs"], 1)
        self.assertIsNone(config["data"]["max_train_samples"])
        self.assertTrue(config["data"]["shuffle"])
        self.assertFalse(config["evaluation"]["enabled"])
        self.assertFalse(train["checkpoint"]["save_final"])

    def test_four_node_training_uses_global_dp_and_node_local_rollout_dp(self) -> None:
        """Full-model selection is independent of the small recipe filename."""
        environment = {"RUN_MODEL_PATH": "/models/Qwen3-30B-A3B", "TRAIN_TP": "4", "TRAIN_CP": "2",
                       "TRAIN_EP": "4", "ROLLOUT_TP": "2", "ROLLOUT_PCP": "1", "ROLLOUT_EP": "8",
                       "ROLLOUT_LOCAL_WORLD_SIZE": "16", "CPU_OFFLOAD": "false"}
        with patch.dict(os.environ, environment, clear=True):
            config = CONFIGURE(yaml.safe_load((ROOT / "configs/qwen3_moe_4l_cp_rl.yaml").read_text()), 64)
        accelerator = config["train"]["accelerator"]
        self.assertEqual((accelerator["dp_shard"], accelerator["edp_shard"]), (8, 16))
        self.assertEqual(config["rollout"]["vllm"]["data_parallel_size"], 8)
        self.assertEqual(config["rollout"]["vllm"]["node_local_world_size"], 16)
        self.assertEqual(config["model"]["registry_name"], "qwen3_30b_a3b")

    def test_device_residency_uses_standard_offload_switch(self) -> None:
        """One standard switch selects the existing device or CPU offload policy."""
        for enabled in (False, True):
            env = {"RUN_MODEL_PATH": "/models/Qwen3-30B-A3B", "CPU_OFFLOAD": str(enabled).lower()}
            with self.subTest(enabled=enabled), patch.dict(os.environ, env, clear=True):
                config = CONFIGURE(yaml.safe_load((ROOT / "configs/qwen3_moe_4l_cp_rl.yaml").read_text()), 16)
            self.assertEqual(config["train"]["accelerator"]["cpu_offload"], enabled)
            self.assertNotIn("experimental_moe_device_residency", config["train"]["accelerator"])

    def test_fixed_global_batch_across_tp_and_explicit_transport(self) -> None:
        """Changing TP preserves an explicitly selected global prompt workload."""
        for tp in (1, 2, 4):
            env = {"RUN_MODEL_PATH": "/models/Qwen3-30B-A3B", "TRAIN_TP": str(tp),
                   "GLOBAL_PROMPT_BATCH_SIZE": "16", "CANDIDATES": "4", "MICRO_BATCH_SIZE": "2",
                   "WEIGHT_SYNC_TRANSPORT": "ipc", "MAX_NUM_SEQS": "4", "WEIGHT_BUCKET_MB": "64"}
            with self.subTest(tp=tp), patch.dict(os.environ, env, clear=True):
                result = CONFIGURE(yaml.safe_load((ROOT / "configs/qwen3_moe_4l_rl.yaml").read_text()), 16)
                self.assertEqual(result["train"]["prompt_batch_size"] * (16 // tp), 16)
                self.assertEqual(result["train"]["micro_batch_size"], 2)
                self.assertNotIn("transport", result["rollout"]["vllm"]["weight_sync"])
                self.assertEqual(result["rollout"]["vllm"]["max_num_seqs"], 4)

    def test_invalid_batch_and_transport(self) -> None:
        """Reject uneven global work, invalid micro-batches and unknown transports."""
        for values in ({"GLOBAL_PROMPT_BATCH_SIZE": "3"}, {"MICRO_BATCH_SIZE": "3"},
                       {"WEIGHT_SYNC_TRANSPORT": "fallback"}):
            with self.subTest(values=values), patch.dict(
                os.environ, {"RUN_MODEL_PATH": "/models/full", **values}, clear=True,
            ):
                with self.assertRaises(ValueError):
                    CONFIGURE(yaml.safe_load((ROOT / "configs/qwen3_moe_4l_rl.yaml").read_text()), 16)

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

    def test_internal_lengths_and_native_experts_follow_launcher(self) -> None:
        """Direct generation honors length overrides and all selected devices."""
        env = {"RUN_MODEL_PATH": "/models/Qwen3-30B-A3B", "RUNNER": "internal", "STEPS": "5",
               "TRAIN_TP": "2", "TRAIN_EP": "8", "ROLLOUT_TP": "4", "ROLLOUT_EP": "4",
               "MAX_PROMPT": "768", "MAX_NEW_TOKENS": "2048", "MAX_MODEL_LEN": "4096",
               "PREFILL_TOKENS": "1024"}
        with patch.dict(os.environ, env, clear=True):
            config = CONFIGURE(yaml.safe_load((ROOT / "configs/qwen3_moe_4l_rl.yaml").read_text()), 16)
        self.assertEqual(config["train"]["max_steps"], 5)
        self.assertEqual(config["train"]["accelerator"]["dp_shard"], 8)
        self.assertEqual(config["train"]["accelerator"]["edp_shard"], 2)
        self.assertEqual(config["rollout"]["vllm"]["data_parallel_size"], 4)
        self.assertEqual(config["rollout"]["vllm"]["max_model_len"], 4096)
        self.assertEqual(config["data"]["max_prompt_length"], 768)
        self.assertEqual(config["rollout"]["max_new_tokens"], 2048)
        experts = [rule["replace"]["implementation"] for rule in config["train"]["plan_overrides"]
                   if rule.get("replace", {}).get("implementation", {}).get("_target_", "").endswith(
                       "unified_ep_compute_fn")]
        self.assertEqual(len(experts), 1)
        self.assertEqual(set(experts[0]), {"_target_"})

    def test_invalid_budgets_fail_before_model_startup(self) -> None:
        """Reject invalid context budgets and removed transport switches before model startup."""
        for overrides in ({"MAX_PROMPT": "512", "MAX_NEW_TOKENS": "1024", "MAX_MODEL_LEN": "1024"},
                          {"MC2_CAPACITY": "257"},
                          {"INFERENCE_EP_BACKEND": "unsupported"}, {"INFERENCE_EP_BACKEND": "allgather"},
                          {"EP_CAPACITY": "256"}, {"MC2_CAPACITY": "256"}):
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

    def test_cp_pcp_topology_and_global_batch(self) -> None:
        """CP peers share samples while PCP replicas each receive publication."""
        env = {"RUN_MODEL_PATH": "/models/full", "TRAIN_TP": "2", "TRAIN_CP": "2",
               "ROLLOUT_TP": "2", "ROLLOUT_PCP": "2", "TRAIN_EP": "2", "ROLLOUT_EP": "4",
               "GLOBAL_PROMPT_BATCH_SIZE": "8"}
        with patch.dict(os.environ, env, clear=True):
            config = CONFIGURE(yaml.safe_load((ROOT / "configs/qwen3_moe_4l_cp_rl.yaml").read_text()), 16)
        self.assertEqual(config["train"]["accelerator"]["cp"], 2)
        self.assertEqual(config["train"]["accelerator"]["activation_checkpoint"], "off")
        self.assertEqual(config["train"]["accelerator"]["dp_shard"], 4)
        self.assertEqual(config["train"]["prompt_batch_size"], 2)
        self.assertEqual(config["rollout"]["vllm"]["context_parallel_size"], 2)
        self.assertEqual(config["rollout"]["vllm"]["data_parallel_size"], 4)
        self.assertEqual(config["rollout"]["vllm"]["decode_context_parallel_size"], 1)
        implementations = [r.get("replace", {}).get("implementation", {}) for r in config["train"]["plan_overrides"]]
        self.assertTrue(all("inference_backend" not in item and "mc2_capacity" not in item
                            for item in implementations))

    def test_context_recipe_defaults_and_larger_batches(self) -> None:
        """Keep native recipe CP defaults without limiting tokens to an old transport capacity."""
        env = {"RUN_MODEL_PATH": "/models/full", "MAX_MODEL_LEN": "1024", "PREFILL_TOKENS": "512"}
        with patch.dict(os.environ, env, clear=True):
            config = CONFIGURE(yaml.safe_load((ROOT / "configs/qwen3_moe_4l_cp_rl.yaml").read_text()), 4)
        self.assertEqual(config["train"]["accelerator"]["tp"], 2)
        self.assertEqual(config["train"]["accelerator"]["cp"], 2)
        self.assertEqual(config["train"]["accelerator"]["activation_checkpoint"], "off")
        self.assertEqual(config["rollout"]["vllm"]["context_parallel_size"], 2)
        self.assertEqual(config["rollout"]["vllm"]["max_num_batched_tokens"], 512)

    def test_invalid_context_topology_or_missing_recipe(self) -> None:
        """Reject unsupported CP sizes and incomplete CP attention bindings."""
        for overrides in ({"TRAIN_CP": "3"}, {"TRAIN_CP": "4", "TRAIN_TP": "2"},
                          {"ROLLOUT_PCP": "3"}, {"TRAIN_CP": "2", "TRAIN_TP": "2"}):
            with self.subTest(overrides=overrides), patch.dict(
                os.environ, {"RUN_MODEL_PATH": "/models/full", **overrides}, clear=True,
            ):
                with self.assertRaises(ValueError):
                    CONFIGURE(yaml.safe_load((ROOT / "configs/qwen3_moe_4l_rl.yaml").read_text()), 4)


if __name__ == "__main__":
    unittest.main()
