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
"""Validate recipe admission before expensive model or process initialization."""
import unittest

from rl.async_training.protocol import AsyncTrainingConfig, validate_async_config


def recipe():
    """Return the smallest supported logical recipe."""
    return {
        "model": {"name": "qwen3"}, "algorithm": {"name": "grpo"},
        "rollout": {"engine": "vllm", "num_return_sequences": 2, "temperature": 1.,
                    "top_p": 1., "top_k": 0, "vllm": {"deployment": "disjoint"}},
        "agentic": {"max_turns": 1, "interaction_mode": "single_turn"},
        "train": {"micro_batch_size": 1, "response_mini_batch_size": 2,
                  "accelerator": {"dp_shard": 1}},
    }


class TestAsyncConfig(unittest.TestCase):
    """Reject unsupported combinations rather than silently overriding user settings."""

    def test_supported_recipe(self):
        self.assertEqual(validate_async_config(recipe()).groups_per_dp_rank, 2)

    def test_unsupported_algorithms_and_devices(self):
        for section, key, value in (
            ("algorithm", "name", "ppo"), ("model", "name", "qwen3_moe"),
            ("agentic", "max_turns", 2), ("rollout", "temperature", 0.8),
            ("rollout", "top_p", 0.9), ("rollout", "num_return_sequences", 1),
        ):
            config = recipe()
            config[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                validate_async_config(config)

    def test_colocated_is_rejected(self):
        config = recipe()
        config["rollout"]["vllm"]["deployment"] = "colocated"
        with self.assertRaises(ValueError):
            validate_async_config(config)

    def test_capacity_configuration_is_strict(self):
        for settings in ({"max_buffer_groups": True}, {"operation_timeout": float("nan")},
                         {"groups_per_dp_rank": 0}, {"publish_every_updates": -1},
                         {"unknown": 1}, {"max_buffer_bytes": 1}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                AsyncTrainingConfig.from_mapping(settings)

    def test_zero_microbatch_is_rejected_without_division(self):
        config = recipe()
        config["train"]["micro_batch_size"] = 0
        with self.assertRaisesRegex(ValueError, "positive integers"):
            validate_async_config(config)

    def test_local_update_plan_must_have_equal_shapes(self):
        config = recipe()
        config["train"]["response_mini_batch_size"] = 3
        with self.assertRaises(ValueError):
            validate_async_config(config)
