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
"""CPU-only four-node control and recipe contracts without device initialization."""

import ast
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import yaml

import rl.config.validate as validation

ROOT = Path(__file__).resolve().parents[4]


def load_function(name, bindings):
    path = ROOT / "hyper_parallel/rl/rl/weight_sync/sync.py"
    nodes = ast.parse(path.read_text(encoding="utf-8")).body
    node = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    future = ast.parse("from __future__ import annotations").body
    exec(compile(ast.fix_missing_locations(ast.Module(body=[*future, node], type_ignores=[])), str(path), "exec"),
         bindings)
    return bindings[name]


class MultiNodeTests(unittest.TestCase):
    def test_node_groups_cover_all_ranks_in_identical_order(self):
        for rank in range(64):
            dist = Mock()
            dist.get_rank.return_value = rank
            dist.get_world_size.return_value = 64
            dist.new_group.side_effect = lambda ranks: tuple(ranks)
            create = load_function("create_node_control_group", {"dist": dist})
            self.assertEqual(create(16), tuple(range(rank // 16 * 16, (rank // 16 + 1) * 16)))
            self.assertEqual([list(call.args[0]) for call in dist.new_group.call_args_list],
                             [list(range(start, start + 16)) for start in range(0, 64, 16)])

    def test_each_node_leader_controls_its_server_and_errors_remain_global(self):
        for start in (0, 16, 32, 48):
            for local_rank in (0, 3, 15):
                group = object()
                dist = Mock()
                dist.get_world_size.return_value = 64
                dist.get_rank.return_value = local_rank
                dist.get_global_rank.return_value = start
                error_sync = Mock()
                callback = Mock(return_value="ack")
                control = load_function("coordinator_call", {"dist": dist, "synchronize_error": error_sync})
                result = control("pause", callback, group=group)
                self.assertEqual(callback.call_count, int(local_rank == 0))
                self.assertEqual(result, "ack" if local_rank == 0 else None)
                self.assertEqual(dist.broadcast_object_list.call_args.kwargs, {"src": start, "group": group})
                error_sync.assert_called_once_with(None, "pause")

    def test_recipes_keep_global_batch_and_pr_arithmetic(self):
        configs = ROOT / "hyper_parallel/rl/examples/gsm8k/configs"
        for mode in ("native", "unified"):
            original = yaml.safe_load((configs / f"qwen3_30b_a3b_gsm8k_{mode}_16npu.yaml").read_text())
            actual = yaml.safe_load((configs / f"qwen3_30b_a3b_gsm8k_{mode}_64npu.yaml").read_text())
            train = actual["train"]
            self.assertEqual(train["accelerator"]["dp_shard"] * train["accelerator"]["tp"], 64)
            self.assertEqual(train["prompt_batch_size"] * 32, 128)
            self.assertEqual(train["response_mini_batch_size"], 32)
            self.assertEqual(train["max_steps"] * train["prompt_batch_size"] * 32, 6400)
            self.assertEqual(train["optimizer"], original["train"]["optimizer"])
            self.assertEqual(actual["algorithm"], original["algorithm"])
            self.assertEqual(train["plan_overrides"], original["train"]["plan_overrides"])
            self.assertTrue(actual["rollout"]["collect_log_probs"])
            self.assertTrue(train["checkpoint"]["save_final_model_only"])
            self.assertEqual(actual["rollout"]["vllm"]["max_num_seqs"], 32)
            self.assertEqual(actual["evaluation"]["batch_size"] * 32, 512)
            self.assertEqual(actual["rollout"]["vllm"]["compilation_config"]["cudagraph_mode"], "FULL_DECODE_ONLY")

    def test_node_leader_failure_reaches_global_error_synchronization(self):
        dist = Mock()
        dist.get_world_size.return_value = 64
        dist.get_rank.return_value = 0
        failure = RuntimeError("local engine unavailable")
        error_sync = Mock(side_effect=RuntimeError("global failure"))
        callback = Mock(side_effect=failure)
        control = load_function("coordinator_call", {"dist": dist, "synchronize_error": error_sync})
        with self.assertRaisesRegex(RuntimeError, "global failure"):
            control("wake", callback, group=object())
        error_sync.assert_called_once_with(failure, "wake")
        dist.broadcast_object_list.assert_not_called()

    def test_validator_accepts_local_rollout_and_rejects_invalid_local_world(self):
        configs = ROOT / "hyper_parallel/rl/examples/gsm8k/configs"
        for mode in ("native", "unified"):
            config = yaml.safe_load((configs / f"qwen3_30b_a3b_gsm8k_{mode}_64npu.yaml").read_text())
            model = SimpleNamespace(family="qwen3_moe")
            with patch.dict(os.environ, {"LOCAL_WORLD_SIZE": "16"}), patch.object(
                    validation.torch.accelerator, "current_accelerator", return_value=SimpleNamespace(type="npu")), \
                    patch.object(validation, "resolve_vllm_model", return_value=model):
                validation._validate_vllm(config["rollout"], config["train"]["accelerator"], model)
                validation._validate_checkpoint(config["train"]["checkpoint"])
                with patch.dict(os.environ, {"LOCAL_WORLD_SIZE": "3"}):
                    with self.assertRaisesRegex(ValueError, "LOCAL_WORLD_SIZE"):
                        validation._validate_vllm(config["rollout"], config["train"]["accelerator"], model)

    def test_admission_capacity_is_shared_only_by_local_request_owners(self):
        path = ROOT / "hyper_parallel/rl/rl/rollout/vllm.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == "VLLMGenerationEngine")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                      and node.name == "_local_child_capacity")
        bindings = {"os": os, "_VLLM_CHILD_CAPACITY_OVERSUBSCRIPTION": 2}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(path), "exec"),
             bindings)
        with patch.dict(os.environ, {"WORLD_SIZE": "64", "LOCAL_WORLD_SIZE": "16"}):
            for rank in range(32):
                engine = SimpleNamespace(_config={"max_num_seqs": 32, "data_parallel_size": 8},
                                         _deployment="colocated", _trainer_request_size=32,
                                         _trainer_request_rank=rank)
                self.assertEqual(bindings["_local_child_capacity"](engine), 64)
