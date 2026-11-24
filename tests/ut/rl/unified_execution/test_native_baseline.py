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
"""CPU native control-path and CP coordinate tests without creating process groups."""

import ast
import json
import os
import re
import sys
import tempfile
import time
import unittest
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import yaml
from torch.nn import functional

ROOT = Path(__file__).resolve().parents[4]


def load_function(path: str, name: str, bindings: dict):
    """Compile the actual function with mocked optional runtime dependencies."""
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    function = next(node for node in tree.body
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name)
    future = ast.parse("from __future__ import annotations").body
    module = ast.fix_missing_locations(ast.Module(body=[*future, function], type_ignores=[]))
    exec(compile(module, path, "exec"), bindings)
    return bindings[name]


class NativeBaselineTests(unittest.TestCase):
    """Verify native admission and reporting without weakening unified probability gates."""

    def test_native_expert_reload_accepts_flat_and_routed_storage(self) -> None:
        """Publish canonical projections across buckets while checking local expert coverage."""
        adapter = "hyper_parallel/rl/rl/weight_sync/model_adapter.py"
        bindings = {}
        for name in ("_direct_tensor_description", "_owned_experts", "_native_expert_descriptions"):
            load_function(adapter, name, bindings)
        describe = load_function(adapter, "_native_qwen3_tensors", bindings)
        config = SimpleNamespace(model_type="qwen3_moe", vocab_size=8, num_experts=2,
                                 hidden_size=3, moe_intermediate_size=4)
        reload_module = SimpleNamespace(initialize_layerwise_reload=Mock(), finalize_layerwise_reload=Mock())
        for nested in (False, True):
            for ep in (1, 2):
                with self.subTest(nested=nested, ep=ep):
                    prefix = "model.layers.0.mlp.experts"
                    storage = prefix + (".routed_experts" if nested else "")
                    parallel = SimpleNamespace(ep_size=ep, tp_size=1, tp_rank=0)
                    expert_map = torch.tensor([-1, 0]) if ep == 2 else None
                    layer = (SimpleNamespace(moe_config=parallel, expert_map=expert_map) if nested else
                             SimpleNamespace(ep_size=ep, tp_size=1, tp_rank=0, expert_map=expert_map))
                    count = 2 // ep
                    parameters = {storage + ".w13_weight": torch.zeros(count, 3, 8),
                                  storage + ".w2_weight": torch.zeros(count, 4, 3)}
                    model = SimpleNamespace(named_modules=lambda: [(storage, layer)],
                                            named_parameters=lambda: list(parameters.items()))
                    descriptions = describe(model, config, tp_rank=0, tp_size=2)
                    owned = range(2) if ep == 1 else (1,)
                    expected_names = {f"{prefix}.{expert}.{proj}_proj.weight" for expert in owned
                                      for proj in ("gate", "up", "down")}
                    self.assertEqual({row["name"] for row in descriptions}, expected_names)
                    self.assertEqual({row["destination_name"] for row in descriptions}, set(parameters))
                    # The loader owns mapping canonical checkpoint names into native runtime storage.
                    def load_weights(weights):
                        name, value = weights[0]
                        match = re.fullmatch(re.escape(prefix) + r"\.(\d+)\.(gate|up|down)_proj\.weight", name)
                        self.assertIsNotNone(match)
                        expert, projection = int(match[1]), match[2]
                        local = expert if ep == 1 else int(expert_map[expert])
                        self.assertGreaterEqual(local, 0)
                        if projection == "down":
                            parameters[storage + ".w2_weight"][local].copy_(value.T)
                        else:
                            offset = 0 if projection == "gate" else 4
                            parameters[storage + ".w13_weight"][local, :, offset:offset + 4].copy_(value.T)
                        return {name}

                    model.load_weights = load_weights
                    worker = SimpleNamespace(device="cpu", model_config=SimpleNamespace(hf_config=config),
                                             model_runner=SimpleNamespace(get_model=lambda: model))
                    runtime = {"torch": torch, "re": re,
                               "_native_weight_layout": lambda _worker: {"tensors": descriptions}}
                    path = "hyper_parallel/rl/rl/weight_sync/vllm_worker.py"
                    receive = load_function(path, "_load_moe_checkpoint_weights", runtime)
                    finish = load_function(path, "_finalize_moe_checkpoint_reload", runtime)
                    source = torch.arange(12, dtype=torch.float32).reshape(4, 3)
                    all_weights = [(f"{prefix}.{expert}.{proj}_proj.weight",
                                    (source + expert * 20 + offset).T if proj == "down"
                                    else source + expert * 20 + offset)
                                   for expert in range(2) for proj, offset in (("gate", 0), ("up", 1), ("down", 2))]
                    before = reload_module.finalize_layerwise_reload.call_count
                    with patch.dict(sys.modules, {"vllm.model_executor.model_loader.reload": reload_module}):
                        receive(worker, model, all_weights[:-1])
                        with self.assertRaisesRegex(RuntimeError, "incomplete"):
                            finish(worker)
                        self.assertEqual(reload_module.finalize_layerwise_reload.call_count, before)
                        receive(worker, model, all_weights[-1:])
                        self.assertEqual(worker._rl_received_weights, expected_names)
                        with patch.object(torch, "npu", SimpleNamespace(current_stream=lambda: Mock()), create=True):
                            finish(worker)
                        self.assertEqual(reload_module.finalize_layerwise_reload.call_count, before + 1)
                        with self.assertRaisesRegex(ValueError, "Duplicate"):
                            receive(worker, model, all_weights[-1:])
                        with self.assertRaisesRegex(ValueError, "Unexpected"):
                            receive(worker, model, [(f"{prefix}.2.gate_proj.weight", source)])
                    for expert in owned:
                        local = expert if ep == 1 else 0
                        torch.testing.assert_close(parameters[storage + ".w13_weight"][local, :, :4],
                                                   (source + expert * 20).T)
                        torch.testing.assert_close(parameters[storage + ".w13_weight"][local, :, 4:],
                                                   (source + expert * 20 + 1).T)
                        torch.testing.assert_close(parameters[storage + ".w2_weight"][local],
                                                   source + expert * 20 + 2)

    def test_native_cp_masks_boundary_targets_and_gradients(self) -> None:
        """Real CPU tensors verify both shards against the global next-token computation."""
        bindings = {"torch": torch, "functional": functional, "DataLoaderParallelContext": SimpleNamespace}
        load_function("hyper_parallel/data/parallel/batch_parallel.py", "CPBatchSharder", bindings)
        path = "hyper_parallel/rl/rl/workers/adapters/context_parallel.py"
        bind = load_function(path, "bind_native_context_parallel", bindings)
        compute = load_function(path, "compute_cp_token_log_probs", bindings)
        prepare_mask = load_function(
            "hyper_parallel/models/qwen3_moe/adapter/distributed/context_parallel.py",
            "_prepare_qwen3_moe_flash_attention_cp_mask", {"torch": torch})
        sequences = torch.tensor([[0, 0, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]])
        padding_mask = sequences.ne(0)
        weights = torch.arange(64, dtype=torch.float32).reshape(8, 8).div(16).requires_grad_()
        reference_weights = weights.detach().clone().requires_grad_()
        calls = []

        class Model:
            config = SimpleNamespace(model_type="qwen3_moe")

            def __call__(self, **kwargs):
                calls.append(kwargs)
                return {"logits": weights[kwargs["input_ids"]]}

        model = Model()
        parts = []
        for rank in range(2):
            mesh = SimpleNamespace(size=lambda: 2, get_local_rank=lambda rank=rank: rank)
            bind(model, mesh)
            self.assertIs(model.unified_cp_mesh, mesh)
            parts.append(compute(model, sequences, padding_mask, mesh))
            call = calls[-1]
            torch.testing.assert_close(call["input_ids"], sequences[:, rank * 4:(rank + 1) * 4])
            torch.testing.assert_close(call["position_ids"], torch.arange(rank * 4, (rank + 1) * 4).view(1, 4))
            self.assertFalse(call["use_cache"])
            mask = call["attention_mask"]
            self.assertEqual(mask.shape, (2, 1, 4, 8))
            expected = ((torch.arange(8).view(1, 1, 1, 8)
                         <= torch.arange(rank * 4, (rank + 1) * 4).view(1, 1, 4, 1))
                        & padding_mask.view(2, 1, 1, 8))
            torch.testing.assert_close(mask, expected)
            self.assertIs(prepare_mask(mask, 4, 8, rank * 4, sequences.device, True), mask)
        actual = torch.cat(parts, dim=1)[:, :-1]
        expected = reference_weights[sequences].log_softmax(-1)[:, :-1].gather(
            -1, sequences[:, 1:].unsqueeze(-1)).squeeze(-1)
        torch.testing.assert_close(actual, expected)
        # Query 3 on rank 0 must predict token 4 on rank 1, with gradients retained.
        torch.testing.assert_close(parts[0][:, -1], expected[:, 3])
        actual.sum().backward()
        expected.sum().backward()
        torch.testing.assert_close(weights.grad, reference_weights.grad)
        with self.assertRaisesRegex(ValueError, "CP2"):
            bind(model, SimpleNamespace(size=lambda: 1))
        model.config = SimpleNamespace(model_type="other")
        with self.assertRaisesRegex(ValueError, "Qwen3 MoE"):
            bind(model, SimpleNamespace(size=lambda: 2))

    def test_actual_native_cp_contract_requires_complete_attention_recipe(self) -> None:
        """Exercise the exact contract that failed before build_trainer on A3."""
        path = "hyper_parallel/rl/rl/unified_execution/core/contract.py"
        bindings = {"Mapping": Mapping}
        load_function(path, "_mapping", bindings)
        load_function(path, "native_moe_cp_enabled", bindings)
        validate = load_function(path, "validate_unified_config", bindings)
        recipe = ROOT / "hyper_parallel/rl/examples/gsm8k/configs/qwen3_30b_a3b_gsm8k_native.yaml"
        config = yaml.safe_load(recipe.read_text())
        validate(config)
        self.assertEqual(config["train"]["accelerator"]["cp"], 2)
        self.assertNotIn("consistency", config)
        missing = deepcopy(config)
        missing["train"]["plan_overrides"] = [rule for rule in config["train"]["plan_overrides"]
                                               if rule.get("when") != "cp"]
        with self.assertRaisesRegex(ValueError, "cp=1"):
            validate(missing)
        wrong = deepcopy(config)
        wrong["model"]["name"] = "qwen3_5"
        with self.assertRaisesRegex(ValueError, "cp=1"):
            validate(wrong)
        wrong = deepcopy(config)
        wrong["rollout"]["vllm"]["context_parallel_size"] = 2
        with self.assertRaisesRegex(ValueError, "PCP"):
            validate(wrong)

    def test_node_local_native_ep_allows_device_residency(self) -> None:
        """Native resident mode requires node-local expert parallelism; other guards remain enforced."""
        bindings = {
            "torch": SimpleNamespace(accelerator=SimpleNamespace(
                current_accelerator=lambda: SimpleNamespace(type="npu"))),
            "optional_mapping": lambda mapping, key: mapping.get(key, {}),
            "_validate_vllm_basics": lambda _config: ("colocated", 8, 2),
            "_trainer_world_size": lambda _config: 64,
            "_validate_colocated_vllm": Mock(), "_validate_disjoint_vllm": Mock(),
            "_validate_vllm_limits": Mock(), "_validate_dense_parallelism": Mock(),
            "resolve_vllm_model": lambda model, _implementation: model,
            "resolve_weight_sync_config": Mock(),
        }
        validate = load_function("hyper_parallel/rl/rl/config/validate.py", "_validate_vllm", bindings)
        rollout = {"vllm": {"node_local_world_size": 16, "model_implementation": "native",
                            "enable_expert_parallel": True}}
        model = SimpleNamespace(family="qwen3_moe")
        validate(rollout, {"cpu_offload": True, "tp": 4}, model)
        bindings["_validate_colocated_vllm"].assert_called_once()
        validate(rollout, {"cpu_offload": False, "tp": 4}, model)
        rollout["vllm"]["enable_expert_parallel"] = False
        with self.assertRaisesRegex(ValueError, "device-resident"):
            validate(rollout, {"cpu_offload": False, "tp": 4}, model)
        with self.assertRaisesRegex(ValueError, "Node-local"):
            validate(rollout, {"cpu_offload": True}, SimpleNamespace(family="qwen3"))
        rollout["vllm"]["model_implementation"] = "unified"
        validate(rollout, {"cpu_offload": False, "tp": 4}, model)

    def test_native_training_report_does_not_assert_probability_parity(self) -> None:
        """A completed ordinary run is distinct from a strict consistency pass."""
        class Observer:
            def __init__(self, _tracker, _directory):
                self.rows = []

        class Recorder:
            def __init__(self, _directory):
                pass

            def call(self, _phase, callback):
                return callback()

        trainer = SimpleNamespace(train_dataloader=SimpleNamespace(dataset=[None] * 6400),
                                  state=SimpleNamespace(global_step=0), _tracker=None)

        def train():
            trainer.state.global_step = 2
            trainer._tracker.rows = [{"training/rollout_log_probs_diff_max": 1.0}] * 2

        trainer.train = train
        run = load_function("hyper_parallel/rl/examples/gsm8k/scripts/run_qwen3_30b_native.py", "run", {
            "json": json, "time": time, "os": os, "PhaseRecorder": Recorder,
            "MetricsObserver": Observer, "build_trainer": lambda _config: trainer,
        })
        config = {"rollout": {"vllm": {"model_implementation": "native"}}, "train": {"max_steps": 2}}
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {"RANK": "0"}):
            report = Path(temporary) / "native-training.json"
            run(config, report)
            result = json.loads(report.read_text())
            self.assertTrue(result["passed"])
            self.assertFalse(result["strict_consistency"])
            self.assertEqual(result["completed_steps"], 2)
            with self.assertRaisesRegex(ValueError, "no consistency"):
                run({**config, "consistency": {}}, report)
            trainer.train = lambda: None
            trainer.state.global_step = 1
            with self.assertRaises(RuntimeError):
                run(config, report)
