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
"""Role binding, selection and gradient checks independent of NPU hardware."""

import subprocess
import sys
import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rl.unified_execution.core.execution_plan import UNIFIED_EP_TARGET, has_unified_ep
from rl.unified_execution.ops.moe.training_ep import training_ep
from rl.unified_execution.runtimes.hyper_parallel.builder import build_inference_model
from rl.unified_execution.runtimes.vllm.moe import (
    bind_inference_ep,
    register_inference_experts,
    validate_inference_ep,
)

from hyper_parallel.components.modules.grouped_experts import GroupedExperts


class TestUnifiedEP(unittest.TestCase):
    """Reject transport drift and compare dynamic routing with independent derivatives."""

    def test_inference_builder_binds_role_without_mutating_shared_plan(self) -> None:
        """Worker adaptation must not change the serialized Actor plan or EP1 model."""
        entry = {"match": "*.mlp.forward", "when": "ep", "replace": {"kind": "method",
                 "implementation": {"_target_": UNIFIED_EP_TARGET}}}
        model_class = Mock()
        context = SimpleNamespace(ep_size=2, mesh_context=object())
        scope = "rl.unified_execution.runtimes.hyper_parallel.builder"
        with patch(scope + ".resolve_runtime_symbol", return_value=model_class), \
                patch(scope + ".normalized_compute_setup") as setup:
            build_inference_model(SimpleNamespace(model_type="qwen3_moe"), [entry], parallel_context=context)
        implementation = setup.call_args.args[0][0]["replace"]["implementation"]
        self.assertTrue(implementation["_target_"].endswith(".inference_ep_compute_fn"))
        self.assertEqual(set(implementation), {"_target_"})
        self.assertEqual(entry["replace"]["implementation"]["_target_"], UNIFIED_EP_TARGET)
        context.ep_size = 1
        with patch(scope + ".resolve_runtime_symbol", return_value=model_class), \
                patch(scope + ".normalized_compute_setup") as setup:
            build_inference_model(SimpleNamespace(model_type="qwen3_moe"), [entry], parallel_context=context)
        self.assertEqual(setup.call_args.args[0], [entry])

    def test_training_import_does_not_load_inference_runtime(self) -> None:
        """Training/config imports must not register vLLM custom ops or compile entrypoints."""
        subprocess.run([sys.executable, "-c", (
            "import sys; import rl.unified_execution.runtimes.hyper_parallel.moe; "
            "assert not any(n.startswith('rl.unified_execution.runtimes.vllm') for n in sys.modules); "
            "assert 'vllm' not in sys.modules"
        )], check=True, timeout=60)

    def test_inference_audit_rejects_registry_and_ownership_drift(self) -> None:
        """The object reached by the graph must retain the audited experts and limits."""
        group = Mock()
        group.size.return_value, group.rank.return_value = 2, 0
        source = torch.nn.Module()
        source.num_experts, source.hidden_size, source.intermediate_size = 2, 4, 3
        source.config = SimpleNamespace(hidden_act="silu")
        source.gate_up_proj = torch.nn.Parameter(torch.zeros(2, 6, 4))
        source.down_proj = torch.nn.Parameter(torch.zeros(2, 4, 3))
        experts = GroupedExperts(module=source)
        experts.local_expert_count = 2
        experts._ep_use_grouped_gemm = True  # pylint: disable=protected-access
        module = torch.nn.Module()
        module.experts = experts
        options = {"group": group, "rank": 0, "world_size": 2, "local_experts": 2}
        router = Mock()
        registry = {}
        config = SimpleNamespace(compilation_config=SimpleNamespace(static_forward_context=registry))
        fake_config = SimpleNamespace(get_current_vllm_config=lambda: config)
        native = SimpleNamespace(group=group, local_count=2, first=0)
        scope = "rl.unified_execution.runtimes.vllm.moe"
        with patch.dict("sys.modules", {"vllm.config": fake_config}), \
                patch(scope + ".NativeQwen3Experts", return_value=native), \
                patch(scope + ".bind_expert_ownership", return_value=(router, options)):
            compute = bind_inference_ep(module, object())
            register_inference_experts(module, object(), "mlp")
            validate_inference_ep(module, partial(compute, module), 2, "mlp", router, native=native)
            with patch.object(experts, "add_bias", True), self.assertRaisesRegex(ValueError, "bias-free"):
                validate_inference_ep(module, partial(compute, module), 2, "mlp", router, native=native)
            key = next(iter(registry))
            registry[key] = object()
            with self.assertRaisesRegex(ValueError, "expert ownership differs"):
                validate_inference_ep(module, partial(compute, module), 2, "mlp", router, native=native)
            registry[key] = module.unified_expert_executor
            options["rank"] = 1
            with self.assertRaisesRegex(ValueError, "expert ownership differs"):
                validate_inference_ep(module, partial(compute, module), 2, "mlp", router, native=native)

    def test_cached_graph_keeps_expert_layer_across_registration_orders(self) -> None:
        """A saved layer call must reach that layer in a newly constructed worker."""
        registry = {}
        config = SimpleNamespace(compilation_config=SimpleNamespace(static_forward_context=registry))
        fake_config = SimpleNamespace(get_current_vllm_config=lambda: config)
        fake_context = SimpleNamespace(get_forward_context=lambda: SimpleNamespace(no_compile_layers=registry))
        scope = "rl.unified_execution.runtimes.vllm.moe"
        saved_graphs = {}
        names = ("layer0", "layer1", "layer2")

        def expert_forward(experts, states, indices, weights, context):
            del indices, weights, context
            return states + experts.marker

        with patch.dict("sys.modules", {"vllm.config": fake_config, "vllm.forward_context": fake_context}), \
                patch(scope + ".bind_expert_ownership", return_value=(Mock(), {})), \
                patch(scope + ".NativeQwen3Experts", return_value=expert_forward):
            for run, order in enumerate((names, tuple(reversed(names)))):
                registry.clear()
                # Other native layers can register first without changing EP identity.
                registry.update({f"unrelated{index}": object() for index in range(run + 1)})
                model = torch.nn.Module()
                for name in order:
                    module = torch.nn.Module()
                    module.experts = SimpleNamespace(marker=10 * run + names.index(name) + 1)
                    model.add_module(name, module)
                    bind_inference_ep(module, object())
                register_inference_experts(model, object(), "model")

                for name in names:
                    key = model.get_submodule(name).unified_expert_executor.layer_name
                    self.assertEqual(key, f"hyper_rl.ep.model.{name}")
                    if run == 0:
                        graph = torch.fx.Graph()
                        inputs = [graph.placeholder(arg) for arg in ("states", "indices", "weights")]
                        graph.output(graph.call_function(torch.ops.hyper_rl.inference_ep.default, (*inputs, key)))
                        saved_graphs[name] = torch.fx.GraphModule(torch.nn.Module(), graph)

                    # Reuse the first worker's graph without retracing or rewriting its key.
                    states = torch.ones(2, 3)
                    actual = saved_graphs[name](states, torch.zeros(2, 1, dtype=torch.int64), torch.ones(2, 1))
                    expected = states + 10 * run + names.index(name) + 1
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_dynamic_inverse_permutation_and_gradients(self) -> None:
        """A deliberately shuffled return must restore token slots before FP32 weighting."""
        states = torch.tensor([[2., 3.], [4., 5.]], requires_grad=True)
        ids = torch.tensor([[0, 1], [1, 0]])
        weights = torch.tensor([[.123456, .876544], [.234567, .765433]], requires_grad=True)
        order = torch.tensor([3, 0, 2, 1])
        routed = states.repeat_interleave(2, 0) * (ids.flatten()[:, None] + 1)
        dispatch = SimpleNamespace(states=states, expert_indices=ids, send_counts=[2, 2],
                                   receive_counts=[2, 2], dispatch_order=order)
        with patch("rl.unified_execution.ops.moe.training_ep._prepare_ep_dispatch", return_value=dispatch), \
                patch("rl.unified_execution.ops.moe.training_ep._run_ep_local_experts", return_value=routed[order]):
            actual = training_ep(object(), states, ids, weights, group=object(), rank=0,
                                world_size=2, local_experts=1)
        expected = states * ((ids + 1) * weights).sum(-1, keepdim=True)
        torch.testing.assert_close(actual, expected)
        grads = torch.autograd.grad(actual.sum(), (states, weights), retain_graph=True)
        reference = torch.autograd.grad(expected.sum(), (states, weights))
        for value, target in zip(grads, reference):
            torch.testing.assert_close(value, target)

    def test_recipe_requires_one_expert_binding(self) -> None:
        entry = {"match": "*.mlp.forward", "when": "ep", "replace": {"kind": "method",
                 "implementation": {"_target_": UNIFIED_EP_TARGET}}}
        self.assertTrue(has_unified_ep([entry]))
        self.assertFalse(has_unified_ep([]))
        with self.assertRaisesRegex(ValueError, "unambiguous"):
            has_unified_ep([entry, entry])
        entry["replace"]["implementation"] = UNIFIED_EP_TARGET
        self.assertTrue(has_unified_ep([entry]))
