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
"""Bind the shared router and native rollout experts to stable graph layer names."""

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import torch

from rl.unified_execution.models.bindings import bind_expert_ownership, validate_expert_binding
from rl.unified_execution.runtimes.vllm.native_moe import NativeQwen3Experts

from hyper_parallel.distributed.recipe_spec import local_compute


@dataclass
class InferenceExpertExecutor:
    """Per-layer expert weights and ownership, registered before graph compilation."""

    experts: Any
    options: dict
    layer_name: str = ""
    native_experts: NativeQwen3Experts | None = None


def register_inference_experts(model: torch.nn.Module, parallel_context: Any, prefix: str = "") -> torch.nn.Module | None:
    """Bind EP graph calls to stable model paths after HP finishes construction.

    Args:
        model: Complete HP model whose EP factories have attached their executors.
        prefix: vLLM model namespace, shared with the attention registration.

    The key is embedded as a string constant in cached FX graphs. Registration
    order is not stable across processes, so an ordinal can silently select a
    different layer's same-shaped experts after an AOT cache reload.
    """
    # vLLM is optional in training and configuration processes.
    from vllm.config import get_current_vllm_config  # pylint: disable=C0415

    layers = get_current_vllm_config().compilation_config.static_forward_context
    native_experts = None
    for path, module in model.named_modules():
        executor = getattr(module, "unified_expert_executor", None)
        if executor is None:
            continue
        qualified_path = f"{prefix}.{path}" if prefix and path else prefix or path
        name = f"hyper_rl.ep.{qualified_path}"
        if executor.layer_name or name in layers:
            raise ValueError(f"Inference EP layer is already registered: {name}")
        if native_experts is None:
            native_experts = NativeQwen3Experts(get_current_vllm_config(), executor.options, parallel_context)
        executor.native_experts = native_experts
        executor.layer_name = name
        layers[name] = executor
    return native_experts


@torch.library.custom_op("hyper_rl::inference_ep", mutates_args=())
def execute_inference_ep(states: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor,
                          layer_name: str) -> torch.Tensor:
    """Run routing, expert GEMMs and slot return inside graph capture using live batch metadata."""
    # Resolve metadata at execution/capture time, never freeze it during Dynamo tracing.
    from vllm.forward_context import get_forward_context  # pylint: disable=C0415

    context = get_forward_context()
    executor = context.no_compile_layers[layer_name]
    return executor.native_experts(executor.experts, states, indices, weights, context)


@execute_inference_ep.register_fake
def _execute_inference_ep_fake(states: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor,
                                layer_name: str) -> torch.Tensor:
    del indices, weights, layer_name
    return torch.empty_like(states)


def _inference_forward(router_fn: Callable, executor: InferenceExpertExecutor,
                       module: Any, hidden_states: Any) -> Any:
    if not executor.layer_name:
        raise RuntimeError("Inference EP must be registered before forward")
    states = hidden_states.reshape(-1, hidden_states.shape[-1])
    indices, weights = router_fn(module, states)
    return execute_inference_ep(states, indices, weights, executor.layer_name).view(hidden_states.shape)


def bind_inference_ep(module: Any, ep_mesh: Any) -> Callable:
    """Register the executor under its stable path once HP finishes building the model."""
    router_fn, options = bind_expert_ownership(module, ep_mesh)
    executor = InferenceExpertExecutor(module.experts, options)
    module.unified_expert_executor = executor
    return partial(_inference_forward, router_fn, executor)


@local_compute
def inference_ep_compute_fn(*, module: Any, mesh: Any, tp_mesh: Any, cp_mesh: Any,
                            ep_mesh: Any) -> Callable:
    """Route on the training numerical contract and compute with the worker's native experts."""
    del mesh, tp_mesh, cp_mesh
    return bind_inference_ep(module, ep_mesh)


def validate_inference_ep(module: Any, function: Any, ep_size: int, path: str,
                          router_fn: Callable, *, native: NativeQwen3Experts) -> None:
    """Match the live graph registry, worker transport and HP expert ownership."""
    # These dependencies are needed only when auditing a realized inference model.
    from vllm.config import get_current_vllm_config  # pylint: disable=C0415

    from hyper_parallel.components.modules.grouped_experts import GroupedExperts  # pylint: disable=C0415

    if native is None:
        raise ValueError("Inference EP audit requires native expert kernels")
    executor = module.unified_expert_executor
    options = executor.options
    validate_expert_binding(module, function, ep_size, path, router_fn, _inference_forward, executor=executor)
    registered = get_current_vllm_config().compilation_config.static_forward_context.get(executor.layer_name)
    if (executor is not registered or executor.experts is not module.experts
            or executor.native_experts is not native or native.group is not options["group"]
            or native.local_count != options["local_experts"] or native.first != options["rank"] * native.local_count):
        raise ValueError("Inference EP registry or native expert ownership differs from the HP shard")
    experts = module.experts
    # Quantized subclasses own their projection math and cannot use the plain EIO GMM.
    if (type(experts) is not GroupedExperts  # pylint: disable=unidiomatic-typecheck
            or not experts.has_gate or experts.add_bias or experts.use_2d_experts):
        raise ValueError("Inference EP requires standard gated, bias-free HP GroupedExperts with 3D EIO weights")
