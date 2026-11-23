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
"""Training-role dynamic EP binding and execution-path validation."""

from functools import partial
from typing import Any, Callable

from hyper_parallel.distributed.recipe_spec import local_compute
from rl.unified_execution.models.bindings import bind_expert_ownership, validate_expert_binding
from rl.unified_execution.ops.moe.training_ep import training_ep


def _training_forward(router_fn: Callable, options: dict, module: Any, hidden_states: Any) -> Any:
    states = hidden_states.reshape(-1, hidden_states.shape[-1])
    indices, weights = router_fn(module, states)
    return training_ep(module, states, indices, weights, **options).view(hidden_states.shape)


def bind_training_ep(module: Any, ep_mesh: Any) -> Callable:
    """Bind routing and ownership once, including for Actor probability evaluation."""
    router_fn, options = bind_expert_ownership(module, ep_mesh)
    return partial(_training_forward, router_fn, options)


@local_compute
def unified_ep_compute_fn(*, module: Any, mesh: Any, tp_mesh: Any, cp_mesh: Any,
                          ep_mesh: Any) -> Callable:
    """Bind HP's differentiable expert routing for training and Actor scoring."""
    del mesh, tp_mesh, cp_mesh
    return bind_training_ep(module, ep_mesh)


def validate_training_ep(module: Any, function: Any, ep_size: int, path: str,
                         router_fn: Callable) -> None:
    """Require the differentiable transport even during Actor probability evaluation."""
    validate_expert_binding(module, function, ep_size, path, router_fn, _training_forward)
