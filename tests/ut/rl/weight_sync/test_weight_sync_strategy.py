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
"""CPU unit tests for Hyper-RL weight-sync strategy selection."""

import pytest

import rl.weight_sync.transfer as transfer_module
from rl.workers.engine.model import ModelRegistration, resolve_vllm_model
import rl.weight_sync.transfer as weight_sync
from rl.weight_sync.config import resolve_weight_sync_config
from rl.weight_sync.transfer import build_weight_publisher
from rl.weight_sync.colocated_hccl import ColocatedHCCLWeightTransport


def _rollout_model():
    """Return one supported Hyper Qwen3 rollout registration."""
    model = ModelRegistration(
        "qwen",
        "qwen3",
        "/model",
        "/tokenizer",
        "Qwen3ForCausalLM",
        "qwen3",
        "qwen3",
        True,
    )
    return resolve_vllm_model(model, "native")


@pytest.mark.parametrize("deployment, transport_name", [("colocated", "ipc"), ("disjoint", "hccl")])
@pytest.mark.parametrize("strategy", ["full_gather"])
@pytest.mark.parametrize("tp_size", [1, 2])
def test_weight_sync_preserves_explicit_strategy_for_tp1_and_tp2(deployment, transport_name, strategy, tp_size):
    """All combinations reuse the publisher while preserving the explicit strategy."""
    result = build_weight_publisher(
        deployment, _rollout_model(), tensor_parallel_size=tp_size, data_parallel_size=2,
        bucket_size_bytes=64, strategy=strategy,
    )
    assert result.configured_strategy == strategy
    assert result.transport.name == transport_name
    assert result.source.bucket_size_bytes == 64


def test_only_current_publisher_interface_is_exposed() -> None:
    """Removed combination classes and publication verbs have no compatibility surface."""
    removed_names = (
        "ColocatedDirectReshardWeightTransfer",
        "ColocatedFullGatherWeightTransfer",
        "DirectReshardHCCLWeightTransfer",
        "FallbackWeightTransfer",
        "FullGatherHCCLWeightTransfer",
        "WeightTransfer",
    )
    assert all(not hasattr(weight_sync, name) for name in removed_names)
    assert all(not hasattr(transfer_module, name) for name in removed_names)
    publisher = build_weight_publisher("colocated", _rollout_model())
    assert callable(publisher.publish)
    assert not hasattr(publisher, "transfer")
    assert not hasattr(publisher, "refit")


def test_removed_fallback_configuration_is_rejected() -> None:
    """The removed fallback field cannot silently alter current behavior."""
    with pytest.raises(ValueError, match="fallback_strategy"):
        resolve_weight_sync_config(
            {"strategy": "direct_reshard", "fallback_strategy": "full_gather"},
            deployment="colocated",
            model_family="qwen3",
            rollout_tp=1,
        )


@pytest.mark.parametrize("transport_name", ["ipc"])
def test_colocated_explicit_transport(transport_name):
    """An explicit transport never silently falls back to the deployment default."""
    config = resolve_weight_sync_config(
        {"transport": transport_name}, deployment="colocated", model_family="qwen3", rollout_tp=4,
    )
    result = build_weight_publisher("colocated", _rollout_model(), transport=config.transport)
    assert result.transport.name == transport_name, f"expected={transport_name}, actual={result.transport.name}"


@pytest.mark.parametrize("transport_name", ["invalid", "ipc"])
def test_reject_invalid_disjoint_transport(transport_name):
    """Unknown names and cross-device IPC fail before runtime allocation."""
    with pytest.raises(ValueError):
        resolve_weight_sync_config(
            {"transport": transport_name}, deployment="disjoint", model_family="qwen3", rollout_tp=4,
        )


def test_colocated_hccl_uses_device_unique_routes():
    """Colocated HCCL uses the dedicated route planner, not the disjoint group."""
    config = resolve_weight_sync_config(
        {"transport": "hccl"}, deployment="colocated", model_family="qwen3", rollout_tp=4,
    )
    result = build_weight_publisher("colocated", _rollout_model(), transport=config.transport)
    assert isinstance(result.transport, ColocatedHCCLWeightTransport), f"actual={type(result.transport)}"


def test_disjoint_explicit_hccl_preserved():
    """The existing separate-device HCCL route remains available."""
    config = resolve_weight_sync_config(
        {"transport": "hccl"}, deployment="disjoint", model_family="qwen3", rollout_tp=4,
    )
    result = build_weight_publisher("disjoint", _rollout_model(), transport=config.transport)
    assert result.transport.name == "hccl", f"actual={result.transport.name}"
