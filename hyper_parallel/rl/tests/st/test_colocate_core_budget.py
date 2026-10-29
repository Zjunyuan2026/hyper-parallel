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
"""Framework-free core budget including the new configuration safety controls."""
import ast
from pathlib import Path


def _method_lines(path: Path, names: tuple[str, ...]) -> int:
    """Count complete method bodies, including their pre-existing implementation."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return sum(node.end_lineno - node.lineno + 1 for node in ast.walk(tree)
               if isinstance(node, ast.FunctionDef) and node.name in names)


def test_colocate_core_budget():
    """Count physical lines without moving safety logic out of the measured core."""
    root = Path(__file__).resolve().parents[2] / "rl"
    counts = {}
    for path in (root / "colocate_async").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        ast.parse(source)
        counts[path.name] = len(source.splitlines())
    config = ast.parse((root / "config.py").read_text(encoding="utf-8"))
    nodes = [node for node in config.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
             and (node.name in ("ColocateAsyncConfig", "RolloutCorrectionConfig", "validate_colocate_async_config")
                  or node.name.startswith("_validate_colocate_"))]
    counts["config_controls"] = sum(node.end_lineno - node.lineno + 1 + len(node.decorator_list) for node in nodes)
    assert len(nodes) == 5, f"Expected all colocated configuration definitions, got {len(nodes)}"
    counts["rollout_correction.py"] = len((root / "algorithm/rollout_correction.py").read_text().splitlines())
    metrics = root / "utils/monitoring/colocate_async_metrics.py"
    counts[metrics.name] = len(metrics.read_text(encoding="utf-8").splitlines())
    shared = {
        "roles/policy/actor.py": ("_loss_is_finite", "_micro_batch_loss", "forward_backward",
                                  "_validate_update_experience"),
        "algorithm/loss.py": ("_actor_loss",),
        "dataset/contracts.py": ("_validate_correction_weights",),
    }
    for name, methods in shared.items():
        counts[name] = _method_lines(root / name, methods)
    loss_tree = ast.parse((root / "algorithm/loss.py").read_text())
    grpo = next(node for node in loss_tree.body if isinstance(node, ast.ClassDef) and node.name == "GRPOAlgorithm")
    bridge = next(node for node in grpo.body if isinstance(node, ast.FunctionDef) and node.name == "compute_actor_loss")
    counts["grpo_bridge_and_contract_fields"] = bridge.end_lineno - bridge.lineno + 1 + 4
    print(f"Colocated core physical lines: {sum(counts.values())}: {counts}")
    assert sum(counts.values()) <= 1000, f"Colocated core exceeds 1000 physical lines: {counts}"
