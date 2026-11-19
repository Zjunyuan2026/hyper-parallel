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
"""Observe real optimizer changes without changing rewards, tokens or optimizer settings."""

import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


def parameter_samples(model: Any) -> dict[str, torch.Tensor]:
    """Retain bounded owned-weight probes; a changed value witnesses an actual update."""
    samples = {}
    for name, parameter in model.named_parameters():
        local = parameter.to_local() if hasattr(parameter, "to_local") else parameter
        values = local.detach().reshape(-1)
        if values.numel():
            stride = max(1, values.numel() // 1024)
            samples[name] = values[::stride][:1024].float().cpu().clone()
    return samples


class LearningObserver:
    """Observe updates and retain publication/next-step consistency evidence."""

    def __init__(self, trainer: Any, directory: Path, minimum_gradient: float,
                 *, require_first_update: bool = True) -> None:
        """Bind observation state to one ordinary trainer and evidence directory."""
        self.trainer = trainer
        self.directory = directory
        self.minimum_gradient = minimum_gradient
        self.require_first_update = require_first_update
        self.records: list[dict[str, Any]] = []

    def install(self) -> None:
        """Wrap ordinary update and publication calls, preserving their return values."""
        update = self.trainer.actor.update
        publish = self.trainer.publication.publish

        def checked_update(experience):
            step = self.trainer.state.global_step + 1
            before = parameter_samples(self.trainer.actor.actor_model)
            advantages = experience.advantages
            mask = experience.action_mask[:, 1:].bool()
            valid = advantages[mask]
            finite_advantages = bool(torch.isfinite(valid).all())
            nonzero = bool(valid.numel() and finite_advantages and torch.any(valid != 0))
            metrics = update(experience)
            after = parameter_samples(self.trainer.actor.actor_model)
            changed_names = [name for name, value in before.items() if not torch.equal(value, after[name])]
            gradient = float(metrics.gradient_norm)
            evidence = torch.tensor(
                [int(nonzero), int(bool(changed_names)),
                 int(math.isfinite(gradient) and gradient > self.minimum_gradient),
                 int(math.isfinite(gradient) and gradient >= 0 and finite_advantages)],
                dtype=torch.int64, device=self.trainer.device)
            dist.all_reduce(evidence, op=dist.ReduceOp.SUM)
            record = {
                "step": step, "rank": dist.get_rank(), "gradient_norm": gradient,
                "local_weights_changed": bool(changed_names), "changed_parameter_names": changed_names,
                "ranks_with_nonzero_advantages": int(evidence[0]),
                "ranks_with_weight_changes": int(evidence[1]),
                "ranks_with_positive_finite_gradient": int(evidence[2]),
                "ranks_with_finite_update_evidence": int(evidence[3]),
                "publication_completed": False, "next_step_logp_verified": False,
                "weight_probe": "up to 1024 values per owned parameter; positive change witness",
            }
            self.records.append(record)
            self.save(record)
            if int(evidence[3]) != dist.get_world_size():
                raise RuntimeError("Nonfinite real learning check failed; see learning-step*-rank*.json")
            if (self.require_first_update and step == 1
                    and (not bool(evidence[0]) or not bool(evidence[1])
                         or int(evidence[2]) != dist.get_world_size())):
                raise RuntimeError("First-step real learning check failed; see learning-step1-rank*.json")
            return metrics

        def checked_publish(version):
            result = publish(version)
            record = self.records[-1]
            if record["step"] != version:
                raise RuntimeError("Published policy version differs from observed optimizer step")
            record["publication_completed"] = True
            record["published_policy_version"] = version
            self.save(record)
            return result

        self.trainer.actor.update = checked_update
        self.trainer.publication.publish = checked_publish

    def verified_probabilities(self, step: int) -> None:
        """Mark only an already published previous update after the strict logp gate passes."""
        if self.records and self.records[-1]["step"] == step - 1:
            record = self.records[-1]
            if not record["publication_completed"]:
                raise RuntimeError("Next-step probabilities preceded completed weight publication")
            record["next_step_logp_verified"] = True
            self.save(record)

    def save(self, record: dict[str, Any]) -> None:
        """Preserve rank-local evidence even if a later gate fails."""
        path = self.directory / f'learning-step{record["step"]}-rank{record["rank"]}.json'
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
