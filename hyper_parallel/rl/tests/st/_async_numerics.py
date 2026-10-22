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
"""Real DP Actor parameter-gradient parity against a scalar finite-difference oracle."""
import json
import math
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from rl.algorithm import GRPOAlgorithm, GRPOConfig
from rl.dataset.batch_builder import ExperiencePreparer, build_experience_batch
from rl.dataset.contracts import Trajectory, Turn
from rl.roles.policy.actor import Actor
from rl.roles.rollout.base import GenerationSettings


CONFIG = {"clip_ratio_low": 0.2, "clip_ratio_high": 0.2, "clip_ratio_c": 3.0,
          "kl_coef": 0.07, "advantage_epsilon": 1.e-6}


def objective(table, rows):
    """Compute full-batch loss without Torch loss, log-softmax or target helpers."""
    total, tokens = 0., 0
    for sequence, positions, reward, group_rewards, behavior in rows:
        mean = sum(group_rewards) / len(group_rewards)
        deviation = math.sqrt(sum((value - mean) ** 2 for value in group_rewards) / (len(group_rewards) - 1))
        advantage = (reward - mean) / (deviation + CONFIG["advantage_epsilon"])
        for column in positions:
            context, action = sequence[column - 1], sequence[column]
            logits = table[context]
            current = logits[action] - math.log(sum(math.exp(value) for value in logits))
            old = behavior[context][action] - math.log(sum(math.exp(value) for value in behavior[context]))
            reference = INITIAL[context][action] - math.log(sum(math.exp(value) for value in INITIAL[context]))
            ratio = math.exp(current - old)
            loss = max(-advantage * ratio, -advantage * min(1.2, max(0.8, ratio)))
            if advantage < 0:
                loss = min(loss, -3. * advantage)
            delta = reference - current
            total += loss + .07 * min(10., max(0., math.exp(delta) - delta - 1.))
            tokens += 1
    return total / tokens


INITIAL = [[(row * 8 + column) / 200. - .2 for column in range(8)] for row in range(8)]


class TablePolicy(torch.nn.Module):
    """Use a small parameter table so every parameter can be checked independently."""

    def __init__(self, device):
        super().__init__()
        self.table = torch.nn.Parameter(torch.tensor(INITIAL, dtype=torch.float32, device=device))

    def forward(self, input_ids, attention_mask, use_cache=False):
        del attention_mask, use_cache
        return {"logits": self.table[input_ids]}


class RecordingSGD(torch.optim.SGD):
    """Retain exactly the reduced, pre-step parameter gradient used by the real optimizer."""

    def step(self, closure=None):
        self.observed = [parameter.grad.detach().cpu().clone()
                         for group in self.param_groups for parameter in group["params"]]
        return super().step(closure)


def fixture(rank):
    """Use complete two-candidate groups with unequal token counts across DP ranks."""
    trajectories, reference_rows, frozen_reference = [], [], []
    for index in range(4):
        version = 1 + index // 2
        shift = .25 if version == 1 else -.4
        behavior = [[value + shift * ((column % 3) - 1) for column, value in enumerate(row)]
                    for row in INITIAL]
        sequence = [1 + rank, 2 + index, 4, 5, 6]
        positions = [1, 2, 3] if (rank + index) % 2 == 0 else [1]
        if rank == 1 and index == 3:
            positions = [1, 2]
        reward = float(index % 2)
        action_mask = torch.tensor([column in positions for column in range(5)])
        old, reference = [], []
        for context, action in zip(sequence[:-1], sequence[1:]):
            old.append(behavior[context][action] - math.log(sum(math.exp(v) for v in behavior[context])))
            reference.append(INITIAL[context][action] - math.log(sum(math.exp(v) for v in INITIAL[context])))
        trajectories.append(Trajectory(
            trajectory_id=f"{rank}:{index}", prompt_id=f"{rank}:{index // 2}", group_id=f"{rank}:{index // 2}",
            policy_version=version, worker_policy_version=version,
            turns=(Turn("user", "q", 0, 1, False), Turn("assistant", "a", 1, 5, True)),
            token_ids=torch.tensor(sequence), attention_mask=torch.ones(5, dtype=torch.bool),
            action_mask=action_mask, rollout_log_probs=torch.tensor(old),
            reward=reward, reward_components={"task": reward}, done=True, truncated=False,
            terminal_reason="completed",
        ))
        reference_rows.append((sequence, positions, reward, [0., 1.], behavior))
        frozen_reference.append(reference)
    return trajectories, reference_rows, frozen_reference


def test_parameter_gradient():
    """Run CPU/Gloo by default; explicit NPU mode exercises two real HCCL ranks."""
    backend = os.environ.get("ASYNC_NUMERIC_BACKEND", "gloo")
    device = torch.device("cpu")
    if backend == "hccl":
        # The optional NPU extension must register its backend only inside the hardware worker.
        import torch_npu  # pylint: disable=import-outside-toplevel,unused-import
        device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
        torch.npu.set_device(device)
    dist.init_process_group(backend)
    try:
        rank = dist.get_rank()
        model = TablePolicy(device)
        distributed = DistributedDataParallel(model, device_ids=[device.index] if device.type == "npu" else None)
        optimizer = RecordingSGD(distributed.parameters(), lr=.03)
        algorithm = GRPOAlgorithm(GRPOConfig(kl_coef=.07))
        trajectories, local_rows, reference = fixture(rank)
        batch = build_experience_batch(
            tuple(trajectories), .1, GenerationSettings(
                max_new_tokens=4, temperature=1., top_p=1., top_k=0, do_sample=True,
                pad_token_id=0, eos_token_id=7, collect_log_probs=True), {},
            mixed_policy_versions=True,
        )
        fields = {name: getattr(batch, name).to(device) for name in
                  ("sequences", "attention_mask", "action_mask", "rewards", "old_log_probs", "rollout_log_probs")}
        batch = replace(batch, **fields)
        prepared = ExperiencePreparer(algorithm).prepare(
            batch, reference_log_probs=torch.tensor(reference, device=device),
        )
        actor = Actor(distributed, algorithm, 1, optimizer=optimizer, device=device,
                      dp_group_info=SimpleNamespace(group=dist.group.WORLD), dp_size=dist.get_world_size(),
                      response_mini_batch_size=4, update_epochs=1, max_grad_norm=0.)
        metrics = actor.update(prepared)
        all_rows = [None] * dist.get_world_size()
        dist.all_gather_object(all_rows, local_rows)
        combined = [row for rows in all_rows for row in rows]
        expected = torch.zeros(8, 8, dtype=torch.float64)
        epsilon = 1.e-5
        for row in range(8):
            for column in range(8):
                plus, minus = [values[:] for values in INITIAL], [values[:] for values in INITIAL]
                plus[row][column] += epsilon
                minus[row][column] -= epsilon
                expected[row, column] = (objective(plus, combined) - objective(minus, combined)) / (2 * epsilon)
        observed = optimizer.observed[0].double()
        torch.testing.assert_close(observed, expected, atol=2.e-7, rtol=2.e-5)
        expected_parameters = torch.tensor(INITIAL, dtype=torch.float64) - .03 * expected
        torch.testing.assert_close(model.table.detach().cpu().double(), expected_parameters, atol=1.e-7, rtol=2.e-5)
        if rank == 0:
            result = {"backend": backend, "world_size": dist.get_world_size(), "parameters_checked": 64,
                      "max_abs_gradient_error": float((observed - expected).abs().max()),
                      "gradient_atol": 2.e-7, "gradient_rtol": 2.e-5,
                      "global_action_tokens": sum(len(row[1]) for row in combined),
                      "local_action_tokens": [sum(len(row[1]) for row in rows) for rows in all_rows],
                      "optimizer_steps": metrics.optimizer_steps}
            destination = Path(os.environ.get("ASYNC_NUMERIC_OUTPUT", "async_parameter_numerics.json"))
            destination.write_text(json.dumps(result, indent=2))
    finally:
        dist.destroy_process_group()
