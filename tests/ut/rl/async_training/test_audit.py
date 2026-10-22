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
"""Independent arithmetic checks for asynchronous targets and the real backward observer."""
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rl.algorithm import GRPOAlgorithm, GRPOConfig
from rl.utils.async_audit import ActorLossAudit, EventTrace, state_digest, write_rollout_diagnostics
from rl.async_training.checkpoint import configuration_fingerprint
from rl.async_training.protocol import AsyncTrainingConfig


def scalar_loss(current, old, reference, advantage, coefficient=0.07):
    """Evaluate the clipped objective using Python scalars, independent of Torch losses."""
    ratio = math.exp(current - old)
    loss = max(-advantage * ratio, -advantage * min(1.2, max(0.8, ratio)))
    if advantage < 0:
        loss = min(loss, -3.0 * advantage)
    delta = reference - current
    return loss + coefficient * min(10.0, max(0.0, math.exp(delta) - delta - 1.0))


class TestAudit(unittest.TestCase):
    """Compare production backward with finite differences, including every clipping branch."""

    def test_diagnostics_distinguish_group_signal_from_batch_reward_variance(self):
        rows = []
        for index, reward in enumerate((0., 0., 1., 1., 0., 1.)):
            rows.append(SimpleNamespace(
                group_id=f"g{index // 2}", trajectory_id=f"t{index}", prompt_id="p", reward=reward,
                reward_components={"correctness": reward}, action_mask=torch.tensor([1, 1]),
                truncated=True, policy_version=2,
                turns=(SimpleNamespace(role="user", trainable=False, content="question"),
                       SimpleNamespace(role="assistant", trainable=True, content="long\n回答 #### 3",
                                       metadata={"finish_reason": "stop" if index == 0 else "length"})),
                metadata={"turn_infos": ({"expected_answer": "3", "extracted_answer": "3",
                                          "reward_scorer": "examples.gsm8k.agent.compute_gsm8k_reward"},)},
            ))
        advantages = torch.tensor([[0., 0., 9.]] * 4 + [[-1., -1., 9.], [1., 1., 9.]])
        batch = SimpleNamespace(trajectories=rows, advantages=advantages,
                                loss_action_mask=torch.tensor([[1, 1, 0]] * 6))
        before = advantages.clone()
        with tempfile.TemporaryDirectory() as directory:
            write_rollout_diagnostics(batch, directory, 3, 1, 2)
            path = Path(directory) / "step_3_rank_1.jsonl"
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertFalse(path.with_suffix(".jsonl.tmp").exists())
        self.assertEqual([record["group_nonzero_advantage"] for record in records], [False] * 4 + [True] * 2)
        self.assertEqual([record["group_zero_reward_variance"] for record in records], [True] * 4 + [False] * 2)
        self.assertEqual(records[0]["response"], "long\n回答 #### 3")
        self.assertTrue(records[0]["length_limit_hit"])
        self.assertFalse(records[0]["generation_length_limited"])
        self.assertTrue(records[0]["environment_truncated"])
        self.assertTrue(records[0]["reward_diagnostics_available"])
        self.assertEqual(records[0]["expected_answer"], "3")
        torch.testing.assert_close(batch.advantages, before)

    def test_scalar_finite_differences_match_observed_backward(self):
        ratios = [0.6, 0.9, 1.1, 1.6, 0.6, 0.9, 1.4, 4.0, 1.05]
        old = torch.full((1, len(ratios)), -5.0, dtype=torch.float64)
        current = (old + torch.tensor([math.log(value) for value in ratios])).requires_grad_()
        reference = (current.detach() - 0.27).float()
        advantages = torch.tensor([[1., 1., 1., 1., -1., -1., -1., -1., 1.]], dtype=torch.float64)
        mask = torch.tensor([[True] * 8 + [False]])
        algorithm = GRPOAlgorithm(GRPOConfig(kl_coef=0.07))
        with tempfile.TemporaryDirectory() as directory:
            trace = EventTrace(directory, "trainer")
            audit = ActorLossAudit(algorithm, trace)
            experience = SimpleNamespace(
                sequences=torch.ones(1, 10), rollout_log_probs=old, old_log_probs=old,
                advantages=advantages, rewards=torch.ones(1), loss_action_mask=mask,
                trajectories=[SimpleNamespace(group_id="g", trajectory_id="g:0", policy_version=2)],
            )
            audit.begin(
                3, experience, {"response_mini_batch_size": 1, "micro_batch_size": 1, "policy_update_epochs": 1}, 1,
            )
            output = audit.compute_actor_loss(
                current_log_probs=current, old_log_probs=old, reference_log_probs=reference,
                advantages=advantages, action_mask=mask,
            )
            (output.total_loss_sum / int(mask.sum())).backward()
            audit.finish()
            observed = torch.load(Path(directory) / "loss_step_3_rank_0_call_0.pt", weights_only=False)
            expected = []
            epsilon = 1.0e-4
            for index in range(len(ratios)):
                value, anchor, target = current[0, index].item(), old[0, index].item(), reference[0, index].item()
                advantage = advantages[0, index].item()
                derivative = (scalar_loss(value + epsilon, anchor, target, advantage)
                              - scalar_loss(value - epsilon, anchor, target, advantage)) / (2 * epsilon)
                expected.append(derivative / int(mask.sum()) if mask[0, index] else 0.)
            # Production KL intentionally uses FP32 even when policy inputs are FP64.
            torch.testing.assert_close(observed["log_prob_gradient"], torch.tensor([expected], dtype=torch.float64),
                                       atol=2.0e-8, rtol=2.0e-5)
            torch.testing.assert_close(observed["ratio"], torch.tensor([ratios], dtype=torch.float64),
                                       atol=2.0e-7, rtol=2.0e-7)
            self.assertEqual(float(observed["log_prob_gradient"][0, -1]), 0.0)

    def test_parameter_gradient_matches_independent_softmax_finite_difference(self):
        logits = torch.tensor([0.1, -0.2, 0.4], dtype=torch.float64, requires_grad=True)
        actions = [0, 1, 2, 0]
        old = torch.tensor([[-1.6, -1.1, -1.2, -2.3]], dtype=torch.float64)
        advantage = torch.tensor([[1., -1., 1., -1.]], dtype=torch.float64)
        mask = torch.ones_like(old, dtype=torch.bool)
        current = logits.log_softmax(0)[actions].unsqueeze(0)
        algorithm = GRPOAlgorithm(GRPOConfig(kl_coef=0.0))
        loss = algorithm.compute_actor_loss(current, old, None, advantage, mask).total_loss_sum / 4
        loss.backward()

        def independent(parameters):
            normalizer = math.log(sum(math.exp(value) for value in parameters))
            return sum(scalar_loss(parameters[action] - normalizer, old[0, index].item(), 0.,
                                   advantage[0, index].item(), coefficient=0.)
                       for index, action in enumerate(actions)) / 4

        expected = []
        for index in range(3):
            plus, minus = logits.detach().tolist(), logits.detach().tolist()
            plus[index] += 1.0e-6
            minus[index] -= 1.0e-6
            expected.append((independent(plus) - independent(minus)) / 2.0e-6)
        torch.testing.assert_close(logits.grad, torch.tensor(expected, dtype=torch.float64),
                                   atol=1.0e-9, rtol=1.0e-7)

    def test_trace_preserves_host_clock_and_complete_records(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = EventTrace(directory, "rollouter")
            trace.emit("sampling_start", identity=7)
            trace.emit("sampling_end", identity=7)
            records = [json.loads(line) for line in trace.path.read_text().splitlines()]
            self.assertEqual([row["event"] for row in records], ["clock", "sampling_start", "sampling_end"])
            times = [row["monotonic_ns"] for row in records]
            self.assertEqual(times, sorted(times))
            self.assertTrue(all(row["wall_time_ns"] > 0 for row in records))

    def test_diagnostics_do_not_change_checkpoint_identity(self):
        config = {"train": {}, "async_training": {"publish_every_updates": 2}}
        fingerprint = configuration_fingerprint(config)
        config["async_training"]["audit_dir"] = "/tmp/other-run"
        self.assertEqual(configuration_fingerprint(config), fingerprint)
        config["async_training"]["publish_every_updates"] = 3
        self.assertNotEqual(configuration_fingerprint(config), fingerprint)
        for value in ["", " ", True, 12]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                AsyncTrainingConfig.from_mapping({"audit_dir": value})

    def test_state_hash_preserves_values_dtype_and_ordered_sequence(self):
        self.assertEqual(state_digest({"x": torch.tensor([1., 2.])}),
                         state_digest({"x": torch.tensor([1., 2.])}))
        self.assertNotEqual(state_digest(torch.tensor([1., 2.])), state_digest(torch.tensor([2., 1.])))
        self.assertNotEqual(state_digest(torch.tensor([1.])), state_digest(torch.tensor([1.], dtype=torch.float64)))

    def test_server_progress_records_request_bounds_and_aggregates_dp(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = EventTrace(directory, "rollouter")
            response = Mock()
            response.__enter__ = Mock(return_value=response)
            response.__exit__ = Mock(return_value=None)
            response.read.return_value = (b'vllm:generation_tokens_total{engine="0"} 12\n'
                                          b'vllm:generation_tokens_total{engine="1"} 7\n')
            with patch("rl.utils.async_audit.urllib.request.urlopen", return_value=response):
                trace.probe_generation("http://localhost:8169")
            record = json.loads(trace.path.read_text().splitlines()[-1])
            self.assertEqual(record["counters"]["vllm:generation_tokens_total"], 19.)
            self.assertLessEqual(record["request_start_ns"], record["request_end_ns"])
            self.assertLessEqual(record["request_end_ns"], record["monotonic_ns"])
