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
"""Credit conservation, out-of-order completion, staleness and mixed-version contracts."""
import unittest
from dataclasses import replace

import torch

from rl.algorithm.loss import build_algorithm
from rl.async_training.runtime import ExperienceBuffer
from rl.async_training.protocol import AsyncTrainingConfig
from rl.async_training.protocol import SamplingTask
from rl.dataset.batch_builder import build_experience_batch, ExperiencePreparer
from rl.roles.rollout.base import GenerationSettings
from tests.ut.rl.async_training.support import group


class TestExperienceBuffer(unittest.TestCase):
    """Use actual CPU trajectory payloads rather than mocking data validation."""

    def setUp(self):
        self.settings = AsyncTrainingConfig(max_inflight_groups=2, max_buffer_groups=2,
                                            max_group_bytes=65536, max_buffer_bytes=131072)
        self.buffer = ExperienceBuffer(self.settings, 2)

    def submit(self, task):
        self.buffer.accept(group(task).encode(2, 65536))

    def test_capacity_includes_selected_batch(self):
        first = self.buffer.reserve(1, 0)
        second = self.buffer.reserve(1, 0)
        self.assertIsNone(self.buffer.reserve(1, 0))
        self.submit(first)
        self.submit(second)
        self.buffer.select(2)
        self.assertIsNone(self.buffer.reserve(1, 0))
        with self.assertRaises(RuntimeError):
            self.buffer.state_dict()
        self.buffer.commit()
        self.assertIsNotNone(self.buffer.reserve(2, 1))

    def test_out_of_order_completion_preserves_groups(self):
        first, second = self.buffer.reserve(1, 0), self.buffer.reserve(1, 0)
        self.submit(second)
        self.submit(first)
        self.assertEqual([item.task.identity for item in self.buffer.select(2)], [1, 0])

    def test_duplicate_and_unreserved_results_fail(self):
        task = self.buffer.reserve(1, 0)
        self.submit(task)
        with self.assertRaises(ValueError):
            self.submit(task)
        with self.assertRaises(ValueError):
            self.submit(SamplingTask(100, 1, 0))

    def test_version_mismatch_does_not_release_reservation(self):
        task = self.buffer.reserve(1, 0)
        with self.assertRaises(ValueError):
            self.submit(SamplingTask(task.identity, 2, 1))
        self.assertIn(task.identity, self.buffer.reserved)

    def test_partial_group_is_rejected(self):
        task = self.buffer.reserve(1, 0)
        with self.assertRaises(ValueError):
            replace(group(task), trajectories=group(task).trajectories[:1]).encode(2, 65536)

    def test_staleness_is_whole_group_and_strictly_over_threshold(self):
        task = self.buffer.reserve(1, 0)
        self.submit(task)
        self.assertEqual(self.buffer.evict_stale(4), 0)
        self.assertEqual(self.buffer.evict_stale(5), 1)
        self.assertEqual(self.buffer.dropped_groups, 1)

    def test_future_samples_fail(self):
        self.submit(self.buffer.reserve(3, 2))
        with self.assertRaises(ValueError):
            self.buffer.evict_stale(1)

    def test_snapshot_roundtrip_preserves_accounting(self):
        self.submit(self.buffer.reserve(1, 0))
        state = self.buffer.state_dict()
        restored = ExperienceBuffer(self.settings, 2)
        restored.load_state_dict(state)
        restored.select(1)
        restored.commit()
        self.assertEqual(restored.consumed_groups, 1)
        self.assertEqual(restored.reserve(2, 1).identity, 1)

    def test_corrupt_snapshot_fails(self):
        state = {"next_identity": 4, "ready": [], "consumed_groups": 0, "dropped_groups": 0}
        with self.assertRaises(ValueError):
            self.buffer.load_state_dict(state)

    def test_mixed_versions_keep_sync_default_strict(self):
        rows = group(SamplingTask(0, 1, 0)).trajectories + group(SamplingTask(1, 2, 1)).trajectories
        settings = GenerationSettings(2, 1., 1., 0, True, 0, 3, collect_log_probs=True)
        with self.assertRaises(ValueError):
            build_experience_batch(rows, 0.1, settings, {})
        batch = build_experience_batch(rows, 0.1, settings, {}, mixed_policy_versions=True)
        self.assertIsNone(batch.worker_policy_version)
        algorithm = build_algorithm({"name": "grpo", "kl_coef": 0, "loss_aggregation": "token-mean"})
        prepared = ExperiencePreparer(algorithm).prepare(batch)
        torch.testing.assert_close(prepared.advantages[:, 0],
                                   torch.tensor([-2**-0.5, 2**-0.5, -2**-0.5, 2**-0.5]))
        torch.testing.assert_close(batch.old_log_probs, batch.rollout_log_probs)

    def test_wire_budget_rejects_oversize(self):
        with self.assertRaises(ValueError):
            group(SamplingTask(0, 1, 0)).encode(2, 16)
