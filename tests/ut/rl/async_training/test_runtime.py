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
"""Spawn-process flow, full transport draining and failure propagation."""
import unittest
from dataclasses import asdict

from rl.async_training.protocol import AsyncTrainingConfig
from rl.async_training.runtime import AsyncRuntime, ExperienceBuffer
from tests.ut.rl.async_training.support import broken_rollouter, fake_rollouter


class TestAsyncRuntime(unittest.TestCase):
    """Run actual spawned producers; no distributed or accelerator runtime is initialized."""

    def setUp(self):
        self.settings = AsyncTrainingConfig(
            max_inflight_groups=2, max_buffer_groups=2, max_group_bytes=65536,
            max_buffer_bytes=131072, operation_timeout=30,
        )
        self.config = {"async_training": asdict(self.settings)}

    def test_drain_collects_full_queue_and_snapshot_resumes(self):
        buffer = ExperienceBuffer(self.settings, 2)
        runtime = AsyncRuntime(self.config, buffer, target=fake_rollouter)
        try:
            runtime.resume(1)
            runtime.fill(1, 0)
            runtime.drain()
            self.assertEqual(len(buffer.ready), 2)
            self.assertFalse(buffer.reserved)
            self.assertEqual(runtime.snapshot()["cursor"], 2)
            buffer.select(2)
            buffer.commit()
            runtime.resume(2)
            runtime.fill(2, 1)
            runtime.drain()
            self.assertEqual(runtime.snapshot()["cursor"], 4)
        finally:
            runtime.close(graceful=True)
        self.assertFalse(runtime.process.is_alive())

    def test_startup_failure_is_propagated(self):
        with self.assertRaisesRegex(RuntimeError, "injected startup failure"):
            AsyncRuntime(self.config, ExperienceBuffer(self.settings, 2), target=broken_rollouter)
