# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Host-only failure evidence, without an inference server or device operations."""

import json
import os
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rl.rollout.diagnostics import RolloutDiagnostics, _metric_lines, _server_processes


class TestRolloutDiagnostics(unittest.TestCase):
    """Verify incomplete requests and pre-cleanup backend evidence remain distinguishable."""

    def setUp(self) -> None:
        self.resources = ExitStack()
        self.addCleanup(self.resources.close)
        self.directory = Path(self.resources.enter_context(tempfile.TemporaryDirectory()))
        self.environment = patch.dict(os.environ, {"HYPER_RL_DIAGNOSTICS_DIR": str(self.directory), "RANK": "0"})
        self.environment.start()
        (self.directory / "progress-rank0.json").write_text('{"step": 1}')

    def tearDown(self) -> None:
        self.environment.stop()

    @staticmethod
    def request(ordinal: int) -> SimpleNamespace:
        """Return request metadata with a recognizable prompt that must never be persisted."""
        return SimpleNamespace(ordinal=ordinal, start_row=ordinal, prompts=[[987654321]],
                               num_choices=1, seed=42)

    def test_seven_completed_one_failure_preserves_error_and_log(self) -> None:
        """Retain all clients' latest batches and backend errors before cleanup noise exists."""
        first = RolloutDiagnostics("http://127.0.0.1:8100", 600)
        second = RolloutDiagnostics("http://127.0.0.1:8100", 600)
        first.start_batch(4)
        second.start_batch(4)
        for index in range(4):
            first.finish(first.admit(self.request(index), 1024))
        for index in range(3):
            second.finish(second.admit(self.request(index), 1024))
        identifier = second.admit(self.request(3), 1024)
        (self.directory / "train.log").write_text(
            "(Worker_DP2_TP1) ERROR HCCL operation timeout\n"
            "(Engine 2) Running: 1 reqs, Waiting: 0 reqs\n")
        with self.assertLogs("rl.rollout.diagnostics", level="ERROR"):
            second.finish(identifier, TimeoutError())
        report = json.loads((self.directory / "rollout-failure.json").read_text())
        self.assertEqual(report["counts"], {"completed": 7, "waiting": 0, "failed": 1})
        self.assertEqual(report["failed_request"]["error_type"], "TimeoutError")
        self.assertEqual(report["failed_request"]["error_repr"], "TimeoutError()")
        self.assertEqual(report["root_cause"], "UNDETERMINED")
        self.assertEqual(report["dp_replica"], "UNKNOWN")
        self.assertIn("HCCL", report["log_evidence"]["errors"][0])
        self.assertNotIn("987654321", (self.directory / "rollout-failure.json").read_text())
        text = (self.directory / "rollout-failure.txt").read_text()
        self.assertIn("completed=7", text)
        self.assertIn("before client cleanup", text)
        # Later failures must not replace evidence with shutdown EngineDeadError noise.
        before = (self.directory / "rollout-failure.json").read_bytes()
        second.finish(identifier, RuntimeError("EngineDeadError during shutdown"))
        self.assertEqual((self.directory / "rollout-failure.json").read_bytes(), before)

    def test_pending_and_previous_step_are_not_counted_as_success(self) -> None:
        """Waiting requests remain visible and stale batches from another step are excluded."""
        stale = RolloutDiagnostics("endpoint", 600)
        stale.start_batch(1)
        stale.finish(stale.admit(self.request(0), 32))
        (self.directory / "progress-rank0.json").write_text('{"step": 2}')
        client = RolloutDiagnostics("endpoint", 600)
        client.start_batch(2)
        client.admit(self.request(0), 32)
        failed = client.admit(self.request(1), 32)
        with self.assertLogs("rl.rollout.diagnostics", level="ERROR"):
            client.finish(failed, TimeoutError())
        report = json.loads((self.directory / "rollout-failure.json").read_text())
        self.assertEqual(report["counts"], {"completed": 0, "waiting": 1, "failed": 1})
        self.assertEqual(report["step"], 2)

    def test_unwritable_directory_never_raises(self) -> None:
        """Diagnostic filesystem failures cannot replace the actual generation error."""
        client = RolloutDiagnostics("endpoint", 600)
        with patch("rl.rollout.diagnostics._atomic_json", side_effect=OSError("disk full")):
            with self.assertLogs("rl.rollout.diagnostics", level="WARNING"):
                client.start_batch(1)
                identifier = client.admit(self.request(0), 32)
        self.assertIsNone(identifier)

    def test_no_environment_has_no_side_effects(self) -> None:
        """Other rollout users retain the original behavior unless diagnostics are enabled."""
        with patch.dict(os.environ, {}, clear=True):
            client = RolloutDiagnostics("endpoint", 600)
        client.start_batch(1)
        self.assertIsNone(client.admit(self.request(0), 32))
        client.finish(None, TimeoutError())
        client.close()
        self.assertFalse((self.directory / "rollout-failure.json").exists())

    def test_reused_pid_is_not_reported_as_original_server(self) -> None:
        """Avoid attributing unrelated process state to an exited inference server."""
        path = self.directory / "server.json"
        path.write_text(json.dumps({"pid": os.getpid(), "start_ticks": "invalid"}))
        self.assertTrue(_server_processes(path)["pid_reused"])

    def test_engine_metrics_keep_all_replicas_and_counter_deltas(self) -> None:
        """The last engine's pending work must remain visible in a short photo summary."""
        before = {}
        after = {}
        for engine in range(4):
            key = f'vllm:generation_tokens_total{{engine="{engine}"}}'
            before[key] = 10
            after[key] = 10 if engine == 3 else 20
        for engine in range(4):
            after[f'vllm:num_requests_running{{engine="{engine}"}}'] = int(engine == 3)
        text = "\n".join(_metric_lines({"samples": [
            {"time": 0, "values": before}, {"time": 30, "values": after},
        ]}, 35))
        self.assertIn("engine_label=3: tokens=10(delta=0/30s) running=1", text)
        self.assertIn("metrics_age_s=5.0", text)

    def test_metrics_probe_is_bounded_and_preserves_engine_labels(self) -> None:
        """Metrics collection is a local host probe, with a short independent timeout."""
        client = RolloutDiagnostics("http://127.0.0.1:8100", 600)
        response = unittest.mock.MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = (
            b'# HELP ignored\nvllm:generation_tokens_total{engine="2"} 42\n'
            b'vllm:num_requests_running{engine="2"} 1\n')
        opener = unittest.mock.MagicMock()
        opener.open.return_value = response
        client._stop = unittest.mock.MagicMock()
        client._stop.is_set.side_effect = [False, True]
        with patch("rl.rollout.diagnostics.urllib_request.build_opener", return_value=opener):
            client._sample_metrics()
        self.assertEqual(opener.open.call_args.kwargs["timeout"], 2)
        response.read.assert_called_once_with(131072)
        values = json.loads((self.directory / "rollout-metrics.json").read_text())["samples"][0]["values"]
        self.assertEqual(values['vllm:generation_tokens_total{engine="2"}'], 42)
