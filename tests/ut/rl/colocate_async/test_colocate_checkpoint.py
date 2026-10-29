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
"""Checkpoint integration uses actual durable async manifests and mocked model IO."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from rl.async_training.protocol import AsyncCheckpoint
from tests.ut.rl.colocate_async.support import trainer_fixture


class TestColocateCheckpoint(unittest.TestCase):
    """Freeze complete experience before saving and never publish an incomplete snapshot."""

    def setUp(self) -> None:
        """Install CPU fixtures and register cleanup for each isolated case."""
        self.coordinator = patch("rl.colocate_async.colocate_async_trainer.coordinator_call",
                                 side_effect=lambda _, callback: callback())
        self.coordinator.start()
        self.addCleanup(self.coordinator.stop)
        # TestCase cleanup keeps this resource alive across setUp and the test method.
        self.temporary = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.addCleanup(self.temporary.cleanup)
        self.trainer, self.events, _ = trainer_fixture()
        self.directory = Path(self.temporary.name) / "step_1"
        self.trainer.roles.state.global_step = 1
        self.trainer.roles.checkpoints.directory.return_value = self.directory
        self.trainer.rollouter.drain_and_snapshot.side_effect = self._snapshot
        self.trainer.roles.checkpoints.save.side_effect = self._model_save

    def _snapshot(self):
        self.assertEqual(self.trainer.roles.rollout_engine.phase, "rollout")
        self.events.append("snapshot")
        return {"buffer": {}, "source": {"cursor": 5}}

    def _model_save(self, _state):
        self.assertEqual(self.trainer.roles.rollout_engine.phase, "training")
        self.events.append("model_save")
        self.directory.mkdir()
        (self.directory / "checkpoint_complete.json").write_text("{}")

    def test_periodic_last_step_saves_even_when_final_export_disabled(self):
        self.trainer._save_checkpoint(final=True)
        self.trainer.roles.checkpoints.save.assert_called_once()
        self.trainer.roles.checkpoints.finalize.assert_not_called()
        state = AsyncCheckpoint.load(str(self.directory), self.trainer.config)
        self.assertEqual(state["execution_mode"], "colocate_async")
        self.assertEqual(state["source"], {"cursor": 5})
        self.assertEqual(self.trainer._load_async_state(str(self.directory)), (1, 0))
        self.assertLess(self.events.index("snapshot"), self.events.index("sleep"))
        self.assertLess(self.events.index("model_save"), self.events.index("checkpoint_committed"))
        self.trainer.rollouter.resume.assert_not_called()

    def test_non_final_save_wakes_the_same_policy_without_republication(self):
        self.trainer.roles.evaluator = Mock()
        self.trainer.roles.evaluator.run.side_effect = lambda _: (self.events.append("evaluate") or ({}, []))
        self.trainer._save_checkpoint(final=False)
        self.assertLess(self.events.index("snapshot"), self.events.index("evaluate"))
        self.assertLess(self.events.index("evaluate"), self.events.index("sleep"))
        self.trainer.roles.publish_policy.assert_not_called()
        self.trainer.roles.rollout_engine.prepare_for_rollout.assert_called_once_with()
        self.trainer.rollouter.resume.assert_called_once_with(2)

    def test_save_failure_has_no_commit_marker_and_does_not_resume(self):
        self.trainer.roles.checkpoints.save.side_effect = OSError("disk full")
        with self.assertRaisesRegex(OSError, "disk full"):
            self.trainer._save_checkpoint(final=False)
        self.assertFalse((self.directory / "async_complete.json").exists())
        self.trainer.rollouter.resume.assert_not_called()

    def test_wrong_mode_is_rejected_before_model_restore(self):
        state = {"learner_update": 1, "source_update": 1, "policy_version": 2, "optimizer_steps": 0}
        self.directory.mkdir()
        (self.directory / "checkpoint_complete.json").write_text("{}")
        AsyncCheckpoint.commit(self.directory, state, self.trainer.config)
        with self.assertRaisesRegex(ValueError, "not a colocated"):
            self.trainer._load_async_state(str(self.directory))

    def test_codex_evaluation_hands_off_gateway_and_restores_source(self):
        self.trainer.config["agentic"] = {"runner": "codex"}
        self.trainer.roles.evaluation_needs_gateway_handoff = True
        self.trainer.roles.codex_runtime = Mock()
        self.trainer.roles.codex_runtime.close.side_effect = lambda: self.events.append("eval_gateway_closed")
        self.trainer.roles.evaluator = Mock()
        self.trainer.roles.evaluator.run.side_effect = lambda _: (self.events.append("evaluate") or ({}, []))
        previous = self.trainer.rollouter
        previous.close.side_effect = lambda **_: self.events.append("producer_closed")
        with patch("rl.colocate_async.colocate_async_trainer.ColocateAsyncRollouter") as factory:
            self.trainer._save_checkpoint(final=False)
        self.assertLess(self.events.index("snapshot"), self.events.index("producer_closed"))
        self.assertLess(self.events.index("producer_closed"), self.events.index("evaluate"))
        self.assertLess(self.events.index("evaluate"), self.events.index("eval_gateway_closed"))
        self.assertLess(self.events.index("eval_gateway_closed"), self.events.index("model_save"))
        previous.close.assert_called_once_with(graceful=True)
        previous.resume.assert_not_called()
        self.assertEqual(factory.call_args.args[2]["source"], {"cursor": 5})
        self.assertEqual(factory.call_args.args[2]["learner_update"], 1)
        factory.return_value.resume.assert_called_once_with(2)
        self.assertIsNone(self.trainer._snapshot)

    def test_codex_evaluation_failure_closes_gateway_without_restart_or_save(self):
        self.trainer.config["agentic"] = {"runner": "codex"}
        self.trainer.roles.evaluation_needs_gateway_handoff = True
        self.trainer.roles.codex_runtime = Mock()
        self.trainer.roles.evaluator = Mock()
        self.trainer.roles.evaluator.run.side_effect = RuntimeError("evaluation failed")
        with patch("rl.colocate_async.colocate_async_trainer.ColocateAsyncRollouter") as factory:
            with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
                self.trainer._save_checkpoint(final=False)
        self.trainer.rollouter.close.assert_called_once_with(graceful=True)
        self.trainer.roles.codex_runtime.close.assert_called_once_with()
        self.trainer.roles.checkpoints.save.assert_not_called()
        factory.assert_not_called()

    def test_gateway_close_failure_prevents_save_and_producer_restart(self):
        self.trainer.config["agentic"] = {"runner": "codex"}
        self.trainer.roles.evaluation_needs_gateway_handoff = True
        self.trainer.roles.codex_runtime = Mock()
        self.trainer.roles.codex_runtime.close.side_effect = RuntimeError("gateway close failed")
        self.trainer.roles.evaluator = Mock()
        self.trainer.roles.evaluator.run.return_value = ({}, [])
        with patch("rl.colocate_async.colocate_async_trainer.ColocateAsyncRollouter") as factory:
            with self.assertRaisesRegex(RuntimeError, "gateway close failed"):
                self.trainer._save_checkpoint(final=False)
        self.trainer.roles.checkpoints.save.assert_not_called()
        factory.assert_not_called()

    def test_producer_rebuild_failure_preserves_committed_checkpoint(self):
        self.trainer.config["agentic"] = {"runner": "codex"}
        self.trainer.roles.evaluation_needs_gateway_handoff = True
        self.trainer.roles.codex_runtime = Mock()
        self.trainer.roles.evaluator = Mock()
        self.trainer.roles.evaluator.run.return_value = ({}, [])
        previous = self.trainer.rollouter
        with patch("rl.colocate_async.colocate_async_trainer.ColocateAsyncRollouter",
                   side_effect=RuntimeError("producer rebuild failed")):
            with self.assertRaisesRegex(RuntimeError, "producer rebuild failed"):
                self.trainer._save_checkpoint(final=False)
        state = AsyncCheckpoint.load(str(self.directory), self.trainer.config)
        self.assertEqual(state["source"], {"cursor": 5})
        self.trainer.roles.codex_runtime.close.assert_called_once_with()
        previous.resume.assert_not_called()
        self.trainer.roles.publish_policy.assert_not_called()

    def test_non_root_evaluation_never_reads_root_sampler(self):
        self.trainer.rank, self.trainer.rollouter = 1, None
        self.trainer.config["agentic"] = {"runner": "codex"}
        self.trainer.roles.evaluation_needs_gateway_handoff = True
        self.trainer.roles.codex_runtime = Mock()
        self.trainer.roles.evaluator = Mock()
        self.trainer.roles.evaluator.run.return_value = ({}, [])
        with patch("rl.colocate_async.colocate_async_trainer.coordinator_call", return_value=None):
            self.assertTrue(self.trainer._evaluate_checkpoint())
        self.trainer.roles.evaluator.run.assert_called_once_with(2)
        self.trainer.roles.codex_runtime.close.assert_called_once_with()
