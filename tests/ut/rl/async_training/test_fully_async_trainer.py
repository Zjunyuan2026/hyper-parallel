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
"""Coordinator refresh decisions and synchronous orchestration compatibility."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from pathlib import Path

from rl.cli import load_config

from rl.async_training.checkpoint import AsyncCheckpoint
from rl.async_training.runtime import ExperienceBuffer
from rl.async_training.protocol import AsyncTrainingConfig
from rl.async_training.fully_async_trainer import FullyAsyncTrainer
from rl.trainer import RLRoleRuntime, SyncTrainer


class TestAsyncDecisions(unittest.TestCase):
    """Avoid a stale-generation loop when the trainer outruns its published policy."""

    def test_selected_checkpoint_reaches_both_restore_owners(self):
        trainer = FullyAsyncTrainer.__new__(FullyAsyncTrainer)
        trainer.config = {"train": {"checkpoint": {}}}
        trainer.roles = Mock()
        trainer.roles.checkpoints.config = {}
        trainer.roles.checkpoints.validate_resume.side_effect = ValueError("stop before model loading")
        trainer.rank, trainer.sampler, trainer.trace = 0, None, Mock()
        with patch("rl.async_training.fully_async_trainer.coordinator_call", side_effect=lambda _, call: call()):
            with patch.object(AsyncCheckpoint, "resolve_path", return_value="checkpoints/step_10"):
                with patch.object(AsyncCheckpoint, "load", return_value=None) as load:
                    with self.assertRaisesRegex(ValueError, "stop before model loading"):
                        trainer.train()
        self.assertEqual(trainer.roles.checkpoints.config["load_path"], "checkpoints/step_10")
        load.assert_called_once_with("checkpoints/step_10", trainer.config)
        trainer.roles.checkpoints.begin.assert_not_called()

    def test_cleanup_preserves_primary_failure_and_attempts_every_resource(self):
        trainer = FullyAsyncTrainer.__new__(FullyAsyncTrainer)
        trainer.config, trainer.rank = {}, 0
        trainer.roles, trainer.sampler, trainer.trace = Mock(), Mock(), Mock()
        trainer.roles.state.global_step = 0
        trainer.sampler.close.side_effect = RuntimeError("sampler cleanup failed")
        trainer.roles.close.side_effect = RuntimeError("role cleanup failed")
        trainer.trace.emit.side_effect = OSError("audit write failed")
        with patch("rl.async_training.fully_async_trainer.coordinator_call", side_effect=lambda _, call: call()):
            with patch.object(AsyncCheckpoint, "resolve_path", side_effect=ValueError("original failure")):
                with self.assertRaisesRegex(ValueError, "original failure"):
                    trainer.train()
        trainer.sampler.close.assert_called_once_with(graceful=False)
        trainer.roles.close.assert_called_once()
        trainer.trace.emit.assert_called_once_with("run_end", completed=False, update=0)

    def test_exhausted_freshness_budget_requests_publish_before_new_sampling(self):
        trainer = FullyAsyncTrainer.__new__(FullyAsyncTrainer)
        trainer.settings = AsyncTrainingConfig()
        trainer.buffer = ExperienceBuffer(trainer.settings, 2)
        trainer.roles = SimpleNamespace(state=SimpleNamespace(global_step=5),
                                        parallel_dims=SimpleNamespace(dp_size=1))
        trainer.sampler = Mock()
        trainer.source_update = 0
        trainer.version = 1
        self.assertEqual(trainer._root_command(), ("publish", None))
        trainer.sampler.fill.assert_not_called()

    def test_synchronous_loop_remains_a_separate_owner(self):
        self.assertTrue(issubclass(SyncTrainer, RLRoleRuntime))
        self.assertIn("_train_step", SyncTrainer.__dict__)
        self.assertNotIn("_train_step", RLRoleRuntime.__dict__)
        self.assertNotIn("train", RLRoleRuntime.__dict__)

    def test_async_entry_preserves_common_configuration_validation(self):
        root = Path(__file__).resolve().parents[4]
        recipe = root / "hyper_parallel/rl/examples/async_training/configs/qwen3_4b_gsm8k_grpo.yaml"
        config = load_config(str(recipe), [])
        with patch("rl.async_training.fully_async_trainer.RLRoleRuntime") as runtime:
            with patch("rl.async_training.fully_async_trainer.dist.get_rank", return_value=0):
                trainer = FullyAsyncTrainer(config)
        self.assertIn("async_training", trainer.config)
        self.assertNotIn("async_training", runtime.call_args.args[0])
        expected = {key: value for key, value in config.items() if key != "async_training"}
        self.assertEqual(runtime.call_args.args[0], expected)

    def test_qwen3_fsdp_adapter_forwards_parameter_exclusions(self):
        from rl.roles.qwen3_builder import _Qwen3FSDP2Manager
        manager = _Qwen3FSDP2Manager.__new__(_Qwen3FSDP2Manager)
        model, source = Mock(), Mock()
        ignored = {Mock()}
        with patch("rl.roles.qwen3_builder.restore_tied_parameter") as restore:
            with patch("rl.roles.qwen3_builder.FSDP2Manager.parallelize", return_value=model) as parallelize:
                result = manager.parallelize(model, source_shard_info=source, ignored_params=ignored)
        self.assertIs(result, model)
        restore.assert_called_once_with(model)
        parallelize.assert_called_once_with(model, source_shard_info=source, ignored_params=ignored)
