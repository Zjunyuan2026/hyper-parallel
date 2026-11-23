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
"""Finite-epoch, split isolation and final evaluation regressions on CPU."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

# CPU tensor math only; distributed collectives are mocked.
# pylint: disable=forbidden-backend-import
import pandas as pd
import torch
from transformers import BatchEncoding

from examples.consistency.scripts import full_gsm8k as formal
from rl.algorithm import build_algorithm
from rl.data.experience import ExperiencePreparer, build_experience_batch, pad_rollout_responses
from rl.data.contracts import Trajectory, Turn
from rl.rollout.base import GenerationSettings


class FormalGsm8kTests(unittest.TestCase):
    """Check coverage and padding using real CPU target math and mocked communication."""

    def test_all_dp_shards_cover_one_epoch_and_checkpoint_the_cursor(self) -> None:
        """6726 real questions produce 49 complete batches and one 62-question tail."""
        loaders = [formal.FiniteEpochLoader(range(6726), list, rank, 8) for rank in range(8)]
        iterators = [iter(loader) for loader in loaders]
        seen = []
        for step in range(50):
            shards = [next(iterator) for iterator in iterators]
            actual = [index for shard in shards for index in shard]
            self.assertEqual(len(actual), 136 if step < 49 else 62)
            seen.extend(actual)
        self.assertEqual(sorted(seen), list(range(6726)))
        self.assertEqual([len(shard) for shard in shards], [8, 8, 8, 8, 8, 8, 7, 7])
        restored = formal.FiniteEpochLoader(range(6726), list, 0, 8)
        restored.load_state_dict(loaders[0].state_dict())
        self.assertEqual(list(restored), [])
        with self.assertRaises(ValueError):
            restored.set_epoch(1)
        with self.assertRaises(ValueError):
            restored.load_state_dict(loaders[1].state_dict())

    def test_tail_padding_preserves_real_tokens_and_grpo_advantages(self) -> None:
        """Copied context adds no sampled actions, reward, or advantage to dummy rows."""
        trajectories = tuple(Trajectory(
            trajectory_id=str(index), prompt_id=str(index // 4), group_id=str(index // 4), policy_version=49,
            turns=(Turn("user", "question", 0, 1, False), Turn("assistant", "answer", 1, 3, True)),
            token_ids=torch.tensor([1, 2, 3]), attention_mask=torch.ones(3, dtype=torch.bool),
            action_mask=torch.tensor([False, True, True]), rollout_log_probs=torch.tensor([-1., -2.]),
            reward=float(index % 4 == 0), reward_components={}, done=True, truncated=False,
            terminal_reason="completed", worker_policy_version=49) for index in range(28))
        settings = GenerationSettings(3, 1., 1., 0, True, 0, 4, collect_log_probs=True)
        real = build_experience_batch(trajectories, generation_seconds=1., settings=settings, metadata={})
        padded = pad_rollout_responses(real, 32)
        torch.testing.assert_close(padded.sequences[:28], real.sequences)
        torch.testing.assert_close(padded.rollout_log_probs[:28], real.rollout_log_probs)
        self.assertFalse(padded.loss_action_mask[28:].any().item())
        self.assertEqual(padded.rewards[28:].sum().item(), 0)
        preparer = ExperiencePreparer(build_algorithm({"name": "grpo", "loss_aggregation": "token-mean"}))
        expected = preparer.prepare_targets(real)
        actual = preparer.prepare_targets(padded)
        torch.testing.assert_close(actual.advantages[:28], expected.advantages)
        self.assertEqual(actual.advantages[28:].abs().sum().item(), 0)
        with self.assertRaises(ValueError):
            pad_rollout_responses(actual, 36)

    def test_final_test_uses_separate_dataset_and_does_not_change_validation(self) -> None:
        """Only step 50 triggers the official test; dataset ownership is restored."""
        with TemporaryDirectory() as temporary:
            run = formal.FormalRun.__new__(formal.FormalRun)
            run.rank, run.directory, run.evaluations = 0, Path(temporary), []
            validation, test = range(747), range(1319)
            calls = []
            evaluator = SimpleNamespace(dataset=validation)
            def evaluate(step):
                calls.append((step, len(evaluator.dataset)))
                return {"validation/total": len(evaluator.dataset), "validation/accuracy": 0.5}, []
            evaluator.run = evaluate
            run.trainer = SimpleNamespace(evaluator=evaluator, _run_rank_synchronized=lambda stage, fn: fn())
            run.test_dataset = test
            run._install_evaluation()
            for step in (0, 10, 20, 30, 40, 50):
                metrics, _ = evaluator.run(step)
            self.assertEqual(calls, [(step, 747) for step in (0, 10, 20, 30, 40, 50)] + [(50, 1319)])
            self.assertIs(evaluator.dataset, validation)
            self.assertEqual(metrics["test/total"], 1319)
            self.assertEqual(json.loads((run.directory / "test-step50.json").read_text())["split"], "test")

    def test_source_splits_are_disjoint_and_oversize_prompts_fail_before_writing(self) -> None:
        """CPU preparation preserves official test rows and rejects implicit truncation."""
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            pd.DataFrame({"prompt": [f"training {i}" for i in range(7473)]}).to_parquet(root / "train.parquet")
            pd.DataFrame({"prompt": [f"testing {i}" for i in range(1319)]}).to_parquet(root / "test.parquet")
            def encode(*args, **kwargs):
                self.assertTrue(kwargs["return_dict"])
                self.assertFalse(kwargs["truncation"])
                return BatchEncoding({"input_ids": [1] * 20, "attention_mask": [1] * 20})

            tokenizer = SimpleNamespace(apply_chat_template=encode)
            with patch.object(formal.AutoTokenizer, "from_pretrained", return_value=tokenizer):
                with self.assertRaises(ValueError):
                    formal.prepare(root, root / "invalid", root, 10)
                self.assertFalse((root / "invalid").exists())
                result = formal.prepare(root, root / "prepared", root, 32)
            self.assertEqual(result["counts"], {"train": 6726, "validation": 747, "test": 1319})
            self.assertEqual(result["max_prompt_tokens"], {"train": 20, "test": 20})
            train = set(result["source_indices"]["train"])
            validation = set(result["source_indices"]["validation"])
            self.assertFalse(train & validation)
            self.assertEqual(train | validation, set(range(7473)))
            original = pd.read_parquet(root / "test.parquet")
            prepared = pd.read_parquet(root / "prepared/test.parquet")
            self.assertEqual(original["prompt"].tolist(), prepared["prompt"].tolist())

    def test_runtime_hooks_audit_actual_batches_and_preserve_checkpoint_loader(self) -> None:
        """Use SyncTrainer's existing loader/evaluator/resource contracts rather than builder-only fields."""
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = SimpleNamespace(max_steps=50, global_step=0, consumed_samples=0)
            original_loader = SimpleNamespace(dataset=range(6726), collate_fn=list)
            resources = SimpleNamespace(tokenizer=object(), train_dataloader=original_loader)
            trainer = SimpleNamespace(state=state, train_dataloader=original_loader,
                checkpoints=SimpleNamespace(resources=resources), parallel_dims=SimpleNamespace(dp_rank=0, dp_size=8),
                evaluator=SimpleNamespace(dataset=range(747), run=lambda step: ({}, [])),
                targets=SimpleNamespace(prepare=lambda rollout, diagnostics, timings: rollout),
                actor=SimpleNamespace(_response_mini_batch_size=68),
                rollout_engine=SimpleNamespace(is_request_owner=True),
                _generate_training_rollout=lambda batch, timings: (batch, "real rollout"),
                _run_rank_synchronized=lambda stage, fn: fn())
            with patch.object(formal.dist, "get_rank", return_value=0), \
                 patch.object(formal.PromptDataset, "from_config", return_value=range(1319)):
                run = formal.FormalRun(trainer, {"data": {}}, root, root)
            self.assertIs(resources.train_dataloader, run.loader)
            expected = run.loader.batches[0]
            def gather(result, local):
                result[:] = [expected[rank::8] for rank in range(8)] + [None] * 56
            with patch.object(formal.dist, "get_world_size", return_value=64), \
                 patch.object(formal.dist, "all_gather_object", side_effect=gather):
                trainer._generate_training_rollout({"sample_indices": expected[::8]}, {})
            self.assertEqual(state.consumed_samples, 136)
            self.assertEqual(run.coverage[0]["questions"], 136)
            with self.assertRaises(ValueError):
                run.finish()

    def test_completed_epoch_requires_all_checkpoint_markers_and_final_export(self) -> None:
        """Coverage or validation alone cannot certify a durable formal result."""
        with TemporaryDirectory() as temporary:
            run = formal.FormalRun.__new__(formal.FormalRun)
            run.rank, run.directory, run.shared = 0, Path(temporary), Path(temporary)
            run.coverage = [{"indices": batch} for batch in formal.epoch_batches()]
            run.evaluations = [{"step": step, "split": "validation"} for step in (0, 10, 20, 30, 40, 50)]
            run.evaluations.append({"step": 50, "split": "test"})
            with self.assertRaises(ValueError):
                run.finish()
            for step in (10, 20, 30, 40, 50):
                directory = run.shared / f"checkpoints/step_{step}"
                directory.mkdir(parents=True)
                (directory / "checkpoint_complete.json").write_text(json.dumps({"step": step}))
            export = run.shared / "checkpoints/step_50/hf"
            export.mkdir()
            (export / "config.json").write_text("{}")
            (export / "model.safetensors").write_text("synthetic IO fixture; not model weights")
            run.finish()
            self.assertTrue(json.loads((run.directory / "formal-summary.json").read_text())["passed"])
