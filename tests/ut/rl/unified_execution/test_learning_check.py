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
"""CPU regressions for real update, publication and subsequent probability evidence."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import pandas as pd
import yaml

from hyper_parallel.rl.examples.consistency.scripts import learning_check, configure_run, select_learning_prompts


class RealLearningChecks(unittest.TestCase):
    """Do not accept zero-advantage, unchanged-weight or failed-publication runs."""

    def setUp(self) -> None:
        self.distributed = self.enterContext(patch.object(learning_check, "dist"))
        self.distributed.get_rank.return_value = 0
        self.distributed.get_world_size.return_value = 1
        self.model = torch.nn.Linear(2, 1, bias=False)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.experience = SimpleNamespace(advantages=torch.tensor([[1., 1.], [-1., -1.]]),
                                          action_mask=torch.ones(2, 3))
        self.trainer = SimpleNamespace(
            state=SimpleNamespace(global_step=0), device=torch.device("cpu"),
            actor=SimpleNamespace(actor_model=self.model, update=self.update),
            publication=SimpleNamespace(publish=lambda version: version))

    def update(self, experience):
        """Run an actual CPU backward and optimizer step for the success case."""
        self.optimizer.zero_grad()
        output = self.model(torch.tensor([[1., 2.], [3., 1.]]))
        loss = (output[:, 0] * experience.advantages[:, 0]).sum()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 100.)
        self.optimizer.step()
        return SimpleNamespace(gradient_norm=float(norm))

    def test_actual_update_requires_publication_then_next_probability_gate(self) -> None:
        """Record learning only after a real change and completed publication."""
        with TemporaryDirectory() as directory:
            self.directory = Path(directory)
            self.observer = learning_check.LearningObserver(self.trainer, self.directory, 1e-12)
            self.observer.install()
            self.trainer.actor.update(self.experience)
            record = self.observer.records[0]
            self.assertTrue(record["local_weights_changed"])
            self.assertFalse(record["publication_completed"])
            with self.assertRaisesRegex(RuntimeError, "publication"):
                self.observer.verified_probabilities(2)
            self.assertEqual(self.trainer.publication.publish(1), 1)
            self.assertFalse(record["next_step_logp_verified"])
            self.observer.verified_probabilities(2)
            saved = json.loads((self.directory / "learning-step1-rank0.json").read_text())
            self.assertTrue(saved["next_step_logp_verified"])

    def test_unchanged_parameters_fail_even_with_positive_reported_gradient(self) -> None:
        """A positive metric alone cannot prove a model weight changed."""
        with TemporaryDirectory() as directory:
            self.directory = Path(directory)
            self.observer = learning_check.LearningObserver(self.trainer, self.directory, 1e-12)
            self.trainer.actor.update = lambda experience: SimpleNamespace(gradient_norm=1.)
            self.observer.install()
            with self.assertRaisesRegex(RuntimeError, "real learning check failed"):
                self.trainer.actor.update(self.experience)

    def test_zero_advantages_fail(self) -> None:
        """Uniform-reward groups must not count as a meaningful first update."""
        with TemporaryDirectory() as directory:
            self.directory = Path(directory)
            self.observer = learning_check.LearningObserver(self.trainer, self.directory, 1e-12)
            self.experience.advantages.zero_()
            self.observer.install()
            with self.assertRaisesRegex(RuntimeError, "real learning check failed"):
                self.trainer.actor.update(self.experience)

    def test_nonfinite_gradient_fails(self) -> None:
        """Reject invalid numerical evidence even if some weights changed."""
        with TemporaryDirectory() as directory:
            self.directory = Path(directory)
            self.observer = learning_check.LearningObserver(self.trainer, self.directory, 1e-12)
            def invalid_update(experience):
                result = self.update(experience)
                result.gradient_norm = float("nan")
                return result
            self.trainer.actor.update = invalid_update
            self.observer.install()
            with self.assertRaisesRegex(RuntimeError, "real learning check failed"):
                self.trainer.actor.update(self.experience)

    def test_failed_publication_is_not_marked_complete(self) -> None:
        """Keep failed transfers distinguishable from completed publication."""
        with TemporaryDirectory() as directory:
            self.directory = Path(directory)
            self.observer = learning_check.LearningObserver(self.trainer, self.directory, 1e-12)
            def fail(version):
                raise RuntimeError("transfer failed")
            self.trainer.publication.publish = fail
            self.observer.install()
            self.trainer.actor.update(self.experience)
            with self.assertRaisesRegex(RuntimeError, "transfer failed"):
                self.trainer.publication.publish(1)
            self.assertFalse(self.observer.records[0]["publication_completed"])

    def test_tracking_allows_uniform_reward_zero_gradient_steps(self) -> None:
        """Full-source training records legitimate zero updates without fabricating rewards."""
        with TemporaryDirectory() as directory:
            observer = learning_check.LearningObserver(
                self.trainer, Path(directory), 0, require_first_update=False)
            self.experience.advantages.zero_()
            observer.install()
            result = self.trainer.actor.update(self.experience)
            self.assertEqual(result.gradient_norm, 0)
            self.assertFalse(observer.records[0]["local_weights_changed"])
            self.trainer.publication.publish(1)
            observer.verified_probabilities(2)
            self.assertTrue(observer.records[0]["next_step_logp_verified"])

    def test_tracking_still_rejects_nonfinite_gradients(self) -> None:
        """Observation mode is not permission to ignore numerical failures."""
        with TemporaryDirectory() as directory:
            self.trainer.actor.update = lambda experience: SimpleNamespace(gradient_norm=float("nan"))
            observer = learning_check.LearningObserver(
                self.trainer, Path(directory), 0, require_first_update=False)
            observer.install()
            with self.assertRaisesRegex(RuntimeError, "Nonfinite"):
                self.trainer.actor.update(self.experience)


class RealPromptSelectionTests(unittest.TestCase):
    """Keep real labels private and unchanged while materializing an acceptance subset."""

    def test_selection_preserves_rows_labels_and_source(self) -> None:
        """Selection may repeat records but must not rewrite questions or answers."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source.parquet", root / "selected.parquet"
            frame = pd.DataFrame({"prompt": ["question A", "question B", "question C"],
                                  "extra_info": [{"answer": "10"}, {"answer": "20"}, {"answer": "30"}]})
            frame.to_parquet(source)
            original = source.read_bytes()
            result = select_learning_prompts.select_prompts(source, output, [2, 0], 5)
            selected = pd.read_parquet(output)
            self.assertEqual(selected["prompt"].tolist(), ["question C", "question A", "question C",
                                                          "question A", "question C"])
            self.assertEqual(selected["extra_info"].tolist(),
                             [{"answer": "30"}, {"answer": "10"}, {"answer": "30"},
                              {"answer": "10"}, {"answer": "30"}])
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(result["selected_original_rows"], [2, 0, 2, 0, 2])
            with self.assertRaises(FileExistsError):
                select_learning_prompts.select_prompts(source, output, [0], 1)

    def test_invalid_source_indices_are_rejected(self) -> None:
        """Do not silently choose a different question when a source row is unavailable."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.parquet"
            pd.DataFrame({"prompt": ["question"], "extra_info": [{"answer": "1"}]}).to_parquet(source)
            for rows in ([], [-1], [1]):
                with self.subTest(rows=rows), self.assertRaises(ValueError):
                    select_learning_prompts.select_prompts(source, root / "out.parquet", rows, 2)

    def test_acceptance_is_opt_in_and_preserves_scoring_and_learning_rate(self) -> None:
        """Enabling evidence gates must not inject rewards or enlarge optimizer steps."""
        base = Path(configure_run.__file__).parent.parent / "configs/qwen3_moe_4l_rl.yaml"
        with patch.dict(os.environ, {"RUN_MODEL_PATH": "/models/full"}, clear=True):
            ordinary = configure_run.configure(yaml.safe_load(base.read_text()), 4)
            self.assertFalse(ordinary["train"]["learning_gate"]["enabled"])
        with patch.dict(os.environ, {"RUN_MODEL_PATH": "/models/full", "REQUIRE_LEARNING": "true",
                                    "LEARNING_PROMPT_ROWS": "2,0"}, clear=True):
            config = configure_run.configure(yaml.safe_load(base.read_text()), 4)
        self.assertEqual(config["agentic"]["environment"], "gsm8k_tools")
        self.assertEqual(config["agentic"]["module_path"], "examples.gsm8k.agent")
        self.assertEqual(config["train"]["optimizer"], ordinary["train"]["optimizer"])
        self.assertTrue(config["train"]["learning_gate"]["require_mixed_rewards"])
        self.assertEqual(config["train"]["learning_gate"]["max_step"], 1)

    def test_acceptance_rejects_one_step_or_single_candidate(self) -> None:
        """The following-policy check and a within-prompt reward difference need both."""
        base = Path(configure_run.__file__).parent.parent / "configs/qwen3_moe_4l_rl.yaml"
        for settings in ({"STEPS": "1"}, {"CANDIDATES": "1"}):
            with self.subTest(settings=settings), patch.dict(
                    os.environ, {"RUN_MODEL_PATH": "/models/full", "REQUIRE_LEARNING": "true",
                                 **settings}, clear=True), self.assertRaisesRegex(ValueError, ">=2"):
                configure_run.configure(yaml.safe_load(base.read_text()), 4)
