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
"""Async commit marker, configuration identity and corruption handling."""
import tempfile
import unittest
from pathlib import Path

from rl.async_training.checkpoint import AsyncCheckpoint


class TestAsyncCheckpoint(unittest.TestCase):
    """A model checkpoint alone must never imply a complete async snapshot."""

    def setUp(self):
        self.config = {"train": {"max_steps": 4, "checkpoint": {}}, "data": {"train_path": "a"}}
        self.state = {"learner_update": 2, "source_update": 2, "policy_version": 3,
                      "source": {"cursor": 7}, "buffer": {}}

    def test_commit_requires_model_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError):
                AsyncCheckpoint.commit(Path(directory), self.state, self.config)

    def test_auto_resume_selects_latest_complete_and_honors_explicit_path(self):
        with tempfile.TemporaryDirectory() as directory:
            self.config["train"]["checkpoint"]["output_dir"] = directory
            self.config["train"]["max_steps"] = 20
            self.assertIsNone(AsyncCheckpoint.resolve_path(self.config))
            for step in (2, 10, 11):
                root = Path(directory) / f"step_{step}"
                root.mkdir()
                (root / "checkpoint_complete.json").write_text("{}")
                state = dict(self.state, learner_update=step, source_update=step, policy_version=step + 1)
                AsyncCheckpoint.commit(root, state, self.config)
            AsyncCheckpoint.invalidate(root)
            selected = AsyncCheckpoint.resolve_path(self.config)
            self.assertEqual(Path(selected).name, "step_10")
            self.assertEqual(AsyncCheckpoint.load(selected, self.config)["learner_update"], 10)
            self.config["train"]["checkpoint"]["load_path"] = str(Path(directory) / "step_2")
            self.assertEqual(Path(AsyncCheckpoint.resolve_path(self.config)).name, "step_2")
            (Path(selected) / "async_state.pkl").write_bytes(b"corrupt")
            self.config["train"]["checkpoint"].pop("load_path")
            with self.assertRaises(ValueError):
                AsyncCheckpoint.load(AsyncCheckpoint.resolve_path(self.config), self.config)

    def test_roundtrip_and_longer_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "checkpoint_complete.json").write_text("{}")
            AsyncCheckpoint.commit(root, self.state, self.config)
            self.config["train"]["max_steps"] = 10
            self.assertEqual(AsyncCheckpoint.load(directory, self.config), self.state)
            self.config["data"]["train_path"] = "other"
            with self.assertRaises(ValueError):
                AsyncCheckpoint.load(directory, self.config)

    def test_replaced_dataset_is_rejected_at_the_same_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data.parquet"
            data.write_bytes(b"original source")
            self.config["data"]["train_path"] = str(data)
            (root / "checkpoint_complete.json").write_text("{}")
            AsyncCheckpoint.commit(root, self.state, self.config)
            data.write_bytes(b"replaced source")
            with self.assertRaisesRegex(ValueError, "configuration differs"):
                AsyncCheckpoint.load(directory, self.config)

    def test_corrupted_payload_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "checkpoint_complete.json").write_text("{}")
            AsyncCheckpoint.commit(root, self.state, self.config)
            with (root / "async_state.pkl").open("ab") as handle:
                handle.write(b"corrupt")
            with self.assertRaises(ValueError):
                AsyncCheckpoint.load(directory, self.config)

    def test_invalidate_removes_only_commit_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "checkpoint_complete.json").write_text("{}")
            AsyncCheckpoint.commit(root, self.state, self.config)
            AsyncCheckpoint.invalidate(root)
            self.assertFalse((root / "async_complete.json").exists())
            self.assertTrue((root / "async_state.pkl").exists())
