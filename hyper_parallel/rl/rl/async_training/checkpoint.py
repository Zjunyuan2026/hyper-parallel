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
"""Commit asynchronous source/buffer state only after the model checkpoint is complete."""
import hashlib
import json
import logging
import os
import pickle
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping


logger = logging.getLogger(__name__)


def configuration_fingerprint(config: Mapping[str, Any]) -> str:
    """Identify learning and data semantics while allowing a longer resumed run."""
    value = deepcopy(dict(config))
    value.get("async_training", {}).pop("audit_dir", None)
    value.pop("logging", None)
    value.pop("evaluation", None)
    for key in ("train_path", "test_path"):
        path = value.get("data", {}).get(key)
        if path and Path(path).is_file():
            digest = hashlib.sha256()
            with Path(path).open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            value["data"][key + "_sha256"] = digest.hexdigest()
    value["train"].pop("checkpoint", None)
    value["train"].pop("max_steps", None)
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


class AsyncCheckpoint:
    """Layer a final RL commit marker over the existing model/optimizer persistence."""
    schema = 1

    @staticmethod
    def resolve_path(config: Mapping[str, Any]) -> str | None:
        """Honor an explicit path, otherwise select the latest committed async snapshot.

        Corrupted or incompatible committed snapshots fail during restore rather
        than silently rewinding training. Missing completion files are skipped.
        """
        checkpoint = config["train"]["checkpoint"]
        if checkpoint.get("load_path") is not None:
            return str(checkpoint["load_path"])
        root = Path(checkpoint["output_dir"])
        candidates = sorted(
            (path for path in root.glob("step_*") if path.is_dir() and path.name[5:].isdigit()),
            key=lambda path: int(path.name[5:]), reverse=True,
        )
        for directory in candidates:
            if all((directory / name).is_file() for name in (
                    "checkpoint_complete.json", "async_complete.json", "async_state.pkl")):
                logger.info("Automatically resuming asynchronous training from %s", directory)
                return str(directory)
            logger.warning("Skipping incomplete asynchronous checkpoint: %s", directory)
        logger.info("No complete asynchronous checkpoint in %s; starting a fresh run", root)
        return None

    @classmethod
    def load(cls, directory: str | None, config: Mapping[str, Any]) -> dict | None:
        """Read trusted local state, rejecting a partial or incompatible checkpoint."""
        if directory is None:
            return None
        root = Path(directory)
        manifest = json.loads((root / "async_complete.json").read_text())
        data = (root / "async_state.pkl").read_bytes()
        if manifest.get("schema") != cls.schema or manifest.get("sha256") != hashlib.sha256(data).hexdigest():
            raise ValueError("Incomplete or corrupted asynchronous checkpoint")
        if manifest.get("configuration") != configuration_fingerprint(config):
            raise ValueError("Asynchronous checkpoint learning/data configuration differs")
        state = pickle.loads(data)
        if state["learner_update"] != manifest["learner_update"]:
            raise ValueError("Checkpoint step identity differs")
        if config["train"].get("max_steps", state["learner_update"]) < state["learner_update"]:
            raise ValueError("Resume max_steps is below the checkpoint update")
        if state["policy_version"] != state["source_update"] + 1:
            raise ValueError("Invalid restored policy/source mapping")
        return state

    @staticmethod
    def invalidate(directory: Path) -> None:
        """Hide an older RL marker before overwriting any component at this step."""
        (directory / "async_complete.json").unlink(missing_ok=True)

    @classmethod
    def commit(cls, directory: Path, state: dict, config: Mapping[str, Any]) -> None:
        """Atomically expose a checksum-linked RL snapshot after the model manifest."""
        if not (directory / "checkpoint_complete.json").is_file():
            raise RuntimeError("Cannot commit async state before the model checkpoint")
        data = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
        cls._write(directory / "async_state.pkl", data)
        manifest = {"schema": cls.schema, "sha256": hashlib.sha256(data).hexdigest(),
                    "learner_update": state["learner_update"],
                    "configuration": configuration_fingerprint(config)}
        cls._write(directory / "async_complete.json", json.dumps(manifest, sort_keys=True).encode())

    @staticmethod
    def _write(path: Path, data: bytes) -> None:
        """Flush a temporary file before atomically exposing its final name."""
        pending = path.with_name(path.name + ".tmp")
        with pending.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        pending.replace(path)
        # Persist the rename as well as the contents before reporting a commit.
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
