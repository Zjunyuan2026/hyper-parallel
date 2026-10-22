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
"""Check trusted local async checkpoints against live post-restart evidence."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path

import torch
import yaml

from rl.async_training.protocol import decode_group


def events(root: Path, rank: int) -> list[dict]:
    """Read the complete trainer trace for one physical rank."""
    return [json.loads(line) for line in (root / "audit" / f"trainer_rank_{rank}.jsonl").read_text().splitlines()]


def checkpoint(root: Path) -> tuple[dict, dict]:
    """Require one completed checkpoint and verify its async payload checksum."""
    paths = list((root / "checkpoints").glob("step_*/async_complete.json"))
    if len(paths) != 1:
        raise ValueError(f"Expected one checkpoint under {root}, got {len(paths)}")
    directory = paths[0].parent
    manifest = json.loads(paths[0].read_text())
    payload = (directory / "async_state.pkl").read_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest["sha256"]:
        raise ValueError("Async checkpoint checksum differs")
    state = pickle.loads(payload)
    model = json.loads((directory / "checkpoint_complete.json").read_text())
    if model["step"] != state["learner_update"]:
        raise ValueError("Model/async step identities differ")
    buffer = state["buffer"]
    if buffer["next_identity"] != buffer["consumed_groups"] + buffer["dropped_groups"] + len(buffer["ready"]):
        raise ValueError("Checkpoint group accounting is inconsistent")
    return state, model


def verify(previous: Path, resumed: Path) -> dict:
    """Compare live loaded states and resumed consumption with the saved checkpoint."""
    before, model = checkpoint(previous)
    after, _ = checkpoint(resumed)
    rank_reports = []
    recipe = yaml.safe_load((previous / "base_config.yaml").read_text())
    candidates = recipe["rollout"]["num_return_sequences"]
    limit = recipe["async_training"]["max_group_bytes"]
    retained = {}
    for payload in before["buffer"]["ready"]:
        group = decode_group(payload, candidates, limit)
        for trajectory in group.trajectories:
            retained[trajectory.trajectory_id] = trajectory
    selected_rows = set()
    for rank in range(model["world_size"]):
        saved = next(row for row in events(previous, rank) if row["event"] == "checkpoint_state")
        loaded = next(row for row in events(resumed, rank) if row["event"] == "restore_complete")
        if loaded["update"] != before["learner_update"] or saved["digests"] != loaded["digests"]:
            raise ValueError(f"Live runtime restoration differs on rank {rank}")
        if loaded["optimizer_steps"] != before["optimizer_steps"]:
            raise ValueError("Optimizer step counter was not restored")
        batch = torch.load(resumed / "audit" / f"batch_step_{before['learner_update'] + 1}_rank_{rank}.pt",
                           map_location="cpu", weights_only=True)
        for index, identity in enumerate(batch["trajectory_ids"]):
            trajectory = retained[identity]
            selected_rows.add(identity)
            if batch["policy_versions"][index] != trajectory.policy_version:
                raise ValueError("Restored trajectory policy identity changed")
            if float(batch["rewards"][index]) != trajectory.reward:
                raise ValueError("Restored trajectory reward changed")
            mask = trajectory.action_mask[1:].bool()
            expected = trajectory.rollout_log_probs[mask]
            actual = batch["rollout_log_probs"][index, :mask.numel()][mask]
            torch.testing.assert_close(actual, expected, atol=0., rtol=0.)
        rank_reports.append({"rank": rank, "all_state_digests_equal": True,
                             "components": list(saved["digests"]), "digests": saved["digests"]})
    if selected_rows != set(retained):
        raise ValueError("The resumed batch does not consume exactly the saved ready trajectories")
    root_events = events(resumed, 0)
    coordinator = [json.loads(line) for line in
                   (resumed / "audit/coordinator_rank_0.jsonl").read_text().splitlines()]
    admitted = [row["identity"] for row in coordinator if row["event"] == "admit"]
    expected_ids = list(range(before["buffer"]["next_identity"], after["buffer"]["next_identity"]))
    if admitted != expected_ids:
        raise ValueError("Sampling identities restarted or skipped after recovery")
    source = [json.loads(line) for line in
              (resumed / "audit/rollouter_rank_0.jsonl").read_text().splitlines()]
    read_ids = [row["identity"] for row in source if row["event"] == "source_read"]
    if read_ids != admitted:
        raise ValueError("Source reads do not match post-restart admissions")
    old_epoch, new_epoch = before["source"]["epoch"], after["source"]["epoch"]
    old_cursor = before["source"]["loader"]["_num_yielded"]
    new_cursor = after["source"]["loader"]["_num_yielded"]
    if old_epoch != new_epoch:
        raise ValueError("This acceptance helper requires continuation within the same dataset epoch")
    if new_cursor - old_cursor != len(admitted):
        raise ValueError("DataLoader cursor did not continue from its saved position")
    if not any(row["event"] == "run_end" and row.get("completed") for row in root_events):
        raise ValueError("Resumed training did not finish")
    return {"from_update": before["learner_update"], "to_update": after["learner_update"],
            "rank_state_checks": rank_reports, "retained_trajectories_checked": len(retained),
            "new_sampling_identities": admitted, "source_epoch": old_epoch,
            "source_cursor_before": old_cursor, "source_cursor_after": new_cursor,
            "optimizer_steps_before": before["optimizer_steps"], "optimizer_steps_after": after["optimizer_steps"]}


def main() -> None:
    """Compare two trusted local run directories and persist the result."""
    parser = argparse.ArgumentParser()
    parser.add_argument("previous", type=Path)
    parser.add_argument("resumed", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = verify(arguments.previous, arguments.resumed)
    arguments.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
