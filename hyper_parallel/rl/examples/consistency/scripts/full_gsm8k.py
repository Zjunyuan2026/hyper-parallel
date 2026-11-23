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
"""One finite GSM8K epoch, held-out evaluation and coverage evidence for the A3 example."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Iterator

import pandas as pd
import torch.distributed as dist
from transformers import AutoTokenizer

from rl.data.experience import pad_rollout_responses
from rl.data.prompts import PROMPT_INSTRUCTION, PromptDataset, _normalize_prompt


SEED = 20261123
TRAIN_ROWS = 6726
VALIDATION_ROWS = 747
TEST_ROWS = 1319
GLOBAL_BATCH = 136
STEPS = 50


def epoch_batches(size: int = TRAIN_ROWS) -> list[list[int]]:
    """Visit every row once, retaining the partial final global batch."""
    indices = list(range(size))
    random.Random(SEED).shuffle(indices)
    return [indices[start:start + GLOBAL_BATCH] for start in range(0, size, GLOBAL_BATCH)]


def prepare(source: Path, destination: Path, model: Path, max_prompt: int) -> dict:
    """Validate the complete input and tokenizer before writing an immutable split."""
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Formal data destination must be empty; existing splits are never overwritten")
    train_path, test_path = source / "train.parquet", source / "test.parquet"
    train, test = pd.read_parquet(train_path), pd.read_parquet(test_path)
    if (len(train), len(test)) != (TRAIN_ROWS + VALIDATION_ROWS, TEST_ROWS):
        raise ValueError(f"Expected official GSM8K split sizes 7473/1319, got {len(train)}/{len(test)}")
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True, trust_remote_code=False)
    prompt_maxima = {}
    question_hashes = {}
    for split, frame in (("train", train), ("test", test)):
        hashes, longest = [], 0
        for index, record in enumerate(frame.to_dict("records")):
            question, prompt = _normalize_prompt(record["prompt"], index, PROMPT_INSTRUCTION)
            encoded = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                    add_generation_prompt=True, tokenize=True,
                                                    enable_thinking=False, truncation=False)
            longest = max(longest, len(encoded))
            hashes.append(hashlib.sha256(question.encode()).hexdigest())
        if len(hashes) != len(set(hashes)):
            raise ValueError(f"Duplicate questions in official {split} source")
        if longest > max_prompt:
            raise ValueError(f"{split} prompt needs {longest} tokens, exceeding {max_prompt}; raise --max-prompt")
        prompt_maxima[split] = longest
        question_hashes[split] = set(hashes)
    if question_hashes["train"] & question_hashes["test"]:
        raise ValueError("Training source overlaps official test questions")
    indices = list(range(len(train)))
    random.Random(SEED).shuffle(indices)
    train_ids, validation_ids = indices[:TRAIN_ROWS], indices[TRAIN_ROWS:]
    frames = {"train": train.iloc[train_ids].copy(), "validation": train.iloc[validation_ids].copy(),
              "test": test.copy()}
    source_ids = {"train": train_ids, "validation": validation_ids, "test": list(range(TEST_ROWS))}
    destination.mkdir(parents=True, exist_ok=True)
    for split, frame in frames.items():
        frame["formal_source_id"] = [f"{'test' if split == 'test' else 'train'}:{index}"
                                     for index in source_ids[split]]
        frame.to_parquet(destination / f"{split}.parquet", index=False)
    manifest = {"seed": SEED, "counts": {key: len(value) for key, value in frames.items()},
                "source_sha256": {"train": hashlib.sha256(train_path.read_bytes()).hexdigest(),
                                  "test": hashlib.sha256(test_path.read_bytes()).hexdigest()},
                "split_sha256": {key: hashlib.sha256((destination / f"{key}.parquet").read_bytes()).hexdigest()
                                 for key in frames},
                "source_indices": source_ids, "max_prompt_tokens": prompt_maxima,
                "epoch_batch_sizes": [len(batch) for batch in epoch_batches()]}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


class FiniteEpochLoader:
    """A checkpointable finite schedule; exhaustion must never silently repeat an epoch."""

    def __init__(self, dataset: Any, collate: Any, rank: int, size: int) -> None:
        """Bind one logical DP shard of the fixed training split."""
        if len(dataset) != TRAIN_ROWS or size != 8 or not 0 <= rank < size:
            raise ValueError("Formal GSM8K requires 6726 training rows and eight logical DP ranks")
        self.dataset, self.collate, self.rank, self.size = dataset, collate, rank, size
        self.batches = epoch_batches()
        self.cursor = 0

    def __len__(self) -> int:
        """Return all global steps, including the retained tail."""
        return len(self.batches)

    def __iter__(self) -> Iterator[Any]:
        """Yield the remaining local batches without wrapping the dataset."""
        while self.cursor < len(self.batches):
            indices = self.batches[self.cursor][self.rank::self.size]
            self.cursor += 1
            yield self.collate([self.dataset[index] for index in indices])

    def set_epoch(self, epoch: int) -> None:
        """Reject implicit re-entry that would duplicate training questions."""
        if epoch != 0:
            raise ValueError("Formal GSM8K is exactly one finite epoch")

    def state_dict(self) -> dict:
        """Persist the next batch and the identity of its deterministic shard."""
        return {"cursor": self.cursor, "rank": self.rank, "size": self.size, "seed": SEED,
                "dataset_size": len(self.dataset)}

    def load_state_dict(self, state: dict) -> None:
        """Restore only a compatible schedule at a completed optimizer boundary."""
        expected = self.state_dict()
        if any(state.get(key) != value for key, value in expected.items() if key != "cursor"):
            raise ValueError("Formal data schedule differs from checkpoint")
        cursor = state.get("cursor")
        if isinstance(cursor, bool) or not isinstance(cursor, int) or not 0 <= cursor <= STEPS:
            raise ValueError("Invalid formal data cursor")
        self.cursor = cursor


class FormalRun:
    """Attach task-owned auditing and evaluation to the existing synchronous trainer."""

    def __init__(self, trainer: Any, config: dict, directory: Path, shared: Path) -> None:
        """Attach example-specific scheduling to the trainer's existing role contracts."""
        self.trainer, self.directory, self.shared = trainer, directory, shared
        self.rank = dist.get_rank()
        if trainer.state.max_steps != STEPS or trainer.evaluator is None:
            raise ValueError("Formal run requires 50 steps and a validation evaluator")
        self.loader = FiniteEpochLoader(trainer.train_dataloader.dataset, trainer.train_dataloader.collate_fn,
                                        int(trainer.parallel_dims.dp_rank), int(trainer.parallel_dims.dp_size))
        trainer.train_dataloader = self.loader
        trainer.sampler = self.loader
        trainer.checkpoints.resources.train_dataloader = self.loader
        self.coverage = []
        self.evaluations = []
        test_config = {**config["data"], "test_path": str(shared / "data/test.parquet")}
        self.test_dataset = PromptDataset.from_config(test_config, trainer.checkpoints.resources.tokenizer, split="test")
        if len(trainer.evaluator.dataset) != VALIDATION_ROWS or len(self.test_dataset) != TEST_ROWS:
            raise ValueError("Formal validation/test split counts differ from manifest")
        self._install_rollout_audit()
        self._install_tail_padding()
        self._install_evaluation()

    def _install_rollout_audit(self) -> None:
        original = self.trainer._generate_training_rollout

        def generate(batch, timings):
            step = self.trainer.state.global_step + 1
            local = batch["sample_indices"] if self.trainer.rollout_engine.is_request_owner else None
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, local)
            owners = [rows for rows in gathered if rows is not None]
            actual = [index for rows in owners for index in rows]
            expected = self.loader.batches[step - 1]
            if len(owners) != 8 or sorted(actual) != sorted(expected):
                raise ValueError("Actual DP question coverage differs from the finite epoch schedule")
            self.coverage.append({"step": step, "indices": actual, "questions": len(actual),
                                  "cumulative_questions": sum(row["questions"] for row in self.coverage) + len(actual)})
            result = original(batch, timings)
            self.trainer.state.consumed_samples = self.coverage[-1]["cumulative_questions"]
            self.trainer._run_rank_synchronized("write formal coverage", self._write_coverage)
            return result

        self.trainer._generate_training_rollout = generate

    def _write_coverage(self) -> None:
        if self.rank == 0:
            self.directory.joinpath("coverage.json").write_text(json.dumps(self.coverage) + "\n", encoding="utf-8")

    def _install_tail_padding(self) -> None:
        original = self.trainer.targets.prepare

        def prepare_targets(rollout, diagnostics, timings):
            if self.trainer.state.global_step + 1 == STEPS:
                rollout = pad_rollout_responses(rollout, 32)
                self.trainer.actor._response_mini_batch_size = 32
            return original(rollout, diagnostics, timings)

        self.trainer.targets.prepare = prepare_targets

    def _install_evaluation(self) -> None:
        evaluator = self.trainer.evaluator
        original = evaluator.run

        def evaluate(step):
            metrics, samples = self._evaluate(original, step, "validation", VALIDATION_ROWS)
            if step == STEPS:
                previous = evaluator.dataset
                evaluator.dataset = self.test_dataset
                try:
                    test_metrics, _ = self._evaluate(original, step, "test", TEST_ROWS)
                finally:
                    evaluator.dataset = previous
                metrics.update({key.replace("validation/", "test/", 1): value for key, value in test_metrics.items()})
            return metrics, samples

        evaluator.run = evaluate

    def _evaluate(self, callback: Any, step: int, split: str, count: int) -> tuple[dict, list]:
        started = time.perf_counter()
        metrics, samples = callback(step)
        record = {"step": step, "split": split, "elapsed_seconds": time.perf_counter() - started, **metrics}

        def accept():
            if self.rank != 0:
                return
            accuracy = metrics.get("validation/accuracy")
            if metrics.get("validation/total") != count or not isinstance(accuracy, (int, float)):
                raise ValueError(f"Missing complete {split} accuracy evidence")
            if not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
                raise ValueError(f"Invalid {split} accuracy")
            self.directory.joinpath(f"{split}-step{step}.json").write_text(json.dumps(record) + "\n", encoding="utf-8")

        self.trainer._run_rank_synchronized(f"accept {split} evaluation", accept)
        self.evaluations.append(record)
        return metrics, samples

    def finish(self) -> None:
        """Require exact coverage, six validations, final test and durable checkpoints."""
        if self.rank != 0:
            return
        seen = [index for row in self.coverage for index in row["indices"]]
        if len(seen) != TRAIN_ROWS or sorted(seen) != list(range(TRAIN_ROWS)):
            raise ValueError("Formal training did not cover every training question exactly once")
        validations = [row["step"] for row in self.evaluations if row["split"] == "validation"]
        tests = [row["step"] for row in self.evaluations if row["split"] == "test"]
        if validations != [0, 10, 20, 30, 40, 50] or tests != [50]:
            raise ValueError("Formal validation/final test schedule is incomplete")
        for step in (10, 20, 30, 40, 50):
            marker = self.shared / f"checkpoints/step_{step}/checkpoint_complete.json"
            if not marker.is_file() or json.loads(marker.read_text())["step"] != step:
                raise ValueError(f"Missing completed checkpoint at step {step}")
        export = self.shared / "checkpoints/step_50/hf"
        if not (export / "config.json").is_file() or not list(export.glob("*.safetensors")):
            raise ValueError("Final Hugging Face model export is incomplete")
        report = {"passed": True, "training_questions": len(seen), "unique_training_questions": len(set(seen)),
                  "evaluations": self.evaluations, "final_model": str(export),
                  "scope": "Exact one-epoch coverage and completed evaluations; no guaranteed accuracy improvement"}
        self.directory.joinpath("formal-summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Prepare all three data files on CPU before any training container starts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--max-prompt", type=int, default=512)
    args = parser.parse_args()
    result = prepare(args.source, args.destination, args.model, args.max_prompt)
    print("FORMAL_DATA_READY", json.dumps({key: result[key] for key in ("counts", "max_prompt_tokens")}), flush=True)


if __name__ == "__main__":
    main()
