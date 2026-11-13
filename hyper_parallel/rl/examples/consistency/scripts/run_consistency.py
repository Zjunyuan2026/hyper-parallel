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
"""Run the normal RL loop with a strict pre-update train/rollout probability gate."""

import argparse
import csv
import json
import logging
import math
import os
import time
import traceback
from pathlib import Path
from typing import Any

import yaml

from rl.config.loader import load_config
from rl.trainer.build import build_trainer
from rl.trainer.distributed import run_synchronized


def check_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Require nonempty, finite and exactly equal sampled-token log probabilities."""
    keys = ("rollout_probs_diff_valid", "log_probs_valid_tokens", "rollout_log_probs_diff_max",
            "rollout_log_probs_diff_mean", "actor_log_probs_mean", "rollout_log_probs_mean")
    observed = {key: float(metrics[f"training/{key}"]) for key in keys}
    if (not all(math.isfinite(value) for value in observed.values())
            or observed["rollout_probs_diff_valid"] != 1
            or observed["log_probs_valid_tokens"] <= 0
            or observed["rollout_log_probs_diff_max"] != 0
            or observed["rollout_log_probs_diff_mean"] != 0):
        raise RuntimeError(f"Train/rollout consistency failed before optimizer update: {observed}")
    return observed


def check_metrics_on_all_ranks(metrics: dict[str, float]) -> dict[str, float]:
    """Validate rank-zero global diagnostics and propagate failures to every rank."""
    rank = int(os.environ.get("RANK", "0"))
    return run_synchronized(
        "train/rollout consistency",
        lambda: check_metrics(metrics) if rank == 0 else {},
    )


def run(config: dict[str, Any], report_path: Path) -> None:
    """Observe production diagnostics without changing tokens, rewards or numerical paths."""
    requested_steps = config["train"]["max_steps"]
    if isinstance(requested_steps, bool) or not isinstance(requested_steps, int) or requested_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    if config["logging"]["log_steps"] != 1:
        raise ValueError("Strict consistency requires log_steps=1")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if int(os.environ.get("RANK", "0")) == 0 and report_path.exists():
        report_path.unlink()
    if int(os.environ.get("RANK", "0")) == 0:
        report_path.with_name("config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    phases = PhaseRecorder(report_path.parent)
    trainer = phases.call("build_trainer", lambda: build_trainer(config))
    phases.next_step = lambda: trainer.state.global_step + 1
    prepare = trainer.targets.prepare
    checks = []

    def _checked_prepare(rollout, collect_diagnostics, timings):
        if rollout.rollout_log_probs is None:
            raise RuntimeError("Independent rollout sampling log probabilities are required")
        experience, metrics = prepare(rollout, collect_diagnostics, timings)
        observed = check_metrics_on_all_ranks(metrics)
        checks.append({"step": trainer.state.global_step + 1, **observed})
        if int(os.environ.get("RANK", "0")) == 0:
            logging.info("CONSISTENCY_CHECK %s", json.dumps(checks[-1]))
        return experience, metrics

    # This example adds an observation gate at the existing pre-update boundary.
    trainer.targets.prepare = _checked_prepare
    for owner, method, stage in ((trainer.rollout_manager, "generate", "codex_or_rollout"),
                                 (trainer.targets, "prepare", "recompute_and_consistency"),
                                 (trainer.actor, "update", "actor_update"),
                                 (trainer.publication, "publish", "weight_publish")):
        original = getattr(owner, method)
        def observed(*args, _original=original, _stage=stage, **kwargs):
            return phases.call(_stage, lambda: _original(*args, **kwargs))
        setattr(owner, method, observed)
    observer = MetricsObserver(trainer._tracker, report_path.parent)
    trainer._tracker = observer
    trainer.train()
    if (trainer.state.global_step != requested_steps
            or [check["step"] for check in checks] != list(range(1, requested_steps + 1))):
        raise RuntimeError("The normal RL loop did not complete and verify every requested update")
    if int(os.environ.get("RANK", "0")) == 0:
        report = {"passed": True, "requested_steps": requested_steps,
                  "completed_steps": trainer.state.global_step, "checks": checks,
                  "metrics": observer.rows,
                  "training": config["train"]["accelerator"],
                  "rollout": config["rollout"]["vllm"],
                  "scope": "sampled-token log probabilities before each optimizer update"}
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        logging.info("CONSISTENCY_PASS report=%s", report_path)


class PhaseRecorder:
    """Record rank-local phase boundaries without extra synchronization."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.rank = int(os.environ.get("RANK", "0"))
        self.next_step = lambda: None

    def call(self, stage: str, callback: Any) -> Any:
        """Retain the original return value and exception unchanged."""
        started = time.monotonic()
        self.record(stage, "begin", 0)
        try:
            result = callback()
        except BaseException:
            self.record(stage, "failed", time.monotonic() - started)
            raise
        self.record(stage, "end", time.monotonic() - started)
        return result

    def record(self, stage: str, status: str, seconds: float) -> None:
        """Persist the last phase and a complete event stream for each rank."""
        event = {"rank": self.rank, "stage": stage, "status": status,
                 "step": self.next_step(), "wall_time_unix": time.time(), "elapsed_seconds": seconds}
        content = json.dumps(event)
        self.directory.joinpath(f"progress-rank{self.rank}.json").write_text(content + "\n")
        with self.directory.joinpath(f"events-rank{self.rank}.jsonl").open("a") as stream:
            stream.write(content + "\n")
        logging.info("PHASE_EVENT %s", content)


class MetricsObserver:
    """Persist the Trainer's original metrics without adding device operations."""

    def __init__(self, tracker: Any, directory: Path) -> None:
        self.tracker = tracker
        self.directory = directory
        self.rows: list[dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.tracker, name)

    def log(self, metrics: dict, *, step: int, **kwargs: Any) -> None:
        """Keep the original tracker and write rank-zero step metrics."""
        self.tracker.log(metrics, step=step, **kwargs)
        if int(os.environ.get("RANK", "0")) != 0:
            return
        row = {"step": step, "recorded_at_unix": time.time(), **metrics}
        self.rows.append(row)
        with (self.directory / "steps.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, default=str) + "\n")
        fields = sorted(set().union(*(item.keys() for item in self.rows)))
        with (self.directory / "steps.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.rows)
        logging.info("STEP_RESULT %s", json.dumps(row, default=str))
        important = ("reward/mean", "reward/std", "train/gradient_norm", "train/total_loss",
                     "perf/time_per_step", "training/log_probs_valid_tokens",
                     "training/rollout_log_probs_diff_max", "training/rollout_log_probs_diff_mean")
        logging.info("STEP_SUMMARY %s", json.dumps({"step": step, **{key: row.get(key) for key in important}}))


def main() -> None:
    """Load the explicit recipe and launch the ordinary synchronous trainer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--report", type=Path, default=Path("/results/run/consistency.json"))
    args, overrides = parser.parse_known_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    started = time.monotonic()
    rank = int(os.environ.get("RANK", "0"))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        args.report.unlink(missing_ok=True)
    summary = {"passed": False, "rank": rank, "stage": "configuration", "completed_steps": 0}
    try:
        config = load_config(args.config, overrides)
        summary.update(requested_steps=config["train"]["max_steps"], stage="build_or_train")
        if rank == 0:
            for name in ("steps.jsonl", "steps.csv"):
                args.report.with_name(name).unlink(missing_ok=True)
        logging.info("RUN_START rank=%d topology=%s rollout=%s", rank,
                     config["train"]["accelerator"], config["rollout"]["vllm"])
        run(config, args.report)
        summary.update(passed=True, stage="completed")
    except BaseException as error:
        summary.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        logging.exception("RUN_FAILED rank=%d stage=%s", rank, summary["stage"])
        raise
    finally:
        progress = args.report.with_name(f"progress-rank{rank}.json")
        if progress.exists():
            summary["last_phase"] = json.loads(progress.read_text())
        metrics_file = args.report.with_name("steps.jsonl")
        if rank == 0 and metrics_file.exists():
            rows = [json.loads(line) for line in metrics_file.read_text().splitlines()]
            summary["completed_steps"] = len(rows)
            warm_times = [row["perf/time_per_step"] for row in rows[1:] if "perf/time_per_step" in row]
            summary["mean_step_seconds_excluding_first"] = sum(warm_times) / len(warm_times) if warm_times else None
            summary["warm_step_samples"] = len(warm_times)
        summary["elapsed_seconds"] = time.monotonic() - started
        summary["scope"] = "sampled-token forward consistency; not a learning-quality assertion"
        destination = args.report.with_name("summary.json" if rank == 0 else f"failure-rank{rank}.json")
        if rank == 0 or not summary["passed"]:
            destination.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
