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
"""Record the upstream synchronous trainer for a four-node GSM8K experiment."""

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import yaml


class MetricsObserver:
    """Record original metrics without changing training or adding collectives."""

    def __init__(self, tracker, directory: Path):
        self.tracker, self.directory = tracker, directory
        self.rows = {}

    def __getattr__(self, name):
        return getattr(self.tracker, name)

    def log(self, metrics, *, step, **kwargs):
        self.tracker.log(metrics, step=step, **kwargs)
        if int(os.environ.get("RANK", "0")) != 0:
            return
        if "validation/accuracy" in metrics:
            with (self.directory / "evaluations.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"step": step, **metrics}, default=str) + "\n")
        if step <= 0:
            return
        self.rows.setdefault(step, {"step": step}).update(metrics)
        with (self.directory / "steps.jsonl").open("w", encoding="utf-8") as stream:
            for key in sorted(self.rows):
                stream.write(json.dumps(self.rows[key], default=str) + "\n")
        logging.info("STEP_SUMMARY %s", json.dumps(self.rows[step], default=str))


def main():
    """Run in each torchrun process; only rank zero exports CPU figures."""
    # The trainer imports optional NPU dependencies; reporting utilities remain CPU-importable.
    from rl.config.loader import load_config
    from rl.trainer.build import build_trainer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    args, overrides = parser.parse_known_args()
    config = load_config(args.config, overrides)
    output = Path(os.environ["HYPER_RL_FAILURE_DIR"])
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    started = time.monotonic()
    trainer = build_trainer(config)
    observer = MetricsObserver(trainer._tracker, output)
    trainer._tracker = observer
    trainer.train()
    if int(os.environ.get("RANK", "0")) != 0:
        return
    rows = list(observer.rows.values())
    steps = config["train"]["max_steps"]
    native = "consistency" not in config
    valid = len(rows) == steps and all(
        row.get("training/rollout_probs_diff_valid") == 1
        and row.get("training/log_probs_valid_tokens", 0) > 0
        and math.isfinite(row.get("training/rollout_log_probs_diff_max", math.nan))
        and (native or row["training/rollout_log_probs_diff_max"] == 0) for row in rows)
    report = {"passed": valid, "completed_steps": len(rows), "requested_steps": steps,
              "strict_consistency": not native, "model_implementation": "native" if native else "unified"}
    name = "native-training.json" if native else "consistency.json"
    (output / name).write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    root = output.parent.parent
    summary = {**report, "native_baseline": native,
               "elapsed_launch_to_collection_seconds": time.monotonic() - started,
               "scope": "Rank-zero trainer lifecycle; CPU plotting excluded"}
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not valid:
        raise RuntimeError("Missing diagnostics or nonzero unified logp differences; inspect steps.jsonl")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "consistency/scripts"))
    # Plot dependencies are only used by the reporting rank after training releases device resources.
    from plot_comparison import render
    # Single-run rendering supports either label without replaying training.
    if native:
        render(None, root, output / "native-plots", steps=steps)
    else:
        from plot_training import render as render_training
        render_training(output, output / "plots")
    reference = os.environ.get("GSM8K_COMPARE_WITH")
    if native and reference:
        render(Path(reference), root, output / "comparison-plots", steps=steps)


if __name__ == "__main__":
    main()
