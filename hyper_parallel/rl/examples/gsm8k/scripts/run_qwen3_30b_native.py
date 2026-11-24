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
"""Run ordinary synchronous GRPO with native vLLM and no numerical consistency gate."""

import argparse
import json
import logging
import os
import time
from pathlib import Path

from examples.consistency.scripts.run_consistency import MetricsObserver, PhaseRecorder
from rl.config.loader import load_config
from rl.trainer.build import build_trainer


def run(config: dict, report: Path) -> None:
    """Record normal training metrics without requiring Actor/rollout probability equality."""
    if "consistency" in config or config["rollout"]["vllm"]["model_implementation"] != "native":
        raise ValueError("Native baseline requires native vLLM and no consistency section")
    report.parent.mkdir(parents=True, exist_ok=True)
    phases = PhaseRecorder(report.parent)
    trainer = phases.call("build_trainer", lambda: build_trainer(config))
    if len(trainer.train_dataloader.dataset) != 6400:
        raise ValueError("Native baseline requires the sampled 6400-question training split")
    phases.next_step = lambda: trainer.state.global_step + 1
    observer = MetricsObserver(trainer._tracker, report.parent)
    trainer._tracker = observer
    started = time.monotonic()
    trainer.train()

    def finish() -> None:
        if int(os.environ.get("RANK", "0")) == 0 and len(observer.rows) != config["train"]["max_steps"]:
            raise RuntimeError("Native training metrics are incomplete")
        if trainer.state.global_step != config["train"]["max_steps"]:
            raise RuntimeError("Native training stopped before completing requested steps")
        # Only global rank zero writes MetricsObserver rows and the shared report.
        if observer.rows:
            result = {"passed": True, "completed_steps": trainer.state.global_step,
                      "strict_consistency": False, "model_implementation": "native",
                      "training_seconds_including_evaluation_and_checkpoint": time.monotonic() - started,
                      "scope": "Native GRPO completion only; no train/rollout numerical parity assertion"}
            report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    finish()


def main() -> None:
    """Load the materialized native configuration and run the production trainer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--report", type=Path, required=True)
    args, overrides = parser.parse_known_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run(load_config(args.config, overrides), args.report)


if __name__ == "__main__":
    main()
