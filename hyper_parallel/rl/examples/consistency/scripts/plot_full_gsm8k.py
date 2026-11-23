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
"""Export eight formal GSM8K figures with explicit completion evidence."""

import argparse
import json
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from plot_training import draw_panel, draw_series, read_rows, run_status


KINDS = ("reward", "loss", "consistency", "validation", "step-time", "stage-time", "coverage", "overview")


def render(directory: Path, output: Path) -> dict:
    """Keep missing results missing and write both PNG and SVG for every panel."""
    rows = read_rows(directory / "steps.jsonl")
    status = run_status(directory, rows)
    formal_path = directory / "formal-summary.json"
    formal = json.loads(formal_path.read_text()) if formal_path.is_file() else {}
    accepted = status["accepted"] and formal.get("passed") is True
    validations = sorted((json.loads(path.read_text()) for path in directory.glob("validation-step*.json")),
                         key=lambda row: row["step"])
    coverage_path = directory / "coverage.json"
    coverage = json.loads(coverage_path.read_text()) if coverage_path.is_file() else []
    output.mkdir(parents=True, exist_ok=True)

    def panel(axis, kind):
        if kind in ("reward", "loss", "consistency"):
            draw_panel(axis, kind, rows, status)
        elif kind == "validation":
            draw_series(axis, validations, (("validation/accuracy", "Held-out accuracy", "#2563eb"),))
            axis.set_ylim(0, 1)
            axis.set_title("Validation (747 questions; greedy)")
        elif kind == "step-time":
            draw_series(axis, rows, (("perf/time_per_step", "Training step seconds", "#2563eb"),))
            axis.set_title("Training time (evaluation/checkpoint excluded)")
        elif kind == "stage-time":
            fields = (("timing_s/gen", "Generation", "#2563eb"),
                      ("timing_s/old_log_prob", "Recompute", "#059669"),
                      ("timing_s/update_actor", "Actor update", "#ea580c"),
                      ("timing_s/weight_sync", "Weight sync", "#7c3aed"))
            draw_series(axis, rows, fields)
            axis.set_title("Stage time (seconds)")
        elif kind == "coverage":
            axis.plot([row["step"] for row in coverage], [row["cumulative_questions"] for row in coverage])
            axis.axhline(6726, color="#059669", linestyle="--", label="6726 training questions")
            axis.set_title("Cumulative training coverage")
            axis.legend(fontsize=8)
        axis.set_xlabel("Optimizer step")
        axis.grid(alpha=0.2)

    for kind in KINDS:
        dashboard = kind == "overview"
        figure = Figure(figsize=(16, 12) if dashboard else (11, 5), layout="constrained")
        FigureCanvasAgg(figure)
        axes = figure.subplots(4, 2).flatten() if dashboard else [figure.subplots()]
        for axis, selected in zip(axes, KINDS[:-1] if dashboard else (kind,)):
            panel(axis, selected)
        if dashboard:
            tests = [row for row in formal.get("evaluations", []) if row["split"] == "test"]
            accuracy = tests[0].get("validation/accuracy") if len(tests) == 1 else None
            axes[-1].axis("off")
            axes[-1].text(0.05, 0.8, "Official final test: 1319 questions\n"
                          + (f"Accuracy: {accuracy:.2%}" if accuracy is not None else "No complete test evidence")
                          + "\nFinal checkpoint: step 50\nTraining: 6726; held-out validation: 747",
                          transform=axes[-1].transAxes, fontsize=12)
        figure.suptitle(("COMPLETE" if accepted else "INCOMPLETE / FAILED")
                       + " ? GSM8K, full MoE, 4 nodes, micro-batch 1")
        for extension in ("png", "svg"):
            figure.savefig(output / f"{kind}.{extension}", dpi=150, facecolor="white")
        figure.clear()
    result = {"passed": accepted, "consistency": status, "figures": list(KINDS),
              "formats": ["png", "svg"], "formal": formal}
    (output / "plot-summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    """Render one formal run using CPU-only Matplotlib."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = render(args.run_directory, args.output or args.run_directory / "plots")
    print(json.dumps({"passed": result["passed"], "figures": result["figures"]}), flush=True)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
