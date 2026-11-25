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
"""Compare two completed GSM8K runs using recorded metrics on CPU only."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter
from plot_training import moving_average, number, read_evaluations, read_rows

SERIES = (("Consistency", "#2563eb"), ("NO-Consistency", "#ea580c"))
DISPLAY_ANNOTATION = ("Training: TP4 DP8 EP4 CP2 SP=True\n"
                      "Rollout: TP2 DP8 EP16 PCP1 SP=False")
TIMES = (("Run\ntotal", None), ("Rollout", "timing_s/gen"),
         ("Actor\nupdate", "timing_s/update_actor"),
         ("Weight\nsync", "timing_s/weight_sync"))


def load_run(root: Path, expected_steps: int) -> dict:
    """Require a successful cluster run; retain first-step costs in total timings."""
    root = root.resolve()
    directory = root / "node0/run"
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    rows = read_rows(directory / "steps.jsonl")
    if summary.get("passed") is not True or len(rows) != expected_steps:
        raise ValueError(f"Expected a successful {expected_steps}-step cluster run: {root}")
    times = []
    for _, key in TIMES:
        values = ([number(summary, "elapsed_launch_to_collection_seconds")] if key is None
                  else [number(row, key) for row in rows])
        times.append(sum(values) if all(math.isfinite(v) and v >= 0 for v in values) else None)
    return {"root": str(root), "rows": rows, "evaluations": read_evaluations(directory, rows),
            "seconds": times}


def new_plot(title: str, ylabel: str) -> tuple:
    """Build an exportable 6:4 figure without a display server."""
    figure = Figure(figsize=(6, 4), layout="constrained")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    figure.suptitle(f"Qwen3-30B-A3B on GSM8K:\n{title}\n{DISPLAY_ANNOTATION}", fontsize=9)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.2)
    return figure, axis


def save(figure: Figure, output: Path, name: str) -> None:
    """Export each panel to both PNG and SVG with identical aspect ratio."""
    for extension in ("png", "svg"):
        figure.savefig(output / f"{name}.{extension}", dpi=180, facecolor="white")
    figure.clear()


def render(consistency: Path | None, native: Path, output: Path, window: int = 5, steps: int = 50) -> dict:
    """Overlay metrics without fabricating missing native sampling probabilities."""
    if consistency is not None and consistency.resolve() == native.resolve():
        raise ValueError("Select two different run directories")
    if window < 1 or steps < 1:
        raise ValueError("Window and expected steps must be positive")
    paths = (consistency, native) if consistency is not None else (native,)
    runs = [load_run(path, steps) for path in paths]
    series = SERIES if consistency is not None else SERIES[1:]
    suffix = "-comparison" if consistency is not None else ""
    output.mkdir(parents=True, exist_ok=True)
    report = {"runs": {label: {"root": run["root"], "seconds": dict(zip(
        (label.replace("\n", " ") for label, _ in TIMES), run["seconds"]))}
        for (label, _), run in zip(series, runs)}, "smoothing_window": window,
        "display_annotation": DISPLAY_ANNOTATION,
        "display_annotation_source": "User-specified display text; not inferred from recorded run configurations",
        "reward_std_scope": "Per-step response reward population std; not uncertainty across repeated runs",
        "time_scope": "All recorded steps including first. Run total is launch-to-collection wall time; "
                      "includes startup/evaluation/plotting/cleanup. Component bars cover training steps only; "
                      "the displayed bars are not additive.", "missing_logp": []}

    figure, axis = new_plot("Training Reward", "Reward")
    for (label, color), run in zip(series, runs):
        x = [row["step"] for row in run["rows"]]
        for key, metric, style, marker in (("reward/mean", "mean", "-", "o"),
                                           ("reward/std", "std", "--", "s")):
            values = [number(row, key) for row in run["rows"]]
            axis.plot(x, values, color=color, alpha=0.3, linewidth=0.7, linestyle=style,
                      marker=marker, markersize=2, label=f"{label} {metric} raw")
            axis.plot(x, moving_average(values, window), color=color, linewidth=1.6, linestyle=style,
                      label=f"{label} {metric} MA{window}")
    axis.set_xlabel("Training step")
    axis.set_xlim(1, steps)
    axis.set_ylim(-0.05, 1.05)
    axis.legend(fontsize=6, ncol=2)
    save(figure, output, "reward" + suffix)

    figure, axis = new_plot("Test Accuracy", "Accuracy (500 questions)")
    for index, ((label, color), run) in enumerate(zip(series, runs)):
        evaluations = [row for row in run["evaluations"]
                       if (row["step"] % 10 == 0 or row["step"] == steps) and 0 <= row["step"] <= steps]
        for row in evaluations:
            accuracy, total, correct = (number(row, key) for key in
                                        ("validation/accuracy", "validation/total", "validation/correct"))
            if not (total == 500 and math.isfinite(correct) and 0 <= correct <= total
                    and correct == int(correct) and math.isclose(accuracy, correct / total, abs_tol=1e-12)):
                raise ValueError(f"Invalid 500-question test measurement: {run['root']} step {row['step']}")
        if (not evaluations or evaluations[-1]["step"] != steps
                or not set(range(10, steps + 1, 10)).issubset({row["step"] for row in evaluations})):
            raise ValueError(f"Missing final or ten-step evaluation: {run['root']}")
        axis.plot([row["step"] for row in evaluations], [row["validation/accuracy"] for row in evaluations],
                  marker="o", color=color, label=label)
        final = evaluations[-1]
        axis.text(0.03, 0.08 + index * 0.07, f'{label}: {final["validation/accuracy"]:.2%} '
                  f'({int(final["validation/correct"])}/500)', color=color, transform=axis.transAxes, fontsize=9)
    axis.set_xticks(list(range(0, steps + 1, 10)))
    axis.set_xlabel("Training step (0 = before training)")
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    axis.set_ylim(0, 1)
    axis.legend(fontsize=8)
    save(figure, output, "test-accuracy" + suffix)

    figure, axis = new_plot("Maximum Train–Rollout Log-Probability Difference",
                            "Max |logp(train) - logp(rollout)|")
    for index, ((label, color), run) in enumerate(zip(series, runs)):
        values = [number(row, "training/rollout_log_probs_diff_max")
                  if number(row, "training/rollout_probs_diff_valid") == 1
                  and number(row, "training/log_probs_valid_tokens") > 0 else math.nan for row in run["rows"]]
        missing = [row["step"] for row, value in zip(run["rows"], values) if not math.isfinite(value)]
        if missing:
            report["missing_logp"].append({"label": label, "steps": missing})
            axis.text(0.03, 0.93 - index * 0.09, f"{label}: missing logp measurements ({len(missing)} steps)",
                      color=color, transform=axis.transAxes, fontsize=8)
        if any(math.isfinite(v) for v in values):
            axis.plot([row["step"] for row in run["rows"]], values, color=color, label=label)
    axis.set_xlabel("Training step")
    axis.set_xlim(1, steps)
    if axis.lines:
        axis.legend(loc="lower right", fontsize=8)
    save(figure, output, "logp-max" + suffix)

    figure, axis = new_plot("Runtime Breakdown", "Minutes")
    x = np.arange(len(TIMES))
    for index, ((label, color), run) in enumerate(zip(series, runs)):
        values = [v / 60 if v is not None else math.nan for v in run["seconds"]]
        positions = x + (index - (len(series) - 1) / 2) * 0.36
        bars = axis.bar(positions, values, width=0.36, color=color, label=label)
        for position, value in zip(positions, values):
            if not math.isfinite(value):
                axis.text(position, 0, "N/A", rotation=90, ha="center", va="bottom", fontsize=7)
        axis.bar_label(bars, labels=[f"{v:.1f}" if math.isfinite(v) else "" for v in values], fontsize=6, padding=2)
    axis.set_xticks(x, [label for label, _ in TIMES], fontsize=7)
    axis.legend(fontsize=8)
    axis.set_xlabel("Time category (totals and components; do not sum bars)", fontsize=8)
    axis.margins(y=0.18)
    save(figure, output, "time" + suffix)
    (output / "comparison-summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    """Accept cluster result roots; no training, model loading or device access."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consistency", type=Path)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--expected-steps", type=int, default=50)
    parser.add_argument("--single-output", type=Path, help="Also export four native-only figures here")
    args = parser.parse_args()
    if args.single_output is not None:
        render(None, args.native, args.single_output, args.smooth_window, args.expected_steps)
    result = render(args.consistency, args.native, args.output, args.smooth_window, args.expected_steps)
    print(json.dumps({"output": str(args.output.resolve()), "missing_logp": result["missing_logp"]}, indent=2))


if __name__ == "__main__":
    main()
