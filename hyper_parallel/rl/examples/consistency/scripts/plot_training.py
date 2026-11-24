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
"""Plot recorded rewards, losses and exact sampled-token consistency without inventing missing data."""

import argparse
import json
import math
import textwrap
from pathlib import Path

import yaml
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator


def read_rows(path: Path) -> list[dict]:
    """Require a contiguous recorded prefix, retaining raw values and failing on corrupt input."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or [row.get("step") for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("Metrics must contain a nonempty contiguous step sequence starting at1")
    return rows


def number(row: dict, key: str) -> float:
    """Represent absent or unusable measurements as NaN, never as zero."""
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return math.nan


def run_status(directory: Path, rows: list[dict]) -> dict:
    """Distinguish complete acceptance, partial failure and unverified live metrics."""
    native_report = directory / "native-training.json"
    if native_report.is_file():
        state = json.loads(native_report.read_text(encoding="utf-8"))
        requested = state.get("completed_steps")
        complete = (state.get("passed") is True and requested == len(rows)
                    and state.get("strict_consistency") is False
                    and state.get("model_implementation") == "native")
        return {"recorded_steps": len(rows), "requested_steps": requested, "completed_run": complete,
                "accepted": complete, "strict_consistency_enabled": False,
                "strict_zero_for_all_recorded_steps": None, "invalid_or_nonzero_steps": [],
                "status": "COMPLETE (NATIVE)" if complete else "PARTIAL / COMPLETION UNVERIFIED",
                "scope": "Native training completion; train/rollout consistency was not checked"}
    state = {}
    for name in ("summary.json", "consistency.json"):
        path = directory / name
        if path.exists():
            state = json.loads(path.read_text(encoding="utf-8"))
            break
    requested = state.get("requested_steps", state.get("completed_steps"))
    complete = state.get("passed") is True and requested == len(rows)
    valid = []
    for row in rows:
        maximum = number(row, "training/rollout_log_probs_diff_max")
        mean = number(row, "training/rollout_log_probs_diff_mean")
        token_count = number(row, "training/log_probs_valid_tokens")
        valid.append(number(row, "training/rollout_probs_diff_valid") == 1
                     and math.isfinite(token_count) and token_count > 0
                     and math.isfinite(number(row, "training/actor_log_probs_mean"))
                     and math.isfinite(number(row, "training/rollout_log_probs_mean"))
                     and math.isfinite(maximum) and math.isfinite(mean) and maximum == mean == 0)
    return {
        "recorded_steps": len(rows), "requested_steps": requested, "completed_run": complete,
        "strict_zero_for_all_recorded_steps": all(valid),
        "invalid_or_nonzero_steps": [row["step"] for row, passed in zip(rows, valid) if not passed],
        "accepted": complete and all(valid),
        "status": ("COMPLETE" if all(valid) else "COMPLETE / INVALID CONSISTENCY EVIDENCE")
        if complete else "PARTIAL / COMPLETION UNVERIFIED",
        "scope": "Sampled-token pre-update equality; missing steps/measurements are not zero",
    }


def topology_caption(directory: Path) -> str:
    """Read the actual training and rollout topology from this run's materialized configuration."""
    path = directory / "config.yaml"
    if not path.exists():
        path = directory / "generated-config.yaml"
    if not path.exists():
        return "Topology unavailable"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    train = config["train"]["accelerator"]
    rollout = config["rollout"]["vllm"]
    ep = rollout.get("expert_parallel_size", rollout["data_parallel_size"] * rollout["tensor_parallel_size"]
                     if rollout.get("enable_expert_parallel", False) else 1)
    return (f'Train: TP{train["tp"]} DP{train["dp_shard"]} EP{train["ep"]} '
            f'CP{train.get("cp", 1)} SP={train["sequence_parallel"]}\n'
            f'Rollout: TP{rollout["tensor_parallel_size"]} DP{rollout["data_parallel_size"]} '
            f'EP{ep} PCP{rollout.get("context_parallel_size", 1)} '
            f'SP={rollout.get("sequence_parallel", False)}')


def moving_average(values: list[float], window: int) -> list[float]:
    """Average trailing steps; use available initial steps and preserve invalid-window gaps."""
    if window < 1:
        raise ValueError("Smoothing window must be positive")
    result = []
    for index in range(len(values)):
        samples = values[max(0, index - window + 1):index + 1]
        result.append(sum(samples) / len(samples) if all(math.isfinite(v) for v in samples) else math.nan)
    return result


def draw_series(axis: Axes, rows: list[dict], fields: tuple[tuple[str, str, str], ...],
                smooth_window: int = 1) -> None:
    """Plot only measured finite series and retain NaN gaps."""
    steps = [row["step"] for row in rows]
    for key, label, color in fields:
        values = [number(row, key) for row in rows]
        if any(math.isfinite(value) for value in values):
            smoothing = smooth_window > 1
            axis.plot(steps, values, label=f"{label} (raw)" if smoothing else label, color=color,
                      linewidth=1, marker="o", markersize=3, alpha=0.3 if smoothing else 1)
            if smoothing:
                axis.plot(steps, moving_average(values, smooth_window),
                          label=f"{label} (trailing MA{smooth_window})", color=color, linewidth=2)
    axis.set_xlabel("Training step")
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.grid(alpha=0.22)
    if axis.lines:
        axis.legend(loc="best", fontsize=9)
    else:
        axis.text(0.5, 0.5, "No finite measurements recorded", transform=axis.transAxes, ha="center")


def draw_panel(axis: Axes, kind: str, rows: list[dict], status: dict, smooth_window: int = 5) -> None:
    """Render a reward, loss or strict-consistency panel with honest completion boundaries."""
    if kind == "reward":
        draw_series(axis, rows, (("reward/mean", "Reward mean", "#2563eb"),
                                ("reward/std", "Reward population std", "#64748b")), smooth_window)
        axis.set_ylabel("GSM8K reward")
        axis.set_ylim(-0.05, 1.05)
    elif kind == "loss":
        draw_series(axis, rows, (("train/total_loss", "Total loss", "#ea580c"),
                                ("train/policy_loss", "Policy loss", "#7c3aed"),
                                ("train/kl_loss", "KL loss", "#64748b")), smooth_window)
        axis.set_ylabel("Loss (signed)")
    elif kind == "step-time":
        draw_series(axis, rows, (("perf/time_per_step", "Training step time", "#2563eb"),))
        axis.set_ylabel("Seconds")
    else:
        draw_series(axis, rows, (("training/rollout_log_probs_diff_max", "Max logp difference", "#059669"),
                                ("training/rollout_log_probs_diff_mean", "Mean logp difference", "#2563eb")))
        axis.set_ylabel("Logp difference")
        values = [abs(number(row, key)) for row in rows for key in
                  ("training/rollout_log_probs_diff_max", "training/rollout_log_probs_diff_mean")]
        bound = max([v for v in values if math.isfinite(v)] or [0.])
        if bound == 0:
            axis.set_ylim(-1e-6, 1e-6)
            axis.text(0.02, 0.9, "Measured zeros stay exactly zero", transform=axis.transAxes, fontsize=9)
        if status["invalid_or_nonzero_steps"]:
            axis.scatter(status["invalid_or_nonzero_steps"], [0] * len(status["invalid_or_nonzero_steps"]),
                         marker="x", color="#dc2626", label="Invalid evidence or nonzero difference", zorder=5)
            axis.legend(fontsize=8)
    requested = status["requested_steps"] or len(rows)
    axis.set_xlim(0.5, max(requested, len(rows)) + 0.5)
    axis.set_title(kind.capitalize(), loc="left", fontsize=11)



def _save_figure(output: Path, kind: str, rows: list[dict], status: dict, heading: str,
                 smooth_window: int) -> None:
    """Keep figure lifetime bounded for offline PNG/SVG export."""
    dashboard = kind == "training-curves"
    panel_kind = kind.removesuffix("-raw")
    figure = Figure(figsize=(12, 8) if dashboard else (6, 4), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2 if dashboard else 1, 2 if dashboard else 1, squeeze=False).flatten()
    panels = ("reward", "loss", "step-time" if status.get("strict_consistency_enabled") is False else "consistency")
    for axis, panel in zip(axes, panels if dashboard else (panel_kind,)):
        draw_panel(axis, panel, rows, status, smooth_window)
    if dashboard:
        axes[-1].axis("off")
    caption = "\n".join(textwrap.fill(line, width=100 if dashboard else 65) for line in heading.splitlines())
    figure.suptitle(caption, fontsize=10, color="#166534" if status["accepted"] else "#b45309")
    for extension in ("png", "svg"):
        figure.savefig(output / f"{kind}.{extension}", dpi=150, facecolor="white")
    figure.clear()


def render(directory: Path, output: Path, smooth_window: int = 5) -> dict:
    """Export three standalone figures and a dashboard; retain partial/failing run status."""
    if smooth_window < 1:
        raise ValueError("Smoothing window must be positive")
    rows = read_rows(directory / "steps.jsonl")
    status = run_status(directory, rows)
    output.mkdir(parents=True, exist_ok=True)
    caption = topology_caption(directory)
    heading = (f'{status["status"]}: {len(rows)}/{status["requested_steps"] or "?"} recorded steps\n'
               + caption)
    timing_or_consistency = "step-time" if status.get("strict_consistency_enabled") is False else "consistency"
    figures = ("reward", "loss", timing_or_consistency, "training-curves", "reward-raw", "loss-raw")
    for kind in figures[:4]:
        _save_figure(output, kind, rows, status, heading, smooth_window)
    for kind in ("reward-raw", "loss-raw"):
        _save_figure(output, kind, rows, status, heading, 1)
    status["smoothing"] = {"method": "trailing moving average", "window": smooth_window,
                           "initial_steps": "use available steps", "invalid_values": "retain window gaps",
                           "panels": ["reward", "loss"], "consistency_smoothed": False}
    observations = []
    for path in directory.glob("learning-step*-rank0.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("ranks_with_weight_changes", 0) > 0:
            observations.append({"step": record["step"], "publication_completed": record["publication_completed"],
                                 "next_step_logp_verified": record["next_step_logp_verified"]})
    status["observed_parameter_updates"] = sorted(observations, key=lambda row: row["step"])
    status["updates_verified_by_following_rollout"] = [
        row["step"] for row in observations if row["publication_completed"] and row["next_step_logp_verified"]]
    status["update_consistency_witnessed"] = bool(status["updates_verified_by_following_rollout"])
    for kind in figures:
        for extension in ("png", "svg"):
            path = output / f"{kind}.{extension}"
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"Missing training figure: {path}")
    status["figures"] = list(figures)
    (output / "plot-summary.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    return status


def main() -> None:
    """Plot one run directory using CPU-only Matplotlib."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smooth-window", type=int, default=5,
                        help="Trailing mean window for reward/loss (default: 5; 1: raw only)")
    args = parser.parse_args()
    if args.smooth_window < 1:
        parser.error("--smooth-window must be positive")
    output = args.output or args.run_directory / "plots"
    print(json.dumps(render(args.run_directory, output, args.smooth_window), indent=2))


if __name__ == "__main__":
    main()
