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
"""Offline asynchronous acceptance oracle; never imports production loss or target helpers."""
import argparse
import csv
import html
import json
import math
import struct
from pathlib import Path

import torch
import yaml


def token_objective(current: float, old: float, reference: float | None,
                    advantage: float, config: dict) -> tuple[float, float, float]:
    """Evaluate the scalar GRPO objective and its analytic derivative."""
    ratio = math.exp(current - old)
    low, high, cap = 1 - config["clip_ratio_low"], 1 + config["clip_ratio_high"], config["clip_ratio_c"]
    policy = max(-advantage * ratio, -advantage * min(high, max(low, ratio)))
    derivative = -advantage * ratio
    if (advantage > 0 and ratio > high) or (advantage < 0 and ratio < low):
        derivative = 0.
    if advantage < 0:
        policy = min(policy, -advantage * cap)
        if ratio > cap:
            derivative = 0.
        elif ratio == cap:
            derivative *= 0.5
    delta = reference - current if reference is not None else 0.
    kl = math.exp(delta) - delta - 1.
    coefficient = config["kl_coef"]
    if 0. < kl < 10.:
        derivative += coefficient * (1. - math.exp(delta))
    return policy + coefficient * min(10., max(0., kl)), derivative, ratio


def check_close(actual, expected, *, absolute, relative, label):
    """Fail explicitly with the maximum discrepancy and predeclared tolerance."""
    actual, expected = torch.as_tensor(actual).double(), torch.as_tensor(expected).double()
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise ValueError(f"{label}: non-finite evidence")
    difference = (actual - expected).abs()
    if not torch.all(difference <= absolute + relative * expected.abs()):
        raise ValueError(f"{label}: max error={difference.max().item()}, atol={absolute}, rtol={relative}")
    return float(difference.max().item()) if difference.numel() else 0.


def float32(value: float) -> float:
    """Round one scalar using the same IEEE-754 precision as captured KL primitives."""
    return struct.unpack("f", struct.pack("f", value))[0]


def verify_numerics(directory: Path, config: dict) -> list[dict]:
    """Check every captured token and DP normalizer using independent scalar arithmetic."""
    batches = {}
    for path in directory.glob("batch_step_*_rank_*.pt"):
        step, rank = (int(part) for part in (path.stem.split("_")[2], path.stem.split("_")[4]))
        batches[step, rank] = torch.load(path, map_location="cpu", weights_only=True)
    if not batches:
        raise ValueError("No actual Actor targets were captured")
    results = []
    for (step, rank), batch in sorted(batches.items()):
        mask = batch["loss_action_mask"].bool()
        captured_segments = batch.get("behavior_segments")
        if captured_segments is not None:
            if len(captured_segments) != mask.shape[0]:
                raise ValueError("Behavior segment rows differ from captured actions")
            for row, segments in enumerate(captured_segments):
                if segments is None:
                    continue
                position, previous_version = 0, 0
                for segment in segments:
                    if (segment["start"] != position or segment["end"] <= position
                            or segment["policy_version"] < previous_version
                            or segment["policy_version"] != segment["source_update"] + 1
                            or segment["source_update"] > step - 1):
                        raise ValueError("Invalid independently captured behavior segment")
                    position, previous_version = segment["end"], segment["policy_version"]
                if position != int(mask[row].sum()) or previous_version != batch["policy_versions"][row]:
                    raise ValueError("Behavior segment coverage or terminal version differs")
        check_close(batch["old_log_probs"][mask], batch["rollout_log_probs"][mask],
                    absolute=0., relative=0., label="behavior anchor preservation")
        grouped = {}
        for index, group in enumerate(batch["groups"]):
            grouped.setdefault(group, []).append(index)
        row_versions = [({segment["policy_version"] for segment in captured_segments[row]}
                         if captured_segments is not None and captured_segments[row]
                         else {batch["policy_versions"][row]}) for row in range(mask.shape[0])]
        cross_policy_responses = sum(len(versions) > 1 for versions in row_versions)
        mixed_policy_groups = sum(len(set().union(*(row_versions[row] for row in indices))) > 1
                                  for indices in grouped.values())
        expected_advantage = torch.zeros_like(batch["advantages"], dtype=torch.float64)
        for indices in grouped.values():
            rewards = [float(batch["rewards"][index]) for index in indices]
            mean = sum(rewards) / len(rewards)
            std = math.sqrt(sum((value - mean) ** 2 for value in rewards) / (len(rewards) - 1))
            for index, reward in zip(indices, rewards):
                expected_advantage[index] = (reward - mean) / (std + config["advantage_epsilon"])
        expected_advantage *= mask
        advantage_error = check_close(batch["advantages"], expected_advantage,
                                      absolute=2.e-6, relative=2.e-6, label="group advantages")
        paths = sorted(directory.glob(f"loss_step_{step}_rank_{rank}_call_*.pt"),
                       key=lambda path: int(path.stem.split("_")[-1]))
        if len(paths) != len(batch["plan"]):
            raise ValueError("Incomplete micro-batch evidence")
        maximum = {"ratio": 0., "gradient": 0., "gradient_relative_l2": 0.,
                   "loss_sum": 0., "finite_difference": 0.}
        active_tokens = 0
        native_kl_sources = set()
        for path, plan in zip(paths, batch["plan"]):
            call = torch.load(path, map_location="cpu", weights_only=True)
            start, end, mini_start, mini_end = plan
            if (call["start"], call["end"], call["mini_start"], call["mini_end"]) != tuple(plan):
                raise ValueError("Actor traversal differs from the recorded mini/micro plan")
            world = batch["dp_size"]
            global_tokens = sum(int(batches[step, other]["loss_action_mask"][mini_start:mini_end].sum())
                                for other in range(world))
            current, old = call["current_log_probs"], call["old_log_probs"]
            reference, advantages = call["reference_log_probs"], call["advantages"]
            action = call["action_mask"].bool()
            expected_mask = batch["loss_action_mask"][start:end].bool()
            if not torch.equal(action, expected_mask):
                raise ValueError("Actor micro-batch mask differs from its immutable targets")
            for name in ("old_log_probs", "advantages"):
                check_close(call[name][action], batch[name][start:end][action],
                            absolute=0., relative=0., label="micro-batch fixed " + name)
            native = call.get("kl_native")
            if native is None and reference is not None:
                primitive_path = path.with_name(path.name.replace("loss_step_", "kl_step_"))
                native = torch.load(primitive_path, map_location="cpu", weights_only=True)
            if native is not None:
                native_kl_sources.add(native["source"])
            expected_gradient = torch.zeros_like(current, dtype=torch.float64)
            ideal_gradient = torch.zeros_like(current, dtype=torch.float64)
            expected_ratio = torch.ones_like(current, dtype=torch.float64)
            loss = 0.
            for row, column in action.nonzero().tolist():
                values = (float(current[row, column]), float(old[row, column]),
                          None if reference is None else float(reference[row, column]),
                          float(advantages[row, column]))
                objective, derivative, ratio = token_objective(*values, config)
                ideal_gradient[row, column] = derivative * world / global_tokens
                if native is not None:
                    current32 = float32(values[0])
                    delta = float32(float32(values[2]) - current32)
                    exponential = float(native["reference_ratio"][row, column])
                    check_close(exponential, math.exp(delta), absolute=2.e-6, relative=2.e-6,
                                label="native reference exp")
                    raw = float(native["raw_kl"][row, column])
                    rounded = float32(float32(exponential - delta) - 1.)
                    if raw != rounded:
                        raise ValueError("Recorded KL primitive does not match scalar FP32 operation order")
                    ideal_kl = math.exp(delta) - delta - 1.
                    if (raw < 0. or raw > 10.) and 0. < ideal_kl < 10.:
                        derivative -= config["kl_coef"] * (1. - math.exp(delta))
                loss += objective
                expected_gradient[row, column] = derivative * world / global_tokens
                expected_ratio[row, column] = ratio
            maximum["loss_sum"] = max(maximum["loss_sum"], check_close(
                call["total_loss_sum"], loss, absolute=1.e-4, relative=2.e-5, label="loss sum"))
            maximum["ratio"] = max(maximum["ratio"], check_close(
                call["ratio"][action], expected_ratio[action], absolute=2.e-6, relative=2.e-6,
                label="importance ratio"))
            maximum["gradient"] = max(maximum["gradient"], check_close(
                call["log_prob_gradient"], expected_gradient, absolute=2.e-8, relative=1.e-4,
                label="actual backward gradient with native FP32 clamp branch"))
            norm = float(expected_gradient.norm())
            error_norm = float((call["log_prob_gradient"].double() - expected_gradient).norm())
            relative_l2 = error_norm / norm if norm > 0 else error_norm
            if relative_l2 > 1.e-4:
                raise ValueError(f"Backward gradient relative L2 error exceeds 1e-4: {relative_l2}")
            maximum["gradient_relative_l2"] = max(maximum["gradient_relative_l2"], relative_l2)
            # Check selected scalar derivatives by central differences away from clipping kinks.
            positions = action.nonzero().tolist()
            for row, column in positions[::max(1, len(positions) // 32)]:
                value, anchor = float(current[row, column]), float(old[row, column])
                target = None if reference is None else float(reference[row, column])
                advantage = float(advantages[row, column])
                ratio = math.exp(value - anchor)
                distance = min(abs(math.log(ratio / bound))
                               for bound in (1 - config["clip_ratio_low"], 1 + config["clip_ratio_high"],
                                             config["clip_ratio_c"]))
                epsilon = min(1.e-4, distance / 4.)
                if epsilon < 1.e-8:
                    continue
                plus = token_objective(value + epsilon, anchor, target, advantage, config)[0]
                minus = token_objective(value - epsilon, anchor, target, advantage, config)[0]
                numeric = (plus - minus) / (2 * epsilon) * world / global_tokens
                maximum["finite_difference"] = max(maximum["finite_difference"], check_close(
                    numeric, ideal_gradient[row, column], absolute=1.e-9, relative=1.e-5,
                    label="scalar finite difference"))
            active_tokens += len(positions)
        results.append({"update": step, "rank": rank, "micro_batches": len(paths), "action_tokens": active_tokens,
                        "advantage_max_abs_error": advantage_error, "max_abs_errors": maximum,
                        "kl_primitive_sources": sorted(native_kl_sources),
                        "cross_policy_responses": cross_policy_responses,
                        "mixed_policy_groups": mixed_policy_groups})
    return results


def read_events(directory: Path) -> list[dict]:
    """Require a common host boot and monotonic clock before comparing process intervals."""
    events, clocks = [], []
    for path in sorted(directory.glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        times = [row["monotonic_ns"] for row in rows]
        if times != sorted(times):
            raise ValueError(f"Non-monotonic per-process trace: {path}")
        events.extend(rows)
        clocks.extend(row for row in rows if row["event"] == "clock")
    identities = {(row["hostname"], row["boot_id"], row["clock"]) for row in clocks}
    if not clocks or len(identities) != 1 or not clocks[0]["boot_id"]:
        raise ValueError("Cross-process timeline lacks a verified common host clock")
    return sorted(events, key=lambda row: row["monotonic_ns"])


def intervals(events: list[dict], start: str, end: str, identity: str) -> list[dict]:
    """Pair events strictly by process/rank and logical identity."""
    pending, result = {}, []
    for row in events:
        if row["event"] not in (start, end):
            continue
        key = row["role"], row["rank"], row[identity]
        if row["event"] == start:
            if key in pending:
                raise ValueError(f"Duplicate start: {key}")
            pending[key] = row
        else:
            if key not in pending:
                raise ValueError(f"Missing start: {key}")
            first = pending.pop(key)
            result.append({**first, **row, "start_ns": first["monotonic_ns"], "end_ns": row["monotonic_ns"]})
    if pending:
        raise ValueError(f"Unfinished intervals: {list(pending)}")
    return result


def union_length(parts: list[tuple[int, int]]) -> int:
    """Measure union duration, never adding concurrent sampling groups twice."""
    total, right = 0, None
    for left, end in sorted(parts):
        if right is None or left >= right:
            total += end - left
        elif end > right:
            total += end - right
        right = max(right or end, end)
    return total


def verify_timeline(directory: Path, require_progress: bool = False, max_policy_lag: int = 4) -> dict:
    """Verify real update overlap, publication exclusion, and unique admitted groups."""
    events = read_events(directory)
    if not any(row["event"] == "run_end" and row.get("completed") for row in events):
        raise ValueError("Trace lacks successful training completion")
    selected = set()
    for row in events:
        if row["event"] != "batch_selected":
            continue
        for group in row["groups"]:
            ages = [row["update"] - 1 - source for source in
                    group.get("behavior_source_updates", [group["source_update"]])]
            if group["identity"] in selected or any(not 0 <= age <= max_policy_lag for age in ages):
                raise ValueError("Duplicate consumption or invalid behavior-policy age")
            if group["policy_version"] != group["source_update"] + 1:
                raise ValueError("Invalid behavior policy/source update mapping")
            selected.add(group["identity"])
    sampling = intervals(events, "sampling_start", "sampling_end", "identity")
    segments = intervals(events, "segment_start", "segment_end", "segment_id")
    execution = segments if segments else sampling
    updates = intervals(events, "update_start", "update_end", "update")
    publications = intervals(events, "publication_start", "publication_end", "policy_version")
    admitted = [row["identity"] for row in events if row["event"] == "admit"]
    if len(admitted) != len(set(admitted)) or set(admitted) != {row["identity"] for row in sampling}:
        raise ValueError("Admitted and completed sampling identities differ")
    overlap = []
    for update in updates:
        pieces = [(max(update["start_ns"], sample["start_ns"]), min(update["end_ns"], sample["end_ns"]))
                  for sample in execution
                  if max(update["start_ns"], sample["start_ns"]) < min(update["end_ns"], sample["end_ns"])]
        seconds = union_length(pieces) / 1.e9
        if seconds <= 0:
            raise ValueError(f"No actual sampling/update overlap for update={update['update']} rank={update['rank']}")
        probes = [row for row in events if row["event"] == "generation_progress"
                  and row["request_start_ns"] >= update["start_ns"]
                  and row["request_end_ns"] <= update["end_ns"]]
        progress = None
        if len(probes) >= 2:
            progress = (probes[-1]["counters"].get("vllm:generation_tokens_total", 0.)
                        - probes[0]["counters"].get("vllm:generation_tokens_total", 0.))
        if require_progress and (progress is None or progress <= 0):
            raise ValueError("No measured inference token progress entirely inside the optimizer interval")
        overlap.append({"update": update["update"], "rank": update["rank"],
                        "generation_token_progress_inside_update": progress,
                        "update_seconds": (update["end_ns"] - update["start_ns"]) / 1.e9,
                        "sampling_overlap_seconds": seconds})
    for publication in publications:
        if any(max(publication["start_ns"], update["start_ns"])
               < min(publication["end_ns"], update["end_ns"]) for update in updates):
            raise ValueError("Actor update overlapped in-place parameter publication")
        if any(max(publication["start_ns"], sample["start_ns"])
               < min(publication["end_ns"], sample["end_ns"]) for sample in execution):
            raise ValueError("Sampling overlapped in-place parameter publication")
    if not updates or not sampling:
        raise ValueError("Missing sampling or optimizer intervals")
    with (directory / "events.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["monotonic_ns", "wall_time_ns", "role", "rank", "event", "details"])
        for row in events:
            writer.writerow([row["monotonic_ns"], row["wall_time_ns"], row["role"], row["rank"],
                             row["event"], json.dumps(row, sort_keys=True)])
    render_timeline(directory, sampling, updates, publications)
    crossing = sum(any(max(publication["start_ns"], sample["start_ns"])
                       < min(publication["end_ns"], sample["end_ns"]) for publication in publications)
                   for sample in sampling) if segments else 0
    return {"updates": overlap, "sampling_groups": len(sampling), "publication_exclusion": True,
            "unique_admissions": admitted, "responses_crossing_publications": crossing}


def require_partial_coverage(result: dict) -> None:
    """Reject successful short runs that never consumed the intended cross-policy cases."""
    if (not any(row["cross_policy_responses"] for row in result["numerics"])
            or not any(row["mixed_policy_groups"] for row in result["numerics"])
            or not result["timeline"]["responses_crossing_publications"]):
        raise ValueError("Insufficient partial-rollout coverage: require consumed cross-policy responses and groups")


def render_timeline(directory: Path, sampling: list, updates: list, publications: list) -> None:
    """Write a standalone SVG/HTML timeline with hoverable original event records."""
    rows = [("sampling", item) for item in sampling] + [("update", item) for item in updates]
    rows += [("publication", item) for item in publications if item["rank"] == 0]
    rows.sort(key=lambda pair: pair[1]["start_ns"])
    origin = min(item["start_ns"] for _, item in rows)
    duration = (max(item["end_ns"] for _, item in rows) - origin) / 1.e9
    colors = {"sampling": "#2563eb", "update": "#16a34a", "publication": "#d97706"}
    content = []
    for index, (kind, item) in enumerate(rows):
        start = (item["start_ns"] - origin) / 1.e9
        length = (item["end_ns"] - item["start_ns"]) / 1.e9
        if kind == "sampling":
            identity = f"g{item['identity']} v{item['policy_version']}"
        elif kind == "update":
            identity = f"u{item['update']}"
        else:
            identity = f"v{item['policy_version']}"
        label = f"{kind} {identity} / rank {item['rank']}"
        content.append(f'<text x="5" y="{index * 32 + 25}">{html.escape(label)}</text>')
        content.append(f'<rect x="{240 + start / duration * 900:.2f}" y="{index * 32 + 10}" '
                       f'width="{max(1, length / duration * 900):.2f}" height="20" fill="{colors[kind]}">'
                       f'<title>{html.escape(json.dumps(item, indent=2))}</title></rect>')
        content.append(f'<text x="1150" y="{index * 32 + 25}">{length:.3f}s</text>')
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1260 {len(rows) * 32 + 15}">'
           + "".join(content) + "</svg>")
    (directory / "timeline.svg").write_text(svg)
    (directory / "timeline.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Async execution evidence</title>'
        '<style>body{font:14px system-ui;margin:24px}svg{width:100%}text{font:13px monospace}</style>'
        '<h1>Async execution evidence</h1><p>Common-host monotonic clock. Blue: sampling; green: Actor update; '
        'orange: parameter publication. Hover over a bar for original events. '
        'Audit IO is included in these durations.</p>'
        + svg + '<p><a href="events.csv">Raw event CSV</a> &middot; '
        '<a href="validation.json">Numerical and overlap report</a></p>')


def main() -> None:
    """Validate one completed audit directory and write machine-readable evidence."""
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("--require-generation-progress", action="store_true")
    parser.add_argument("--require-partial-rollout", action="store_true")
    arguments = parser.parse_args()
    recipe = yaml.safe_load(arguments.config.read_text())
    config = recipe["algorithm"]
    result = {"tolerances": {"ratio_atol": 2.e-6, "ratio_rtol": 2.e-6, "gradient_atol": 2.e-8,
                             "gradient_rtol": 1.e-4, "gradient_relative_l2": 1.e-4,
                             "loss_sum_atol": 1.e-4, "loss_sum_rtol": 2.e-5},
              "numerics": verify_numerics(arguments.directory, config),
              "timeline": verify_timeline(arguments.directory, arguments.require_generation_progress,
                                          recipe["async_training"]["max_policy_lag_updates"])}
    if arguments.require_partial_rollout:
        require_partial_coverage(result)
    (arguments.directory / "validation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
