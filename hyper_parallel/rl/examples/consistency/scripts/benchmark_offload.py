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
"""Compare full-model A3 CPU offload using the existing GSM8K training preset.

Run on the A3 host with Python 3; only the standard library is required.
All ordinary image, device and consistency guards remain active.
"""

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import tempfile


def summarize(directory: Path, exit_code: int, steps: int) -> dict:
    """Read completed-run evidence; failed runs never produce speedup evidence."""
    reports = list(directory.glob("*/run/consistency.json"))
    result = {"exit_code": exit_code, "passed": False, "directory": str(directory)}
    if len(reports) != 1:
        result["reason"] = "Missing or ambiguous consistency report; inspect this run locally"
        failures = []
        for path in sorted(directory.glob("*/run/failure-rank*.json")):
            failure = json.loads(path.read_text(encoding="utf-8"))
            failures.append({key: failure.get(key) for key in
                             ("rank", "error_type", "error", "last_phase")})
        if failures:
            result["failures"] = failures
            result["reason"] = "Training failed; see the original rank errors below"
        return result
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    checks = report.get("checks", [])
    if (exit_code or report.get("passed") is not True or report.get("completed_steps") != steps
            or [check.get("step") for check in checks] != list(range(1, steps + 1))
            or any(check.get("log_probs_valid_tokens", 0) <= 0
                   or check.get("rollout_log_probs_diff_max") != 0
                   or check.get("rollout_log_probs_diff_mean") != 0 for check in checks)):
        result["reason"] = "Training or strict consistency did not pass"
        return result
    rows = [json.loads(line) for line in reports[0].with_name("steps.jsonl").read_text().splitlines()]
    if [row.get("step") for row in rows] != list(range(1, steps + 1)):
        result["reason"] = "Incomplete step metrics"
        return result
    warm = rows[1:]
    means = {}
    keys = sorted({key for row in warm for key in row
                   if key.startswith("timing_s/") or key == "perf/time_per_step"})
    for key in keys:
        values = [row.get(key) for row in warm]
        if all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
            means[key] = statistics.mean(values)
    if means.get("perf/time_per_step", 0) <= 0:
        result["reason"] = "Missing finite positive steady-step timing"
        return result
    peaks = [row["system/max_memory_allocated_gb"] for row in rows
             if "system/max_memory_allocated_gb" in row]
    result.update(passed=True, warm_step_samples=len(warm), mean_seconds=means,
                  reported_peak_allocated_gb=max(peaks) if peaks else None,
                  memory_scope="existing Trainer metric; not whole-device HBM or all-rank peak",
                  generated_tokens=sum(row.get("rollout/generated_tokens", 0) for row in warm))
    return result


def matching_workloads(results: dict) -> bool:
    """Compare effective configs, allowing only the two residency switches to differ."""
    configs = []
    for mode in ("cpu_offload", "device_resident"):
        paths = list(Path(results[mode]["directory"]).glob("*/run/config.yaml"))
        if len(paths) != 1:
            return False
        configs.append([line for line in paths[0].read_text(encoding="utf-8").splitlines()
                        if not line.strip().startswith(("cpu_offload:", "experimental_moe_device_residency:"))])
    return configs[0] == configs[1]


def main() -> None:
    """Run each mode sequentially and print a self-contained local comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--result-base", type=Path, default=Path.cwd() / "offload-results")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize", type=Path, help="Inspect an existing mode directory without training")
    parser.add_argument("--mode", choices=("both", "cpu_offload", "device_resident"), default="both")
    parser.add_argument("--baseline", type=Path, help="Reuse a successful CPU-offload mode directory")
    args = parser.parse_args()
    if args.summarize is not None:
        result = summarize(args.summarize.resolve(), 0, args.steps)
        print(json.dumps(result, indent=2), flush=True)
        raise SystemExit(0 if result["passed"] else 1)
    if args.steps < 3:
        parser.error("Use at least three steps to separate initialization from steady steps")
    root = Path(__file__).resolve().parents[5]
    if args.baseline and args.mode != "device_resident":
        parser.error("--baseline requires --mode device_resident")
    results = {}
    if args.baseline:
        results["cpu_offload"] = summarize(args.baseline.resolve(), 0, args.steps)
        if not results["cpu_offload"]["passed"]:
            parser.error("The supplied baseline has no complete successful consistency evidence")
    base = args.result_base.resolve()
    if not args.dry_run:
        base.mkdir(parents=True, exist_ok=True)
        base = Path(tempfile.mkdtemp(prefix="comparison-", dir=base))
    for mode, enabled in (("cpu_offload", "true"), ("device_resident", "false")):
        if args.mode not in ("both", mode):
            continue
        directory = base / mode
        env = {**os.environ, "CPU_OFFLOAD": enabled, "RESULT_BASE": str(directory)}
        command = ["bash", str(root / "run_a3_gsm8k50.sh"), "--steps", str(args.steps)]
        if args.dry_run:
            command.append("--dry-run")
        print(f"\nOFFLOAD_CASE {mode}: full MoE, {args.steps} steps", flush=True)
        completed = subprocess.run(command, cwd=root, env=env, check=False)
        if args.dry_run:
            if completed.returncode:
                raise SystemExit(completed.returncode)
            continue
        results[mode] = summarize(directory, completed.returncode, args.steps)
        print(json.dumps(results[mode], indent=2), flush=True)
    if args.dry_run:
        return
    if (len(results) == 2 and all(result["passed"] for result in results.values())
            and matching_workloads(results)):
        baseline = results["cpu_offload"]["mean_seconds"]["perf/time_per_step"]
        resident = results["device_resident"]["mean_seconds"]["perf/time_per_step"]
        results["resident_speedup"] = baseline / resident
        results["note"] = "Compare generated token counts too; repeat in reverse order for cache effects"
    elif len(results) == 2 and all(result["passed"] for result in results.values()):
        results["note"] = "Effective workloads differ or config evidence is missing; no speedup calculated"
    destination = base / "comparison.json"
    destination.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print("\nOFFLOAD_COMPARISON", json.dumps(results, indent=2), "\nSaved:", destination, flush=True)
    if not all(result["passed"] for result in results.values() if isinstance(result, dict)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
