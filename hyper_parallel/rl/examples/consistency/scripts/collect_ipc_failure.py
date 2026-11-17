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
"""Print at most 650 lines of existing run evidence without importing Torch."""

import argparse
from collections import deque
import json
from pathlib import Path
import re


def collect(run: Path) -> list[str]:
    """Prioritize storage diagnostics, then the original exception context."""
    lines = ["IPC_FAILURE_REPORT v1"]
    for name in ("source-commit.txt", "source-status.txt", "environment.txt", "generated-config.yaml"):
        path = run / name
        lines.append(f"=== {name} ===")
        if path.is_file():
            with path.open(errors="replace") as stream:
                for index, line in enumerate(stream):
                    if index >= 140:
                        lines.append("[remaining file omitted]")
                        break
                    lines.append(line.rstrip()[:1800])
    for name in ("progress-rank0.json", "summary.json"):
        path = run / name
        lines.append(f"=== {name} ===")
        if path.is_file():
            try:
                value = json.loads(path.read_text())
                lines.extend(f"{key}={str(value[key])[:900]}" for key in
                             ("stage", "step", "last_phase", "error_type", "error") if key in value)
            except (OSError, ValueError) as error:
                lines.append(str(error))
    log = run / "train.log"
    if log.is_file():
        diagnostics = []
        context = []
        before = deque(maxlen=45)
        remaining = 0
        found = False
        with log.open(errors="replace") as stream:
            for number, line in enumerate(stream, 1):
                line = re.sub(r"\x1b\[[0-9;]*m", "", line.rstrip())
                if "IPC_WEIGHT_DIAG" in line and len(diagnostics) < 16:
                    diagnostics.append(f"L{number}: {line[:6000]}")
                if not found and ("data is not allocated yet" in line or "Logical weight load failed" in line):
                    context.extend(before)
                    remaining = 25
                    found = True
                if remaining:
                    context.append(f"L{number}: {line[:1600]}")
                    remaining -= 1
                before.append(f"L{number}: {line[:1600]}")
        lines.extend(["=== storage diagnostics (first 16) ===", *diagnostics,
                      "=== first failure context ===", *context])
    lines.append("=== events-rank0.jsonl (last 12) ===")
    events = run / "events-rank0.jsonl"
    if events.is_file():
        with events.open(errors="replace") as stream:
            lines.extend(line.rstrip()[:1800] for line in deque(stream, maxlen=12))
    return lines[:649] + ["END IPC_FAILURE_REPORT (at most 650 lines)"]


def main() -> None:
    """Accept either RESULT_ROOT or its run subdirectory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    run = args.directory / "run" if (args.directory / "run").is_dir() else args.directory
    print("\n".join(collect(run)))


if __name__ == "__main__":
    main()
