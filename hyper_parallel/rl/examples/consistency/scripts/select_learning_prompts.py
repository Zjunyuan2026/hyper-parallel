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
"""Select unchanged GSM8K rows for an explicit real-reward acceptance run."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def select_prompts(source: Path, output: Path, rows: list[int], minimum_rows: int) -> dict:
    """Preserve source questions and labels; repeat selected rows for distributed loading."""
    if not rows or min(rows) < 0 or minimum_rows < 1:
        raise ValueError("Select nonnegative source rows and a positive minimum row count")
    if output.exists():
        raise FileExistsError(output)
    frame = pd.read_parquet(source)
    if max(rows) >= len(frame):
        raise ValueError("A selected row is outside the source GSM8K dataset")
    selected = (rows * ((minimum_rows + len(rows) - 1) // len(rows)))[:max(minimum_rows, len(rows))]
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.iloc[selected].to_parquet(output, index=False)
    evidence = {
        "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "selected_original_rows": selected,
        "selected_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "scope": "Unchanged original questions and labels; repeated rows only for acceptance, not accuracy",
    }
    output.with_suffix(".json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return evidence


def main() -> None:
    """Materialize an explicit host-selected subset without generating or changing labels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", required=True)
    parser.add_argument("--minimum-rows", type=int, required=True)
    args = parser.parse_args()
    rows = [int(value) for value in args.rows.split(",")]
    print(json.dumps(select_prompts(args.source, args.output, rows, args.minimum_rows)))


if __name__ == "__main__":
    main()
