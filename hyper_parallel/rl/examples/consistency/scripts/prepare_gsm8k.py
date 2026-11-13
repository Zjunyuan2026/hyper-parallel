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
"""Convert original GSM8K JSONL without exposing reference answers in prompts."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def convert(source: Path, destination: Path) -> None:
    """Validate both splits before writing a new prepared directory."""
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Output directory must be empty; existing data is never overwritten")
    frames, manifest = {}, {}
    for split in ("train", "test"):
        path = source / f"{split}.jsonl"
        rows = []
        for number, line in enumerate(path.read_text().splitlines(), 1):
            item = json.loads(line)
            question, answer = item["question"], item["answer"]
            if not isinstance(question, str) or not question.strip() or not isinstance(answer, str):
                raise ValueError(f"Invalid question/answer at {path}:{number}")
            if "####" not in answer or not answer.rsplit("####", 1)[1].strip():
                raise ValueError(f"Missing final numeric answer at {path}:{number}")
            ground_truth = answer.rsplit("####", 1)[1].strip().replace(",", "")
            rows.append({"prompt": question, "extra_info": {"answer": ground_truth}})
        if not rows:
            raise ValueError(f"Empty split: {path}")
        frames[split] = pd.DataFrame(rows)
        manifest[split] = {"source": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                           "rows": len(rows)}
    destination.mkdir(parents=True, exist_ok=True)
    for split, frame in frames.items():
        frame.to_parquet(destination / f"{split}.parquet", index=False)
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("DATA_READY", json.dumps(manifest), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    convert(arguments.source, arguments.output)
