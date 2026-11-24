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
"""Reproduce the Qwen3-4B baseline's ordered 6400/500 GSM8K samples."""

import argparse
import hashlib
import json
import random
import tempfile
from pathlib import Path

import pandas as pd


def prepare(source: Path, destination: Path) -> dict:
    """Sample official RL-format splits without replacement, restarting seed 42 per split."""
    if source.resolve() == destination.resolve():
        raise ValueError("Source and destination must differ")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Destination must be empty; existing splits are never overwritten")
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": 42, "counts": {}, "source_indices": {}, "source_sha256": {}, "split_sha256": {}}
    with tempfile.TemporaryDirectory(dir=destination) as temporary:
        for split, official, count in (("train", 7473, 6400), ("test", 1319, 500)):
            path = source / f"{split}.parquet"
            frame = pd.read_parquet(path)
            if len(frame) != official or not {"prompt", "extra_info"}.issubset(frame.columns):
                raise ValueError(f"Expected official {split} size {official} and prompt/extra_info columns")
            indices = random.Random(42).sample(range(len(frame)), count)
            sampled = Path(temporary) / path.name
            frame.iloc[indices].to_parquet(sampled, index=False)
            manifest["counts"][split] = count
            manifest["source_indices"][split] = indices
            manifest["source_sha256"][split] = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest["split_sha256"][split] = hashlib.sha256(sampled.read_bytes()).hexdigest()
        for split in ("train", "test"):
            (Path(temporary) / f"{split}.parquet").replace(destination / f"{split}.parquet")
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    """Prepare the fixed samples without initializing any device runtime."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.source, args.destination)
    print("BASELINE_DATA_READY", json.dumps(result["counts"]), flush=True)


if __name__ == "__main__":
    main()
