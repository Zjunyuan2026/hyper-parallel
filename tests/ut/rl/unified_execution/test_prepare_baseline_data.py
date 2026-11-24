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
"""Check historical sampling order and safe destination handling using real parquet files."""

import random
import runpy
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
PREPARE = runpy.run_path(str(ROOT / "hyper_parallel/rl/examples/gsm8k/scripts/prepare_baseline_data.py"))["prepare"]


class PrepareBaselineDataTests(unittest.TestCase):
    """Validate split identity and sampling without Torch or device dependencies."""

    def test_each_split_restarts_rng_and_preserves_draw_order(self) -> None:
        """Independent seeds preserve exactly the historical ordered samples."""
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            source.mkdir()
            for split, count in (("train", 7473), ("test", 1319)):
                pd.DataFrame({"prompt": [f"{split}:{index}" for index in range(count)],
                              "extra_info": [{"answer": str(index)} for index in range(count)]}).to_parquet(
                                  source / f"{split}.parquet", index=False)
            destination = Path(temporary) / "sampled"
            manifest = PREPARE(source, destination)
            for split, official, count in (("train", 7473, 6400), ("test", 1319, 500)):
                expected = random.Random(42).sample(range(official), count)
                actual = pd.read_parquet(destination / f"{split}.parquet")
                self.assertEqual(actual["prompt"].tolist(), [f"{split}:{index}" for index in expected])
                self.assertEqual(len(set(manifest["source_indices"][split])), count)
            before = (destination / "train.parquet").read_bytes()
            with self.assertRaises(ValueError):
                PREPARE(source, destination)
            self.assertEqual((destination / "train.parquet").read_bytes(), before)
            with self.assertRaises(ValueError):
                PREPARE(source, source)
