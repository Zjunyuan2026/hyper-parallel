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
"""Storage failure evidence without Torch or device allocation."""

import json
from types import SimpleNamespace
import unittest

from rl.weight_sync.storage_diagnostics import describe_weight_failure, storage_metadata


class TestStorageDiagnostics(unittest.TestCase):
    """Distinguish unavailable source storage from stale destination references."""

    @staticmethod
    def tensor(pointer: int = 1) -> SimpleNamespace:
        """Supply metadata only; any attempted value access would fail."""
        return SimpleNamespace(shape=(2, 3), dtype="bfloat16", device="npu:0", is_meta=False,
                               stride=lambda: (3, 1), storage_offset=lambda: 0,
                               untyped_storage=lambda: SimpleNamespace(nbytes=lambda: 12),
                               data_ptr=lambda: pointer)

    def test_missing_pointer_is_distinct_from_positive_shape(self) -> None:
        """A positive logical shape must not imply allocated storage."""
        result = storage_metadata(self.tensor(0))
        self.assertEqual(result["shape"], [2, 3])
        self.assertFalse(result["has_data_pointer"])

    def test_failed_pointer_probe_preserves_other_fields(self) -> None:
        """One bad metadata operation must not hide the remaining evidence."""
        tensor = self.tensor()

        def fail() -> None:
            raise RuntimeError("storage unavailable")

        tensor.data_ptr = fail
        result = storage_metadata(tensor)
        self.assertIn("storage unavailable", result["has_data_pointer"])
        self.assertEqual(result["storage_bytes"], 12)

    def test_stale_target_is_reported_with_stage(self) -> None:
        """Compare object identity rather than device values or tensor equality."""
        result = json.loads(describe_weight_failure(
            "lm_head.weight", "copy_target", self.tensor(), self.tensor(0), self.tensor(), lambda t: t,
        ))
        self.assertFalse(result["cached_target_is_current"])
        self.assertFalse(result["cached_target"]["has_data_pointer"])
        self.assertTrue(result["current_target"]["has_data_pointer"])
        self.assertEqual(result["stage"], "copy_target")

    def test_local_conversion_failure_is_preserved(self) -> None:
        """A broken local wrapper must not discard source evidence."""
        tensor = self.tensor()

        def fail(unused_tensor):
            raise RuntimeError("local unavailable")

        result = json.loads(describe_weight_failure("x", "target_to_local", tensor, tensor, tensor, fail))
        self.assertTrue(result["cached_target_is_current"])
        self.assertIn("local unavailable", result["cached_target"]["unavailable"])
        self.assertTrue(result["source"]["has_data_pointer"])
