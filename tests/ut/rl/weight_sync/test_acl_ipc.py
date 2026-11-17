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
"""CPU-only regressions for same-device ACL IPC ownership and cleanup."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from rl.weight_sync.acl_ipc import copy_malloc_ipc, malloc_ipc_key


class TestACLIPC(unittest.TestCase):
    """Require non-null mappings, byte-correct offsets, and release before ACK."""

    def setUp(self) -> None:
        self.counter = Mock()
        self.args = [torch.Tensor, (8,), (1,), 2, self.counter, torch.uint8,
                     0, b"\x01c" + b"k" * 65, 10, 16, False, b"ref", 0, b"event", False]
        self.runtime = SimpleNamespace(open=Mock(side_effect=self._open), copy=Mock(return_value=0),
                                       close=Mock(return_value=0))
        self.output = Mock()
        self.output.data_ptr.return_value = 2000
        self.runtime_patch = patch("rl.weight_sync.acl_ipc._ACLMemory", return_value=self.runtime)
        self.runtime_patch.start()
        self.addCleanup(self.runtime_patch.stop)
        self.empty_patch = patch("rl.weight_sync.acl_ipc.torch.empty", return_value=self.output)
        self.empty_patch.start()
        self.addCleanup(self.empty_patch.stop)
        self.stream = Mock()
        self.npu_patch = patch("rl.weight_sync.acl_ipc.torch.npu",
                               SimpleNamespace(current_stream=lambda: self.stream), create=True)
        self.npu_patch.start()
        self.addCleanup(self.npu_patch.stop)
        self.device_patch = patch("rl.weight_sync.acl_ipc.torch.device", return_value=torch.device("cpu"))
        self.device_patch.start()
        self.addCleanup(self.device_patch.stop)

    @staticmethod
    def _open(pointer, unused_key, unused_flags):
        pointer._obj.value = 1000  # pylint: disable=protected-access
        return 0

    def test_malloc_and_expandable_wire_formats(self) -> None:
        self.assertEqual(malloc_ipc_key(self.args), b"k" * 65)
        self.args[7] = b"k" * 65
        self.assertEqual(malloc_ipc_key(self.args), b"k" * 65)
        self.args[7] = b"\x01e" + b"metadata"
        self.assertIsNone(malloc_ipc_key(self.args))
        self.args[7] = b"\x02c" + b"k" * 65
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            malloc_ipc_key(self.args)

    def test_same_device_copy_offsets_and_counter(self) -> None:
        self.assertIs(copy_malloc_ipc(self.args, b"key"), self.output)
        self.assertEqual(self.runtime.open.call_args.args[2], 0)
        self.runtime.copy.assert_called_once_with(2000, 8, 1018, 8, 3)
        self.runtime.close.assert_called_once_with(b"key")
        self.counter._release_ipc_counter_npu.assert_called_once_with(b"ref", 0, device=0)
        self.stream.synchronize.assert_called_once()

    def test_null_import_fails_before_copy(self) -> None:
        self.runtime.open.side_effect = lambda *unused: 0
        with self.assertRaisesRegex(RuntimeError, "null_pointer=True"):
            copy_malloc_ipc(self.args, b"key")
        self.runtime.copy.assert_not_called()
        self.runtime.close.assert_not_called()
        self.counter._release_ipc_counter_npu.assert_called_once()

    def test_copy_error_closes_and_releases(self) -> None:
        self.runtime.copy.return_value = 7
        self.runtime.close.return_value = 9
        with self.assertRaisesRegex(RuntimeError, "copy failed: code=7"):
            copy_malloc_ipc(self.args, b"key")
        self.runtime.close.assert_called_once()
        self.counter._release_ipc_counter_npu.assert_called_once()

    def test_close_error_prevents_acknowledgment(self) -> None:
        self.runtime.close.return_value = 9
        with self.assertRaisesRegex(RuntimeError, "close failed: code=9"):
            copy_malloc_ipc(self.args, b"key")
        self.counter._release_ipc_counter_npu.assert_called_once()

    def test_invalid_bounds_and_layout_fail_before_import(self) -> None:
        self.args[3] = 3
        with self.assertRaisesRegex(ValueError, "exceeds"):
            copy_malloc_ipc(self.args, b"key")
        self.runtime.open.assert_not_called()
        self.args[5] = torch.float32
        with self.assertRaisesRegex(ValueError, "uint8"):
            copy_malloc_ipc(self.args, b"key")
