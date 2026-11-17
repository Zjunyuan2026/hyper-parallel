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
"""Same-device ACL IPC copies into receiver-owned packed weight storage."""

import ctypes
import logging
import sys
from typing import Any

import torch


logger = logging.getLogger(__name__)


def malloc_ipc_key(args: list[Any]) -> bytes | None:
    """Decode Torch's versioned malloc handle without exposing its key in logs."""
    handle = args[7]
    if not isinstance(handle, bytes):
        raise ValueError("NPU IPC handle must be bytes")
    if len(handle) == 65:
        return handle
    if len(handle) == 67 and handle[:2] == b"\x01c":
        return handle[2:]
    if len(handle) >= 2 and handle[:2] == b"\x01e":
        return None
    raise ValueError("Unsupported NPU IPC handle format")


class _ACLMemory:
    """Bind the documented CANN runtime ABI only when importing NPU memory."""

    def __init__(self) -> None:
        self.library = ctypes.CDLL("libascendcl.so")
        self.open = self.library.aclrtIpcMemImportByKey
        self.open.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_uint64]
        self.open.restype = ctypes.c_int
        self.copy = self.library.aclrtMemcpy
        self.copy.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        self.copy.restype = ctypes.c_int
        self.close = self.library.aclrtIpcMemClose
        self.close.argtypes = [ctypes.c_char_p]
        self.close.restype = ctypes.c_int


def copy_malloc_ipc(args: list[Any], key: bytes) -> torch.Tensor:
    """Import same-device malloc storage without the peer-access reconstruction path.

    The sender synchronizes the broadcast before exporting and holds the buffer
    until receiver acknowledgments. Copying to owned storage keeps imported ACL
    memory out of model operations and releases the mapping before acknowledgment.
    """
    if args[5] != torch.uint8 or len(args[1]) != 1 or tuple(args[2]) != (1,):
        raise ValueError("ACL IPC receiver requires a contiguous uint8 packed buffer")
    count = int(args[1][0])
    offset = int(args[9]) + int(args[3])
    if count <= 0 or offset < 0 or int(args[3]) < 0 or int(args[3]) + count > int(args[8]):
        raise ValueError("ACL IPC packed buffer exceeds its exported storage")
    runtime = _ACLMemory()
    imported = ctypes.c_void_p()
    opened = False
    try:
        # Both processes use the same physical NPU: no peer-access flag is needed.
        code = runtime.open(ctypes.byref(imported), key, 0)
        opened = bool(imported.value)
        if code != 0 or not opened:
            raise RuntimeError(f"ACL same-device IPC import failed: code={code}, null_pointer={not imported.value}")
        output = torch.empty(count, dtype=torch.uint8, device=torch.device("npu", int(args[6])))
        code = runtime.copy(output.data_ptr(), count, imported.value + offset, count, 3)
        if code != 0:
            raise RuntimeError(f"ACL IPC device-to-device copy failed: code={code}, bytes={count}")
        torch.npu.current_stream().synchronize()
        return output
    finally:
        # Preserve an active copy error while still reporting cleanup failures.
        failed = sys.exc_info()[0] is not None
        cleanup_errors = []
        if opened:
            code = runtime.close(key)
            if code != 0:
                logger.error("ACL IPC mapping close failed: code=%d", code)
                cleanup_errors.append(RuntimeError(f"ACL IPC mapping close failed: code={code}"))
        try:
            args[4]._release_ipc_counter_npu(args[11], args[12], device=args[6])  # pylint: disable=protected-access
        except Exception as error:  # pylint: disable=broad-exception-caught
            logger.error("NPU IPC counter release failed: %r", error)
            cleanup_errors.append(error)
        if cleanup_errors and not failed:
            raise cleanup_errors[0]
