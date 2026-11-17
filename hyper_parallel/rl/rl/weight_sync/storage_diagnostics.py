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
"""Host metadata for failed weight copies; never read or copy tensor values."""

import json
from typing import Any, Callable


def storage_metadata(tensor: Any) -> dict[str, Any]:
    """Inspect each field independently so inaccessible storage remains evidence."""
    if tensor is None:
        return {"present": False}
    readers = {
        "shape": lambda: list(tensor.shape),
        "dtype": lambda: str(tensor.dtype),
        "device": lambda: str(tensor.device),
        "is_meta": lambda: bool(tensor.is_meta),
        "stride": lambda: list(tensor.stride()),
        "offset": lambda: tensor.storage_offset(),
        "storage_bytes": lambda: tensor.untyped_storage().nbytes(),
        "has_data_pointer": lambda: bool(tensor.data_ptr()),
    }
    details = {"present": True}
    for key, reader in readers.items():
        try:
            details[key] = reader()
        except Exception as error:  # pylint: disable=broad-exception-caught
            details[key] = f"unavailable: {type(error).__name__}: {str(error)[:160]}"
    return details


def describe_weight_failure(name: str, stage: str, source: Any, target: Any,
                            current: Any, to_local: Callable) -> str:
    """Describe cached and live destinations without altering the original failure."""
    result = {"name": name, "stage": stage, "source": storage_metadata(source),
              "cached_target_is_current": target is current}
    for key, tensor in (("cached_target", target), ("current_target", current)):
        try:
            result[key] = storage_metadata(None if tensor is None else to_local(tensor))
        except Exception as error:  # pylint: disable=broad-exception-caught
            result[key] = {"unavailable": f"{type(error).__name__}: {str(error)[:160]}"}
    return json.dumps(result, ensure_ascii=True, sort_keys=True)
