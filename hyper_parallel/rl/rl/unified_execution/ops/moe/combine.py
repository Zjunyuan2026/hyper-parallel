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
"""Canonical expert-owner selection and FP32 top-k slot reduction."""

import torch


def ordered_combine(routes: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Combine [tokens, slots, hidden] in slot order, with one final cast.

    Both multiplication and accumulation use FP32. This is shared by eager
    training and captured inference and never reduces across the rank axis.

    Args:
        routes: [T, K, H] unweighted expert outputs in original router slot order.
        weights: [T, K] matching probabilities, before any low-precision cast.

    Returns:
        [T, H] output in routes.dtype; only the completed FP32 sum is cast.
    """
    if routes.ndim != 3 or weights.shape != routes.shape[:2]:
        raise ValueError("MoE combine requires [tokens, slots, hidden] and [tokens, slots]")
    result = torch.zeros_like(routes[:, 0], dtype=torch.float32)
    # Slot order is the numerical contract across EP sizes; sum/index_add or
    # an early low-precision cast can change the result despite equal algebra.
    for slot in range(routes.shape[1]):
        product = routes[:, slot].float() * weights[:, slot, None].float()
        result = result + product
    return result.to(routes.dtype)
