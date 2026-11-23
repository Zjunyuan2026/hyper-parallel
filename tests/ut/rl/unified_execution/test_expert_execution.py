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
"""Native expert row placement, dispatch payload and slot-wise combination."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from rl.unified_execution.ops.moe.combine import ordered_combine
from rl.unified_execution.runtimes.vllm.native_moe import NativeQwen3Experts, expert_source_ranks


class TestNativeExpertLayout(unittest.TestCase):
    """Cover sparse shard ownership and the distinction between rows and replicas."""

    def test_row_partitions(self):
        domain = list(range(8))
        cases = [
            (domain, 2, False, 5, [1, 3, 5, 7]),
            (domain, 2, True, 5, domain),
            ([4, 5], 4, False, 5, [5]),
            ([4, 5], 2, True, 5, [4, 5]),
            ([0, 2, 4, 6], 2, False, 4, [0, 2, 4, 6]),
        ]
        for owners, tp, sp, rank, expected in cases:
            with self.subTest(owners=owners, tp=tp, sp=sp, rank=rank):
                self.assertEqual(expert_source_ranks(owners, domain, tp, sp, rank), expected)

    def test_source_exchange_preserves_bits_and_pads_after_sp(self):
        native = object.__new__(NativeQwen3Experts)
        torch.nn.Module.__init__(native)
        native.common_source_rows = False
        native.dp_size, native.sp_size, native.source_count = 2, 2, 2
        native.source_group = object()
        native.config = SimpleNamespace(num_experts=513)
        context = SimpleNamespace(dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=torch.tensor([3, 5])))
        states = torch.tensor([[-0., 1.], [2., -3.]], dtype=torch.bfloat16)
        ids = torch.tensor([[511, 1], [256, 17]], dtype=torch.int32)

        def gather(destination, packet, *, group):
            self.assertIs(group, native.source_group)
            destination.copy_(packet.repeat(2, 1))

        with patch("torch.distributed.all_gather_into_tensor", side_effect=gather):
            for rows in (2, 0):
                values, routes, bucket = native._source_rows(states[:rows], ids[:rows], context)
                self.assertEqual(bucket, 3)
                for offset in (0, bucket):
                    self.assertTrue(torch.equal(values[offset:offset+rows].view(torch.int16),
                                                states[:rows].view(torch.int16)))
                    self.assertTrue(torch.equal(routes[offset:offset+rows], ids[:rows]))
                    self.assertTrue(routes[offset+rows:offset+bucket].eq(513).all())

    def test_single_scheduler_eager_uses_physical_pcp_rows(self):
        native = object.__new__(NativeQwen3Experts)
        torch.nn.Module.__init__(native)
        native.common_source_rows = False
        native.dp_size, native.sp_size, native.source_count = 1, 2, 2
        native.source_group = object()
        native.config = SimpleNamespace(num_experts=128, max_num_tokens=2048)
        states = torch.ones(1, 16, dtype=torch.bfloat16)
        ids = torch.zeros(1, 8, dtype=torch.int32)
        context = SimpleNamespace(dp_metadata=None, batch_descriptor=None)
        with patch("torch.distributed.all_gather_into_tensor"):
            values, routes, bucket = native._source_rows(states, ids, context)
        self.assertEqual((bucket, values.shape, routes.shape), (1, (2, 16), (2, 8)))

    def test_slot_order_and_final_bfloat16_rounding(self):
        # A balanced tree loses the last contribution. Early BF16 accumulation
        # loses the small middle contribution even with the correct slot order.
        routes = torch.tensor([[[1e8], [1.], [-1e8], [3.]]])
        self.assertEqual(ordered_combine(routes, torch.ones(1, 4)).item(), 3.)
        routes = torch.tensor([[[256.], [1.], [-256.]]], dtype=torch.bfloat16)
        actual = ordered_combine(routes, torch.tensor([[1., .001, 1.]]))
        expected = (torch.tensor(256., dtype=torch.float32) + .001 - 256.).bfloat16()
        torch.testing.assert_close(actual.squeeze(), expected, rtol=0, atol=0)

    def test_native_combine_seeds_positive_zero_and_preserves_float_scores(self) -> None:
        """A rank-ordered/cast-early implementation gives different bytes for these slots."""
        native = object.__new__(NativeQwen3Experts)
        torch.nn.Module.__init__(native)
        native.identity = torch.arange(9, dtype=torch.int32)
        slots = torch.tensor([[-0.0] * 8, [2**25, 1, -2**25, 1, 0, 0, 0, 0]],
                             dtype=torch.bfloat16).T.contiguous().unsqueeze(0)
        weights = torch.full((1, 8), 0.125, dtype=torch.bfloat16)

        def unpermute(*, permuted_tokens, sorted_indices, probs):
            self.assertEqual(sorted_indices.tolist(), list(range(9)))
            products = permuted_tokens.float() * probs.flatten().float()[:, None]
            value = products[0]
            for product in products[1:]:
                value = value + product
            return value.to(permuted_tokens.dtype).unsqueeze(0)

        operator = Mock()
        operator.npu_moe_token_unpermute.side_effect = unpermute
        dependency = SimpleNamespace(DeviceOperator=operator)
        with patch.dict("sys.modules", {"vllm_ascend.device.device_op": dependency}):
            for scores in (weights, weights.float() + 0.00017):
                expected = ordered_combine(slots, scores)
                actual = native._combine(slots, scores)
                self.assertTrue(torch.equal(expected.view(torch.uint8), actual.view(torch.uint8)))
        operator.npu_moe_token_unpermute.assert_called_once()
