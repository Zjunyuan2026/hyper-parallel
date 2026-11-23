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
"""Check dense and MoE Qwen3 Q/K bytes against the HP attention path on Ascend."""
import os

import pytest

from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard", essential_mark="essential")
def test_qk_norm_rope() -> None:
    """Compare the bound rollout Q/K to HP before and after a captured weight update."""
    if not os.environ.get("RL_QK_ST_DEVICE"):
        pytest.skip("Set RL_QK_ST_DEVICE to an allocated Ascend device")

    import torch
    from transformers import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, Qwen3RMSNorm

    from hyper_parallel.components.modules.gqa_attention import GQAAttention
    from rl.unified_execution.runtimes.vllm.attention.gqa import rollout_attention_forward
    from rl.unified_execution.runtimes.vllm.model import rollout_factory

    class InheritedNorm(Qwen3RMSNorm):
        """A new model family may inherit the same normalization forward."""

    def expose_qk(_module, query, key, _value, _mask, **_kwargs):
        return query.transpose(1, 2).contiguous(), key

    torch.npu.set_device(int(os.environ["RL_QK_ST_DEVICE"]))
    torch.npu.set_deterministic_level(3)
    config = Qwen3Config(hidden_size=512, num_attention_heads=4, num_key_value_heads=2, head_dim=128)
    attention = GQAAttention(module=Qwen3Attention(config, 0), module_fqn="block", context={},
                             attention_interface=expose_qk)
    attention.q_norm = InheritedNorm(128, eps=attention.q_norm.variance_epsilon)
    attention.o_proj = torch.nn.Identity()
    attention.to("npu", torch.bfloat16).eval()
    reference = attention.forward
    factory = rollout_factory(lambda *, module, module_fqn, context: module)
    factory(module=torch.nn.ModuleDict({"block": attention}), module_fqn="decoder", context={})
    assert attention.forward.__func__ is rollout_attention_forward

    # Multiple KV groups and a partial token tile exercise the interleaved QKV layout.
    inputs = torch.randn(1, 19, 512, device="npu", dtype=torch.bfloat16)
    phase = torch.randn(1, 19, 64, device="npu").repeat(1, 1, 2)
    positions = (phase.cos().bfloat16(), phase.sin().bfloat16())
    with torch.inference_mode():
        attention(inputs, positions)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            captured = attention(inputs, positions)
        inputs.normal_()
        attention.q_norm.weight.normal_()
        attention.k_norm.weight.normal_()
        graph.replay()
        expected = reference(inputs, positions)
        for result in (attention(inputs, positions), captured):
            for actual, wanted in zip(result, expected, strict=True):
                assert torch.equal(actual.contiguous().view(torch.int16), wanted.contiguous().view(torch.int16))
