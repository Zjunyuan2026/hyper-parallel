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
"""Materialize an explicit colocated topology from portable launcher settings."""

import argparse
import os
from pathlib import Path

import yaml


def integer(name: str, default: int) -> int:
    """Read a strictly positive launcher integer."""
    value = int(os.environ.get(name, default))
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def boolean(name: str, default: bool) -> bool:
    """Reject misspelled switches rather than silently disabling them."""
    value = os.environ.get(name, str(default)).lower()
    if value not in ("true", "false"):
        raise ValueError(f"{name} must be true or false")
    return value == "true"


def configure(config: dict, world: int) -> dict:
    """Set documented overrides; the production loader performs full validation."""
    train = config["train"]
    accelerator = train["accelerator"]
    inference = config["rollout"]["vllm"]
    train_tp, rollout_tp = integer("TRAIN_TP", 4), integer("ROLLOUT_TP", 2)
    train_ep, rollout_ep = integer("TRAIN_EP", 4), integer("ROLLOUT_EP", 4)
    if train_tp not in (1, 2, 4) or rollout_tp not in (1, 2, 4):
        raise ValueError("This recipe supports TP1/TP2/TP4 only")
    if any(world % size for size in (train_tp, rollout_tp, train_ep, rollout_ep)):
        raise ValueError("Each TP/EP size must divide the selected device count")
    if 128 % train_ep or 128 % rollout_ep:
        raise ValueError("EP must divide the model's 128 experts")
    accelerator.update(tp=train_tp, dp_shard=world // train_tp, ep=train_ep,
                       edp_shard=world // train_ep if train_ep > 1 else 1,
                       sequence_parallel=boolean("TRAIN_SP", True), cp=1, pp=1)
    inference.update(tensor_parallel_size=rollout_tp, data_parallel_size=world // rollout_tp,
                     expert_parallel_size=rollout_ep, sequence_parallel=boolean("ROLLOUT_SP", False))
    train["max_steps"] = integer("STEPS", 2)
    train["model_init_cpu_threads"] = integer("MODEL_INIT_THREADS", 8)
    accelerator["activation_checkpoint"] = "full"
    train["prompt_batch_size"] = integer("PROMPTS_PER_RANK", 1)
    train["micro_batch_size"] = 1
    candidates = integer("CANDIDATES", 2)
    config["rollout"]["num_return_sequences"] = candidates
    train["response_mini_batch_size"] = candidates
    config["data"]["max_train_samples"] = max(4, world // train_tp * train["prompt_batch_size"])
    config["logging"]["log_steps"] = 1
    config["logging"]["log_samples"] = 2
    config["model"]["weights_path"] = os.environ["RUN_MODEL_PATH"]
    config["model"]["tokenizer_path"] = os.environ["RUN_MODEL_PATH"]
    if os.environ["RUN_MODEL_PATH"] == "/models/Qwen3-30B-A3B":
        config["model"]["registry_name"] = "qwen3_30b_a3b"
    runner = os.environ.get("RUNNER", "codex")
    if runner not in ("codex", "internal"):
        raise ValueError("RUNNER must be codex or internal")
    if runner == "codex":
        root = Path(__file__).resolve().parents[2]
        source = root / "gsm8k/configs/qwen3_30b_a3b_codex_multi_turn.yaml"
        config["agentic"] = yaml.safe_load(source.read_text())["agentic"]
        math_source = root / "gsm8k/configs/qwen3_4b_gsm8k_codex_50step.yaml"
        math_harness = yaml.safe_load(math_source.read_text())["agentic"]["codex"]
        for key in ("model_instructions", "compact_harness_context", "allowed_tools"):
            config["agentic"]["codex"][key] = math_harness[key]
        config["agentic"]["codex"]["session_root"] = "/results/run/trajectories"
        config["data"].update(prompt_instruction=None, max_prompt_length=integer("MAX_PROMPT", 512))
        context = integer("MAX_MODEL_LEN", 8192)
        config["agentic"]["max_episode_tokens"] = context
        inference.update(max_model_len=context, max_num_batched_tokens=integer("PREFILL_TOKENS", 512),
                         enable_auto_tool_choice=True, tool_call_parser="hermes")
        config["rollout"]["max_new_tokens"] = integer("MAX_NEW_TOKENS", 128)
        capacity = integer("EP_CAPACITY", context)
        for rule in train["plan_overrides"]:
            implementation = rule.get("replace", {}).get("implementation", {})
            if implementation.get("_target_", "").endswith("fixed_capacity_ep_compute_fn"):
                implementation["capacity"] = capacity
        if capacity < context or capacity < inference["max_num_batched_tokens"]:
            raise ValueError("EP_CAPACITY must cover a training sequence and inference prefill rows")
    return config


def main() -> None:
    """Write the effective recipe before any NPU model initialization."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--world", type=int, required=True)
    args = parser.parse_args()
    if args.world < 1:
        raise ValueError("world must be positive")
    config = configure(yaml.safe_load(args.source.read_text()), args.world)
    args.destination.write_text(yaml.safe_dump(config, sort_keys=False))
    print("CONFIG_READY", args.destination, "training=", config["train"]["accelerator"],
          "rollout=", config["rollout"]["vllm"], flush=True)


if __name__ == "__main__":
    main()
