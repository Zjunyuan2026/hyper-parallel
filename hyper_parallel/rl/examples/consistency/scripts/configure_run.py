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
    local_world = integer("ROLLOUT_LOCAL_WORLD_SIZE", world)
    if local_world > world or world % local_world:
        raise ValueError("ROLLOUT_LOCAL_WORLD_SIZE must divide the global Trainer world")
    if local_world != world:
        inference["node_local_world_size"] = local_world
    train_tp = integer("TRAIN_TP", accelerator["tp"])
    rollout_tp = integer("ROLLOUT_TP", inference["tensor_parallel_size"])
    train_cp = integer("TRAIN_CP", accelerator.get("cp", 1))
    rollout_pcp = integer("ROLLOUT_PCP", inference.get("context_parallel_size", 1))
    train_ep = integer("TRAIN_EP", accelerator["ep"])
    rollout_ep = integer("ROLLOUT_EP", inference["expert_parallel_size"])
    if any(size not in (1, 2, 4) for size in (train_tp, rollout_tp, train_cp, rollout_pcp)):
        raise ValueError("This recipe supports TP/CP/PCP sizes 1, 2 or 4 only")
    if world < 1 or any(world % size for size in (
        train_tp * train_cp, rollout_tp * rollout_pcp, train_ep, rollout_ep,
    )):
        raise ValueError("TP x CP, TP x PCP and each EP size must divide the selected device count")
    if 128 % train_ep or 128 % rollout_ep:
        raise ValueError("EP must divide the model's 128 experts")
    if any(local_world % size for size in (train_tp * train_cp, rollout_tp * rollout_pcp, rollout_ep)):
        raise ValueError("Trainer TP x CP and rollout TP x PCP / EP groups must fit within each node")
    if train_cp > 1 and not any(
        rule.get("when") == "cp" for rule in train["plan_overrides"]
    ):
        raise ValueError("TRAIN_CP > 1 requires the CP/PCP consistency recipe")
    accelerator.update(tp=train_tp, dp_shard=world // (train_tp * train_cp), ep=train_ep,
                       edp_shard=world // train_ep if train_ep > 1 else 1,
                       sequence_parallel=boolean("TRAIN_SP", accelerator["sequence_parallel"]), cp=train_cp, pp=1)
    inference.update(tensor_parallel_size=rollout_tp, data_parallel_size=local_world // (rollout_tp * rollout_pcp),
                     context_parallel_size=rollout_pcp, decode_context_parallel_size=1,
                     expert_parallel_size=rollout_ep,
                     sequence_parallel=boolean("ROLLOUT_SP", inference.get("sequence_parallel", False)))
    train["max_steps"] = integer("STEPS", 2)
    accelerator["cpu_offload"] = boolean("CPU_OFFLOAD", accelerator.get("cpu_offload", True))
    train["model_init_cpu_threads"] = integer("MODEL_INIT_THREADS", 8)
    accelerator["activation_checkpoint"] = "full" if train_cp == 1 else "off"
    train["prompt_batch_size"] = integer("PROMPTS_PER_RANK", 1)
    global_prompts = os.environ.get("GLOBAL_PROMPT_BATCH_SIZE")
    dp_size = world // (train_tp * train_cp)
    if global_prompts is not None:
        global_prompts = integer("GLOBAL_PROMPT_BATCH_SIZE", 1)
        if global_prompts % dp_size:
            raise ValueError("GLOBAL_PROMPT_BATCH_SIZE must be divisible by training DP size")
        train["prompt_batch_size"] = global_prompts // dp_size
    train["micro_batch_size"] = integer("MICRO_BATCH_SIZE", 1)
    candidates = integer("CANDIDATES", 2)
    config["rollout"]["num_return_sequences"] = candidates
    train["response_mini_batch_size"] = integer("RESPONSE_MINI_BATCH_SIZE", candidates)
    responses = train["prompt_batch_size"] * candidates
    mini = train["response_mini_batch_size"]
    if responses % mini or mini % train["micro_batch_size"]:
        raise ValueError("Responses per rank must divide into mini-batches and mini-batches into micro-batches")
    transport = os.environ.get("WEIGHT_SYNC_TRANSPORT", "auto")
    if transport not in ("auto", "ipc"):
        raise ValueError("Colocated weight synchronization requires auto or ipc")
    inference["weight_sync"].pop("transport", None)
    inference["weight_sync"]["bucket_size_mb"] = integer("WEIGHT_BUCKET_MB", 128)
    if "GENERATION_TIMEOUT" in os.environ:
        inference["request_timeout"] = integer("GENERATION_TIMEOUT", 1800)
    inference["max_num_seqs"] = integer("MAX_NUM_SEQS", 2)
    inference["kv_cache_memory_bytes"] = integer("KV_CACHE_MEMORY_BYTES", 536870912)
    full_dataset = boolean("FULL_DATASET", False)
    if full_dataset and os.environ.get("LEARNING_PROMPT_ROWS"):
        raise ValueError("Full dataset mode requires LEARNING_PROMPT_ROWS to be unset")
    config["data"]["max_train_samples"] = None if full_dataset else max(4, dp_size * train["prompt_batch_size"])
    if full_dataset:
        config["data"]["shuffle"] = True
    config["logging"]["log_steps"] = 1
    config["logging"]["log_samples"] = 2
    config["model"]["weights_path"] = os.environ["RUN_MODEL_PATH"]
    config["model"]["tokenizer_path"] = os.environ["RUN_MODEL_PATH"]
    if os.environ["RUN_MODEL_PATH"] == "/models/Qwen3-30B-A3B":
        config["model"]["registry_name"] = "qwen3_30b_a3b"
    runner = os.environ.get("RUNNER", "internal")
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
        config["data"]["prompt_instruction"] = None
        context = integer("MAX_MODEL_LEN", 8192)
        config["agentic"]["max_episode_tokens"] = context
        inference.update(enable_auto_tool_choice=True, tool_call_parser="hermes")
    else:
        config["agentic"] = {
            "runner": "internal", "module_path": "examples.gsm8k.agent", "environment": "gsm8k_tools",
            "interaction_mode": "single_turn", "max_turns": 1, "max_observation_tokens": 0,
            "apply_chat_template": False, "max_episode_tokens": None,
        }
        config["data"]["prompt_instruction"] = 'Let\'s think step by step and output the final answer after "####".'
        # Bound Qwen3 reasoning explicitly for the short consistency benchmark.
        config["data"]["chat_template_kwargs"] = {"enable_thinking": False}
        context = integer("MAX_MODEL_LEN", 256)
        inference.update(enable_auto_tool_choice=False, tool_call_parser=None)
    config["data"]["max_prompt_length"] = integer("MAX_PROMPT", 128 if runner == "internal" else 512)
    config["rollout"]["max_new_tokens"] = integer("MAX_NEW_TOKENS", 32 if runner == "internal" else 128)
    if config["data"]["max_prompt_length"] + config["rollout"]["max_new_tokens"] > context:
        raise ValueError("MAX_PROMPT + MAX_NEW_TOKENS must not exceed MAX_MODEL_LEN")
    inference.update(max_model_len=context,
                     max_num_batched_tokens=integer("PREFILL_TOKENS", 128 if runner == "internal" else 512))
    obsolete = [name for name in ("MC2_CAPACITY", "EP_CAPACITY", "INFERENCE_EP_BACKEND") if name in os.environ]
    if obsolete:
        raise ValueError(f"Unset removed expert transport options {obsolete}; inference now uses native Ascend experts")
    if boolean("REQUIRE_LEARNING", False):
        if runner != "internal" or train["max_steps"] < 2 or candidates < 2:
            raise ValueError("Real learning acceptance requires internal GSM8K, >=2 steps and >=2 candidates")
        train["learning_gate"].update(enabled=True, min_gradient_norm=1e-12,
                                      require_mixed_rewards=True, max_step=1)
        if os.environ.get("LEARNING_PROMPT_ROWS"):
            config["data"]["train_path"] = "/results/run/learning-data/train.parquet"
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
    print("TASK_READY", "runner=", config["agentic"]["runner"],
          "interaction=", config["agentic"]["interaction_mode"], "data=", config["data"]["train_path"],
          "max_prompt=", config["data"]["max_prompt_length"],
          "max_new_tokens=", config["rollout"]["max_new_tokens"],
          "max_model_len=", config["rollout"]["vllm"]["max_model_len"],
          "max_train_samples=", config["data"]["max_train_samples"],
          "shuffle=", config["data"]["shuffle"], flush=True)
    print("CONFIG_READY", args.destination, "training=", config["train"]["accelerator"],
          "rollout=", config["rollout"]["vllm"], flush=True)


if __name__ == "__main__":
    main()
