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
"""Configuration and CPU wire contracts for complete, versioned GRPO groups."""
import math
import pickle
from dataclasses import dataclass, fields
from typing import Any, Mapping


@dataclass(frozen=True)
class SamplingTask:
    """One admitted sampling event; version identifies committed inference weights."""
    identity: int
    policy_version: int
    source_update: int


@dataclass(frozen=True)
class SampleGroup:
    """An indivisible set of candidates, with their original token probabilities."""
    task: SamplingTask
    trajectories: tuple[Any, ...]
    generation_seconds: float
    started_at: float = 0.0
    finished_at: float = 0.0

    @property
    def source_updates(self) -> tuple[int, ...]:
        """Return actual behavior ages, retaining legacy single-version payload support."""
        return tuple(segment["source_update"] for row in self.trajectories
                     for segment in behavior_segments(row, self.task))

    def validate(self, candidates: int) -> None:
        """Reject partial groups, inconsistent identities and non-CPU payloads."""
        if (not all(math.isfinite(value) for value in (self.generation_seconds, self.started_at, self.finished_at))
                or self.generation_seconds < 0 or self.started_at < 0 or self.finished_at < self.started_at):
            raise ValueError("Invalid generation timing")
        if self.task.identity < 0 or self.task.source_update < 0 or self.task.policy_version < 1:
            raise ValueError("Invalid sample identity or policy version")
        if self.task.policy_version != self.task.source_update + 1:
            raise ValueError("Policy version must identify its source update plus one")
        if len(self.trajectories) != candidates:
            raise ValueError("Incomplete GRPO candidate group")
        identities = {row.trajectory_id for row in self.trajectories}
        groups = {row.group_id for row in self.trajectories}
        if len(identities) != candidates or groups != {str(self.task.identity)}:
            raise ValueError("Duplicate candidates or inconsistent group identity")
        for row in self.trajectories:
            validate_behavior(row, self.task)
            if not math.isfinite(row.reward) or row.metadata.get("reward_status") == "pending":
                raise ValueError("Group rewards must be complete and finite")
            if row.rollout_log_probs is None:
                raise ValueError("Sampling probabilities are required")
            for value in (row.token_ids, row.attention_mask, row.action_mask, row.rollout_log_probs):
                if value.device.type != "cpu" or value.requires_grad:
                    raise ValueError("Wire tensors must be detached CPU tensors")
            scores = row.rollout_log_probs[row.action_mask[1:].bool()]
            if not bool(scores.isfinite().all()) or bool((scores > 1e-5).any()):
                raise ValueError("Invalid sampled action probabilities")

    def encode(self, candidates: int, max_bytes: int) -> bytes:
        """Serialize owned CPU storage, avoiding multiprocessing tensor-handle lifetimes."""
        self.validate(candidates)
        payload = pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)
        if len(payload) > max_bytes:
            raise ValueError("Sample group exceeds max_group_bytes")
        return payload


def behavior_segments(row: Any, task: SamplingTask) -> tuple[dict, ...]:
    """Normalize legacy trajectory identity into a single behavior segment."""
    segments = row.metadata.get("behavior_segments")
    if segments is not None:
        if not isinstance(segments, (list, tuple)):
            raise ValueError("Behavior segments must be a sequence")
        return tuple(segments)
    return ({"start": 0, "end": int(row.action_mask.sum()),
             "policy_version": task.policy_version, "source_update": task.source_update},)


def validate_behavior(row: Any, task: SamplingTask) -> None:
    """Require exact action coverage and monotonic committed behavior identities."""
    segments = behavior_segments(row, task)
    end, previous = 0, task.policy_version
    for segment in segments:
        if not isinstance(segment, Mapping):
            raise ValueError("Behavior segment must be a mapping")
        if any(isinstance(segment.get(key), bool) or not isinstance(segment.get(key), int)
               for key in ("start", "end", "policy_version", "source_update")):
            raise ValueError("Behavior segments must contain integer boundaries and versions")
        if (segment["start"] != end or segment["end"] <= end
                or segment["policy_version"] < previous
                or segment["policy_version"] != segment["source_update"] + 1):
            raise ValueError("Behavior segments have gaps, overlaps or invalid policy identities")
        end, previous = segment["end"], segment["policy_version"]
    if (not segments or end != int(row.action_mask.sum()) or row.policy_version != previous
            or row.worker_policy_version != previous):
        raise ValueError("Trajectory behavior segments do not cover actions or match the worker policy")


def decode_group(payload: bytes, candidates: int, max_bytes: int) -> SampleGroup:
    """Decode trusted local-process data and revalidate it at the receiving boundary."""
    if not isinstance(payload, bytes) or len(payload) > max_bytes:
        raise ValueError("Invalid or oversized group payload")
    group = pickle.loads(payload)
    if not isinstance(group, SampleGroup):
        raise ValueError("Unexpected group payload")
    group.validate(candidates)
    return group


@dataclass(frozen=True)
class AsyncTrainingConfig:
    """Bounded fixed-topology runtime settings; all timeouts are in seconds."""
    groups_per_dp_rank: int = 2
    max_inflight_groups: int = 4
    max_buffer_groups: int = 16
    max_group_bytes: int = 64 * 1024 * 1024
    max_buffer_bytes: int = 1024 * 1024 * 1024
    publish_every_updates: int = 2
    max_policy_lag_updates: int = 4
    operation_timeout: float = 900.0
    audit_dir: str | None = None

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "AsyncTrainingConfig":
        """Parse settings without accepting unknown keys or boolean integers."""
        if not isinstance(config, Mapping):
            raise ValueError("async_training must be a mapping")
        names = {item.name for item in fields(cls)}
        if set(config) - names:
            raise ValueError(f"Unknown async_training fields: {sorted(set(config) - names)}")
        result = cls(**config)
        for item in fields(cls):
            value = getattr(result, item.name)
            if item.name == "audit_dir":
                if value is not None and (not isinstance(value, str) or not value.strip()):
                    raise ValueError("audit_dir must be a non-empty path or null")
                continue
            if item.name == "operation_timeout":
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError("operation_timeout must be finite")
            elif isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{item.name} must be an integer")
            if value <= 0:
                raise ValueError(f"{item.name} must be positive")
        if result.max_inflight_groups > result.max_buffer_groups:
            raise ValueError("In-flight capacity exceeds total group capacity")
        if result.max_buffer_bytes < result.max_buffer_groups * result.max_group_bytes:
            raise ValueError("Reserve max_group_bytes for every admitted group")
        return result


def validate_async_config(config: Mapping[str, Any]) -> AsyncTrainingConfig:
    """Reject unvalidated combinations before creating models or processes."""
    settings = AsyncTrainingConfig.from_mapping(config.get("async_training", {}))
    data = config.get("data", {})
    if data.get("num_workers", 0) != 0 or data.get("pin_memory", False):
        raise ValueError("Initial async CPU source requires num_workers=0 and pin_memory=false")
    if config.get("algorithm", {}).get("name") != "grpo":
        raise ValueError("Asynchronous training currently supports GRPO only")
    if config.get("rollout", {}).get("engine") != "vllm":
        raise ValueError("Asynchronous training requires vLLM")
    rollout = config["rollout"]
    vllm = rollout.get("vllm", {})
    if vllm.get("deployment") != "disjoint":
        raise ValueError("Asynchronous training requires disjoint inference devices")
    if vllm.get("logprobs_mode", "raw_logprobs") != "raw_logprobs":
        raise ValueError("Asynchronous GRPO requires raw_logprobs")
    if (rollout.get("temperature", 1.0), rollout.get("top_p", 1.0), rollout.get("top_k", 0)) != (1.0, 1.0, 0):
        raise ValueError("Initial async recipe requires temperature=1, top_p=1, top_k=0")
    candidates = rollout.get("num_return_sequences", 0)
    if isinstance(candidates, bool) or not isinstance(candidates, int) or candidates < 2:
        raise ValueError("GRPO requires at least two candidates")
    agent = config.get("agentic", {})
    if agent.get("runner", "internal") != "internal" or agent.get("max_turns", 1) != 1:
        raise ValueError("Initial async recipe supports internal single-turn tasks only")
    if agent.get("interaction_mode", "single_turn") != "single_turn":
        raise ValueError("Initial async recipe requires single_turn")
    if config.get("consistency", {}).get("enabled", False):
        raise ValueError("Synchronous same-version consistency gates cannot validate stale policies")
    if "reward_model" in config or config.get("distillation", {}).get("enabled", False):
        raise ValueError("Model rewards and distillation are not validated for asynchronous training")
    if config.get("model", {}).get("name") != "qwen3":
        raise ValueError("Initial async recipe supports Qwen3 dense only")
    accelerator = config.get("train", {}).get("accelerator", {})
    if any(accelerator.get(name, 1) != 1 for name in ("cp", "pp", "dp_replicate", "ep", "edp_shard")):
        raise ValueError("Unsupported async parallel topology")
    dp_size = accelerator.get("dp_shard", 1)
    if isinstance(dp_size, bool) or not isinstance(dp_size, int) or dp_size <= 0:
        raise ValueError("dp_shard must be an explicit positive size")
    if settings.max_buffer_groups < dp_size * settings.groups_per_dp_rank:
        raise ValueError("Buffer must hold at least one global batch")
    rows = settings.groups_per_dp_rank * rollout["num_return_sequences"]
    train = config["train"]
    mini = train["response_mini_batch_size"]
    micro = train["micro_batch_size"]
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in (mini, micro)):
        raise ValueError("Training mini/micro batch sizes must be positive integers")
    if rows % mini or mini % micro:
        raise ValueError("Local candidate rows must divide into equal mini/micro batches")
    return settings
