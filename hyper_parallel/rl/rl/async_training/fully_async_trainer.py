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
"""Complete-group asynchronous GRPO over the shared role and publication runtime."""
import os
import time
import traceback
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as dist

from rl.utils.async_audit import ActorLossAudit, EventTrace, capture_runtime_state, write_rollout_diagnostics
from rl.async_training.checkpoint import AsyncCheckpoint
from rl.async_training.protocol import behavior_segments, decode_group, validate_async_config
from rl.async_training.runtime import AsyncRuntime, ExperienceBuffer
from rl.dataset.batch_builder import build_experience_batch
from rl.roles.rollout.base import GenerationSettings
from rl.trainer import RLRoleRuntime
from rl.utils.monitoring.metrics import build_training_metrics, enforce_learning_gate, summarize_rollout
from rl.weight_sync.sync import coordinator_call


class FullyAsyncTrainer:
    """Keep optimizer updates and publication collectives on one training thread."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.settings = validate_async_config(config)
        self.config = dict(config)
        self.config["async_training"] = asdict(self.settings)
        role_config = {key: value for key, value in self.config.items() if key != "async_training"}
        self.roles = RLRoleRuntime(role_config)
        self.rank = dist.get_rank()
        self.trace = EventTrace(self.settings.audit_dir, "trainer", self.rank)
        self.loss_audit = None
        if self.settings.audit_dir is not None:
            self.loss_audit = ActorLossAudit(self.roles.actor.algorithm, self.trace)
            self.roles.actor.algorithm = self.loss_audit
        self.buffer = ExperienceBuffer(self.settings, config["rollout"]["num_return_sequences"])
        self.sampler: AsyncRuntime | None = None
        self.source_update = 0
        self.version = 1
        self.last_publication = 0
        self.optimizer_steps = 0

    def _root_command(self) -> tuple[str, Any]:
        self.sampler.pump()
        self.buffer.evict_stale(self.roles.state.global_step)
        required = self.roles.parallel_dims.dp_size * self.settings.groups_per_dp_rank
        if self.roles.state.global_step - self.source_update > self.settings.max_policy_lag_updates:
            return "publish", None
        deadline = time.monotonic() + self.settings.operation_timeout
        while len(self.buffer.ready) < required:
            self.sampler.fill(self.version, self.source_update)
            self.sampler.pump()
            self.buffer.evict_stale(self.roles.state.global_step)
            if self.roles.state.global_step > self.source_update and not self.buffer.reserved:
                self.sampler.drain()
                return "publish", None
            if time.monotonic() >= deadline:
                raise TimeoutError("No complete global batch available")
            time.sleep(0.01)
        self.sampler.fill(self.version, self.source_update)
        groups = self.buffer.select(required)
        self.trace.emit("batch_selected", update=self.roles.state.global_step + 1,
                        groups=[{"identity": group.task.identity, "policy_version": group.task.policy_version,
                                 "source_update": group.task.source_update,
                                 "behavior_source_updates": list(group.source_updates)} for group in groups])
        return "train", tuple(self.buffer.ready[identity] for identity in self.buffer.selected)

    def _command(self) -> tuple[str, Any]:
        command = [None]
        if self.rank == 0:
            try:
                command[0] = self._root_command()
            except Exception:
                command[0] = ("error", traceback.format_exc())
        dist.broadcast_object_list(command, src=0)
        kind, payload = command[0]
        if kind == "error":
            raise RuntimeError(payload)
        return kind, payload

    def _build_batch(self, payloads: tuple[bytes, ...]) -> Any:
        start = self.roles.parallel_dims.dp_rank * self.settings.groups_per_dp_rank
        chosen = payloads[start:start + self.settings.groups_per_dp_rank]
        groups = [decode_group(data, self.buffer.candidates, self.settings.max_group_bytes) for data in chosen]
        rollout = self.config["rollout"]
        settings = GenerationSettings(
            max_new_tokens=rollout["max_new_tokens"], temperature=1.0, top_p=1.0, top_k=0,
            do_sample=True, pad_token_id=self.roles.tokenizer.pad_token_id,
            eos_token_id=self.roles.tokenizer.eos_token_id, collect_log_probs=True,
        )
        batch = build_experience_batch(
            tuple(row for group in groups for row in group.trajectories),
            sum(group.generation_seconds for group in groups),
            settings, {"async_training": True}, mixed_policy_versions=True,
        )
        tensors = {item.name: getattr(batch, item.name).to(self.roles.device)
                   for item in fields(batch) if isinstance(getattr(batch, item.name), torch.Tensor)}
        return replace(batch, **tensors)

    def _update(self, payloads: tuple[bytes, ...]) -> None:
        started = time.monotonic()
        step = self.roles.state.global_step + 1
        self.trace.emit("prepare_start", update=step)
        rollout = self.roles.run_synchronized("async batch validation", lambda: self._build_batch(payloads))
        experience = self.roles.prepare_experience(rollout)
        if self.loss_audit is not None:
            self.roles.run_synchronized("async audit targets", lambda: self.loss_audit.begin(
                step, experience, self.config["train"], self.roles.parallel_dims.dp_size,
            ))
        self.trace.emit("update_start", update=step,
                        groups=sorted({row.group_id for row in experience.trajectories}),
                        behavior_versions=sorted({segment["policy_version"]
                                                  for row in experience.trajectories
                                                  for segment in row.metadata.get("behavior_segments", (
                                                      {"policy_version": row.policy_version},
                                                  ))}))
        update = self.roles.actor.update(experience)
        if self.loss_audit is not None:
            self.roles.run_synchronized("async audit completion", self.loss_audit.finish)
        self.trace.emit("update_end", update=step, optimizer_steps=update.optimizer_steps,
                        gradient_norm=update.gradient_norm)

        def write_samples() -> None:
            if self.roles.parallel_dims.tp_rank == 0:
                directory = Path(self.config["train"]["checkpoint"]["output_dir"]).parent / "rollouts"
                write_rollout_diagnostics(experience, str(directory), step, self.roles.parallel_dims.dp_rank,
                                          self.config["rollout"]["max_new_tokens"])

        self.roles.run_synchronized("async rollout diagnostics", write_samples)
        self.optimizer_steps += update.optimizer_steps
        self.roles.state.global_step += 1
        self.roles.state.consumed_samples += len(payloads) * self.buffer.candidates
        self.roles.state.consumed_tokens += update.valid_tokens
        coordinator_call("commit async batch", self.buffer.commit)
        # Keep transport progress and grant new work before the next optimizer update.
        coordinator_call("async transport progress", lambda: self.sampler.pump())
        metrics, samples = summarize_rollout(
            rollout, {"sample_indices": []}, step=self.roles.state.global_step, sample_limit=0,
            is_request_owner=self.roles.parallel_dims.tp_rank == 0,
        )
        metrics = build_training_metrics(
            step=self.roles.state.global_step, actor_update=update, rollout_metrics=metrics,
            policy=self.roles.rollout_engine,
            diagnostic_metrics={"async/update_seconds": time.monotonic() - started,
                                "async/policy_lag_updates": self.roles.state.global_step - self.source_update,
                                "async/optimizer_steps": self.optimizer_steps},
        )
        finished = time.monotonic()
        counts = coordinator_call("async buffer metrics", lambda: {
            "async/rollout_update_overlap_seconds": self.buffer.overlap_seconds(started, finished),
            "async/ready_groups": len(self.buffer.ready),
            "async/inflight_groups": len(self.buffer.reserved),
            "async/dropped_groups": self.buffer.dropped_groups,
        })
        metrics.update(counts)
        if self.rank == 0:
            metrics["reward/zero_std_group_ratio"] = metrics["reward/zero_std_groups"] / len(payloads)
        versions = [segment for group_payload in payloads
                    for group in [decode_group(group_payload, self.buffer.candidates, self.settings.max_group_bytes)]
                    for row in group.trajectories for segment in behavior_segments(row, group.task)]
        token_ages = [(step - 1 - segment["source_update"], segment["end"] - segment["start"])
                      for segment in versions]
        metrics.update({
            "async/sample_age_max": max(age for age, _ in token_ages),
            "async/sample_age_token_mean": (sum(age * count for age, count in token_ages)
                                            / sum(count for _, count in token_ages)),
            "async/behavior_versions": len({segment["policy_version"] for segment in versions}),
        })
        enforce_learning_gate(
            metrics, step=self.roles.state.global_step,
            config=self.config["train"].get("learning_gate", {}),
            run_synchronized=self.roles.run_synchronized,
        )
        self.roles.tracker.log(metrics, step=self.roles.state.global_step, samples=samples)

    def _publish(self, *, resume: bool = True) -> None:
        started = time.monotonic()
        if resume:
            coordinator_call("pause async sampling", lambda: self.sampler.pause())
        else:
            coordinator_call("drain async sampling", lambda: self.sampler.drain())
        self.source_update = self.roles.state.global_step
        self.version = self.source_update + 1
        self.trace.emit("publication_start", update=self.source_update, policy_version=self.version)
        self.roles.publish_policy(self.version)
        self.trace.emit("publication_end", update=self.source_update, policy_version=self.version)
        self.last_publication = self.roles.state.global_step
        self.roles.tracker.log({"async/publication_seconds": time.monotonic() - started},
                               step=self.roles.state.global_step)
        if resume:
            coordinator_call("resume async sampling", lambda: self.sampler.resume(self.version))

    def _save(self, *, final: bool) -> None:
        coordinator_call("drain for async checkpoint", lambda: self.sampler.drain())
        snapshot = coordinator_call("snapshot async source", lambda: {
            "buffer": self.buffer.state_dict(), "source": self.sampler.snapshot(),
            "learner_update": self.roles.state.global_step, "policy_version": self.version,
            "source_update": self.source_update, "optimizer_steps": self.optimizer_steps,
        })
        directory = self.roles.checkpoints.directory(self.roles.state.global_step)
        coordinator_call("invalidate async checkpoint", lambda: AsyncCheckpoint.invalidate(directory))
        if self.settings.audit_dir is not None:
            state = self.roles.run_synchronized("audit checkpoint state", lambda: capture_runtime_state(self.roles))
            self.trace.emit("checkpoint_state", update=self.roles.state.global_step, digests=state)
        if final:
            self.roles.checkpoints.finalize(self.roles.state)
        else:
            self.roles.checkpoints.save(self.roles.state)
        coordinator_call("commit async checkpoint", lambda: AsyncCheckpoint.commit(directory, snapshot, self.config))
        self.trace.emit("checkpoint_complete", update=self.roles.state.global_step,
                        directory=str(directory), async_state={
                            "learner_update": snapshot["learner_update"], "policy_version": snapshot["policy_version"],
                            "source_update": snapshot["source_update"], "optimizer_steps": snapshot["optimizer_steps"],
                            "next_identity": snapshot["buffer"]["next_identity"],
                            "consumed_groups": snapshot["buffer"]["consumed_groups"],
                            "buffered_groups": len(snapshot["buffer"]["ready"]),
                        })
        if not final:
            coordinator_call("resume after checkpoint", lambda: self.sampler.resume(self.version))

    def train(self) -> None:
        """Execute asynchronous collection, updates, publication and recoverable checkpoints."""
        completed = False
        try:
            restored = coordinator_call("validate async restore", lambda: AsyncCheckpoint.load(
                self.config["train"]["checkpoint"].get("load_path"), self.config,
            ))
            self.roles.checkpoints.validate_resume()
            self.roles.checkpoints.begin(self.roles.state)
            if restored is not None:
                if restored["learner_update"] != self.roles.state.global_step:
                    raise ValueError("Model and async snapshot update identities differ")
                self.optimizer_steps = restored["optimizer_steps"]
                if self.rank == 0:
                    self.buffer.load_state_dict(restored["buffer"])
            if self.settings.audit_dir is not None:
                state = self.roles.run_synchronized("audit restored state", lambda: capture_runtime_state(self.roles))
                self.trace.emit("restore_complete", update=self.roles.state.global_step,
                                checkpoint=self.config["train"]["checkpoint"].get("load_path"),
                                optimizer_steps=self.optimizer_steps, digests=state)
            # Fresh vLLM workers start at version zero. Recovered Actor weights must always be republished.
            self.source_update = self.roles.state.global_step
            self.version = self.source_update + 1
            service_env = {"HYPER_RL_ASYNC_TRAINING": "1",
                           "HYPER_RL_ASYNC_AUDIT_DIR": self.settings.audit_dir}
            previous_env = {key: os.environ.get(key) for key in service_env}
            try:
                for key, value in service_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
                self.roles.rollout_engine.start_service()
            finally:
                for key, value in previous_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
            self.trace.emit("publication_start", update=self.source_update, policy_version=self.version)
            self.roles.publish_policy(self.version)
            self.trace.emit("publication_end", update=self.source_update, policy_version=self.version)
            self.last_publication = self.roles.state.global_step

            def start_sampler() -> None:
                self.sampler = AsyncRuntime(
                    self.config, self.buffer, None if restored is None else restored["source"],
                )
                self.sampler.resume(self.version)
            coordinator_call("start async sampler", start_sampler)
            while self.roles.state.global_step < self.roles.state.max_steps:
                kind, payload = self._command()
                if kind == "publish":
                    self._publish()
                    continue
                self._update(payload)
                due = self.roles.checkpoints.will_save(self.roles.state.global_step)
                periodic = self.roles.state.global_step - self.last_publication >= self.settings.publish_every_updates
                if due or periodic or self.roles.state.global_step == self.roles.state.max_steps:
                    self._publish(resume=not due)
                if due:
                    if self.roles.evaluator is not None:
                        metrics, samples = self.roles.evaluator.run(self.version)
                        self.roles.tracker.log(metrics, step=self.roles.state.global_step,
                                               sample_tables={"validation/samples": samples})
                    final = self.roles.state.global_step == self.roles.state.max_steps
                    self._save(final=final)
            coordinator_call("final drain", lambda: self.sampler.drain())
            completed = True
        finally:
            if self.rank == 0 and self.sampler is not None:
                self.sampler.close(graceful=completed)
            self.roles.close()
            self.trace.emit("run_end", completed=completed, update=self.roles.state.global_step)
