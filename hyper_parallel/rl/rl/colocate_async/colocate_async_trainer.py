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
"""Colocated GRPO: sample, abort, sleep, update, publish, then resume partial work."""
import logging
import sys
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, fields, replace
from functools import partial
from operator import methodcaller
from typing import Any, Mapping, Optional

import torch
import torch.distributed as dist

from rl.algorithm.rollout_correction import correction_metrics, prepare_rollout_correction
from rl.async_training.protocol import AsyncCheckpoint, decode_group
from rl.colocate_async.colocate_async_rollouter import ColocateAsyncRollouter
from rl.config import validate_colocate_async_config
from rl.dataset.batch_builder import build_experience_batch
from rl.roles.rollout.client import AsyncGenerationClient
from rl.trainer import RLRoleRuntime
from rl.utils.async_audit import EventTrace
from rl.utils.monitoring.colocate_async_metrics import report_colocate_update, summarize_colocate_payloads
from rl.utils.process_cleanup import temporary_environment
from rl.weight_sync.sync import coordinator_call

logger = logging.getLogger(__name__)


class _ColocateRoleRuntime(RLRoleRuntime):
    """Adapt residency and evaluation ownership without changing shared roles."""

    def release_training_residency(self) -> None:
        """Reshard and release training allocations before rollout wakes."""
        self._release_training_state_for_rollout()

    @property
    def evaluation_needs_gateway_handoff(self) -> bool:
        """Report whether evaluation shares the producer's external-agent gateway."""
        return self.evaluator is not None and self.resolved_config.get("agentic", {}).get("runner") == "codex"

    def evaluate_policy(self, version: int) -> None:
        """Evaluate on every rank and release the evaluation gateway before sampling restarts."""
        try:
            metrics, samples = self.evaluator.run(version)
            self.tracker.log(metrics, step=self.state.global_step, sample_tables={"validation/samples": samples})
        finally:
            if self.evaluation_needs_gateway_handoff:
                self.codex_runtime.close()


@dataclass(frozen=True)
class ColocateUpdateResult:
    """Carry one lease's update outcome and diagnostics without changing trainer state implicitly."""
    rollout: Any
    update: Any
    diagnostics: dict[str, float]
    rejected_groups: tuple[str, ...] = ()


class ColocateAsyncTrainer:
    """Own stage ordering; sampler and model runtimes retain their existing contracts."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        """Validate the independent mode and construct its shared model-role runtime."""
        self.settings = validate_colocate_async_config(config)
        self.config = deepcopy(dict(config))
        self.config["colocate_async"] = asdict(self.settings)
        if self.settings.off_policy.mode == "off":
            self.config["colocate_async"].pop("off_policy")
        role_config = {key: value for key, value in self.config.items() if key != "colocate_async"}
        self.roles = _ColocateRoleRuntime(role_config)
        self.rank = dist.get_rank()
        self.sampling = self.settings.sampling_config(self.roles.parallel_dims.dp_size)
        self.batch_size = self.roles.parallel_dims.dp_size * self.settings.groups_per_dp_rank
        self.candidates = self.config["rollout"]["num_return_sequences"]
        self.rollouter: Optional[ColocateAsyncRollouter] = None
        self.trace = EventTrace(self.settings.audit_dir, "colocate_trainer", self.rank)
        self.optimizer_steps = 0
        self._restored: Optional[dict] = None
        self._snapshot: Optional[dict] = None
        self.timings: dict[str, float] = {}

    @property
    def version(self) -> int:
        """Return the policy version published at the current successful update boundary."""
        return self.roles.state.global_step + 1

    def _load_async_state(self, path: Optional[str]) -> tuple[Optional[int], int]:
        """Keep the potentially large restored buffer on its root owner only."""
        self._restored = AsyncCheckpoint.load(path, self.config)
        if self._restored is None:
            return None, 0
        if self._restored.get("execution_mode") != "colocate_async":
            raise ValueError("Checkpoint is not a colocated asynchronous snapshot")
        if self._restored["source_update"] != self._restored["learner_update"]:
            raise ValueError("Colocated checkpoint must contain the latest published Actor")
        return self._restored["learner_update"], self._restored["optimizer_steps"]

    def _pause_initial_service(self) -> None:
        """Close partial admission before any producer exists or any Actor is published."""
        client = AsyncGenerationClient.from_config(self.config, self.settings.operation_timeout)
        try:
            client.pause()
        finally:
            client.close()

    def _start_rollouter(self, restored: Optional[dict]) -> None:
        """Start or rebuild the producer from its root-owned source and ready groups."""
        self.rollouter = ColocateAsyncRollouter(self.config, self.sampling, restored)
        self.rollouter.resume(self.version)

    def _restore_and_start(self) -> None:
        path = coordinator_call("select colocated checkpoint", partial(AsyncCheckpoint.resolve_path, self.config))
        self.config["train"]["checkpoint"]["load_path"] = path
        self.roles.checkpoints.config["load_path"] = path
        restored_step, self.optimizer_steps = coordinator_call(
            "validate colocated snapshot", partial(self._load_async_state, path),
        )
        self.roles.checkpoints.validate_resume()
        self.roles.checkpoints.begin(self.roles.state)
        if restored_step is not None and restored_step != self.roles.state.global_step:
            raise ValueError("Model and colocated snapshot update identities differ")
        self.roles.release_training_residency()
        environment = {"HYPER_RL_ASYNC_TRAINING": "1",
                       "HYPER_RL_ASYNC_OPERATION_TIMEOUT": str(self.settings.operation_timeout),
                       "HYPER_RL_ASYNC_AUDIT_DIR": self.settings.audit_dir}
        with temporary_environment(environment):
            self.roles.rollout_engine.start_service()
        coordinator_call("close initial partial admission", self._pause_initial_service)
        self.roles.rollout_engine.prepare_for_training()
        self._publish_policy()
        coordinator_call("start colocated sampling", lambda: self._start_rollouter(self._restored))
        self._restored = None
        self.trace.emit("restore_complete", update=self.roles.state.global_step, checkpoint=path)

    def _pause_rollout_for_training(self) -> None:
        started = time.monotonic()
        coordinator_call("pause colocated sampling", partial(methodcaller("pause"), self.rollouter))
        self.timings["timing_s/pause"] = time.monotonic() - started
        started = time.monotonic()
        self.roles.rollout_engine.prepare_for_training()
        self.timings["timing_s/sleep"] = time.monotonic() - started
        self.trace.emit("sleep_done", update=self.roles.state.global_step)

    def _build_local_batch(self, payloads: tuple[bytes, ...]) -> Any:
        if len(payloads) != self.batch_size:
            raise ValueError("Colocated payload must contain one complete global batch")
        start = self.roles.parallel_dims.dp_rank * self.settings.groups_per_dp_rank
        chosen = payloads[start:start + self.settings.groups_per_dp_rank]
        groups = [decode_group(data, self.candidates, self.settings.max_group_bytes) for data in chosen]
        rollout = build_experience_batch(
            tuple(row for group in groups for row in group.trajectories),
            sum(group.generation_seconds for group in groups),
            self.roles.rollout_manager.agent_runner.settings,
            {"async_training": True, "execution_mode": "colocate_async"}, mixed_policy_versions=True,
        )
        tensors = {item.name: getattr(rollout, item.name).to(self.roles.device)
                   for item in fields(rollout) if isinstance(getattr(rollout, item.name), torch.Tensor)}
        self.trace.emit("device_batch_built", update=self.roles.state.global_step + 1)
        return replace(rollout, **tensors)

    def _update_actor(self, payloads: tuple[bytes, ...]) -> ColocateUpdateResult:
        started = time.monotonic()
        rollout = self.roles.run_synchronized("colocated batch validation", partial(self._build_local_batch, payloads))
        rollout = self.roles.align_rollout(rollout)
        experience = self.roles.prepare_experience(rollout)
        experience, diagnostics, rejected = self._correct_experience(experience)
        if rejected:
            coordinator_call("reject off-policy groups", partial(methodcaller("reject_batch_groups", rejected),
                                                                 self.rollouter))
            return ColocateUpdateResult(None, None, diagnostics, rejected)
        self.roles.release_training_residency()
        step = self.roles.state.global_step + 1
        self.trace.emit("update_start", update=step)
        update = self.roles.run_synchronized("colocated Actor update", lambda: self.roles.actor.update(experience))
        self.trace.emit("update_end", update=step, optimizer_steps=update.optimizer_steps,
                        gradient_norm=update.gradient_norm)
        self.optimizer_steps += update.optimizer_steps
        self.roles.state.global_step = step
        self.roles.state.consumed_samples += len(payloads) * self.candidates
        self.roles.state.consumed_tokens += update.valid_tokens
        coordinator_call("commit colocated batch", partial(methodcaller("commit_batch"), self.rollouter))
        self.timings["timing_s/update"] = time.monotonic() - started
        return ColocateUpdateResult(rollout, update, diagnostics)

    def _correct_experience(self, experience: Any) -> tuple[Any, dict[str, float], tuple[str, ...]]:
        config = self.settings.off_policy
        if config.mode == "off":
            return experience, {}, ()

        def prepare() -> Any:
            """Finish local validation before any diagnostic collective."""
            anchor = self.roles.actor.compute_log_probs(experience)
            return prepare_rollout_correction(experience, anchor, config, self.version)

        experience, analysis = self.roles.run_synchronized("fixed rollout correction anchor", prepare)
        world = self.roles.parallel_dims.dp_size
        group = self.roles._dp_group_info.group if world > 1 else None
        diagnostics = correction_metrics(analysis, experience.loss_action_mask, config,
                                         group=group, world_size=world)
        rejected = [analysis.rejected_groups]
        if world > 1:
            rejected = [None] * world
            dist.all_gather_object(rejected, analysis.rejected_groups, group=group)
        identities = tuple(sorted({identity for values in rejected for identity in values}))
        diagnostics["off_policy/rejected_groups"] = len(identities)
        self.trace.emit("correction_prepared", update=self.roles.state.global_step,
                        anchor_version=self.version, rejected_groups=len(identities))
        return experience, diagnostics, identities

    def _resume_unchanged_policy(self) -> None:
        self.roles.release_training_residency()
        self.roles.rollout_engine.prepare_for_rollout()
        coordinator_call("resume for off-policy refill", partial(methodcaller("resume", self.version), self.rollouter))

    def _publish_policy(self) -> None:
        started = time.monotonic()
        self.trace.emit("publication_start", update=self.roles.state.global_step, policy_version=self.version)
        self.roles.publish_policy(self.version)
        self.trace.emit("publication_end", update=self.roles.state.global_step, policy_version=self.version)
        self.timings["timing_s/publish"] = time.monotonic() - started

    def _publish_policy_and_resume(self) -> None:
        self._publish_policy()
        started = time.monotonic()
        coordinator_call("resume colocated sampling", partial(methodcaller("resume", self.version), self.rollouter))
        self.timings["timing_s/resume"] = time.monotonic() - started

    def _freeze_snapshot(self) -> None:
        self._snapshot = {
            **self.rollouter.drain_and_snapshot(), "execution_mode": "colocate_async",
            "learner_update": self.roles.state.global_step, "source_update": self.roles.state.global_step,
            "policy_version": self.version, "optimizer_steps": self.optimizer_steps,
        }

    def _evaluate_checkpoint(self) -> bool:
        """Hand the shared Codex gateway from a drained producer to evaluation."""
        if self.roles.evaluator is None:
            return False
        handoff = self.roles.evaluation_needs_gateway_handoff
        if handoff:
            coordinator_call("release sampling gateway", partial(methodcaller("close", graceful=True), self.rollouter))
        self.roles.evaluate_policy(self.version)
        return handoff

    def _save_checkpoint(self, *, final: bool) -> None:
        started = time.monotonic()
        # Partial requests must be able to run while drain waits for their completion.
        coordinator_call("freeze colocated source", self._freeze_snapshot)
        self.timings["timing_s/checkpoint_drain"] = time.monotonic() - started
        restart_producer = self._evaluate_checkpoint()
        self._pause_rollout_for_training()
        directory = self.roles.checkpoints.directory(self.roles.state.global_step)
        coordinator_call("invalidate colocated snapshot", partial(AsyncCheckpoint.invalidate, directory))
        self.roles.checkpoints.save(self.roles.state)
        if final and self.config["train"]["checkpoint"].get("save_final", True):
            self.roles.checkpoints.finalize(self.roles.state)
        coordinator_call("commit colocated snapshot",
                         lambda: AsyncCheckpoint.commit(directory, self._snapshot, self.config))
        self.trace.emit("checkpoint_committed", update=self.roles.state.global_step)
        if not final:
            self.roles.release_training_residency()
            self.roles.rollout_engine.prepare_for_rollout()
            if restart_producer:
                coordinator_call("restart sampling after evaluation", lambda: self._start_rollouter(self._snapshot))
            else:
                coordinator_call("resume after colocated checkpoint",
                                 partial(methodcaller("resume", self.version), self.rollouter))
        self._snapshot = None
        self.roles.tracker.log({"timing_s/checkpoint": time.monotonic() - started,
                                "timing_s/checkpoint_drain": self.timings["timing_s/checkpoint_drain"]},
                               step=self.roles.state.global_step)

    def _report_update(self, result: ColocateUpdateResult, payloads: tuple[bytes, ...], started: float) -> None:
        def diagnostics() -> dict[str, float]:
            """Read root-owned counts only after rollout reporting collectives finish."""
            counts = coordinator_call("colocated sampling metrics", partial(methodcaller("metrics"), self.rollouter))
            counts.update(summarize_colocate_payloads(payloads, self.candidates, self.settings.max_group_bytes,
                                                      self.roles.state.global_step))
            counts.update({"colocate/optimizer_steps": self.optimizer_steps,
                           "timing_s/step": time.monotonic() - started, **self.timings, **result.diagnostics})
            return counts
        report_colocate_update(self.roles, result.rollout, result.update, diagnostics)

    def _finish_sampling(self) -> None:
        if self.roles.rollout_engine.phase == "rollout":
            coordinator_call("drain final colocated sampling", self._freeze_snapshot)
            self._snapshot = None
            self._pause_rollout_for_training()

    def _close(self, completed: bool, failed: bool) -> None:
        errors = []
        if self.rank == 0 and self.rollouter is not None:
            try:
                self.rollouter.close(graceful=completed)
            except Exception as error:
                logger.exception("Colocated sampler cleanup failed")
                errors.append(error)
        try:
            self.roles.close()
        except Exception as error:
            logger.exception("Colocated role cleanup failed")
            errors.append(error)
        if errors and not failed:
            raise errors[0]

    def train(self) -> None:
        """Run the explicit sample/abort/sleep/update/publish/resume sequence."""
        completed = False
        refill_rounds = 0
        try:
            self._restore_and_start()
            while self.roles.state.global_step < self.roles.state.max_steps:
                started = time.monotonic()
                self.timings = {}
                payloads = coordinator_call("select colocated groups", lambda: self.rollouter.sample_batch(
                    self.version, self.roles.state.global_step, self.batch_size,
                ))
                self.timings["timing_s/sample_wait"] = time.monotonic() - started
                self._pause_rollout_for_training()
                result = self._update_actor(payloads)
                if result.rejected_groups:
                    self.roles.tracker.log(result.diagnostics, step=self.roles.state.global_step)
                    if refill_rounds >= self.settings.off_policy.max_refill_rounds:
                        raise RuntimeError("Off-policy group rejection exhausted max_refill_rounds")
                    refill_rounds += 1
                    self._resume_unchanged_policy()
                    continue
                if self.settings.off_policy.mode != "off":
                    result.diagnostics["off_policy/refill_rounds"] = refill_rounds
                refill_rounds = 0
                self._publish_policy_and_resume()
                final = self.roles.state.global_step == self.roles.state.max_steps
                self._report_update(result, payloads, started)
                if self.roles.checkpoints.will_save(self.roles.state.global_step):
                    self._save_checkpoint(final=final)
            self._finish_sampling()
            completed = True
        finally:
            self._close(completed, sys.exc_info()[0] is not None)
