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
"""Bounded experience ownership and spawned-producer supervision."""
import multiprocessing as mp
import queue
import time
from collections import OrderedDict, deque
from typing import Any, Mapping

from rl.utils.async_audit import EventTrace
from rl.async_training.protocol import AsyncTrainingConfig, SampleGroup, SamplingTask, decode_group
from rl.async_training.fully_async_rollouter import run_rollouter


class ExperienceBuffer:
    """Own all admitted identities until consumption or explicit stale eviction."""

    def __init__(self, settings: AsyncTrainingConfig, candidates: int) -> None:
        self.settings = settings
        self.candidates = candidates
        self.next_identity = 0
        self.reserved: dict[int, SamplingTask] = {}
        self.ready: OrderedDict[int, bytes] = OrderedDict()
        self.selected: tuple[int, ...] = ()
        self.consumed_groups = 0
        self.dropped_groups = 0
        self.completed_intervals: deque = deque(maxlen=1024)

    def reserve(self, version: int, source_update: int) -> SamplingTask | None:
        """Reserve worst-case storage before the producer reads another prompt."""
        occupied = len(self.reserved) + len(self.ready)
        if occupied >= self.settings.max_buffer_groups or len(self.reserved) >= self.settings.max_inflight_groups:
            return None
        task = SamplingTask(self.next_identity, version, source_update)
        self.next_identity += 1
        self.reserved[task.identity] = task
        return task

    def accept(self, payload: bytes) -> None:
        """Accept exactly one result for a live reservation."""
        group = decode_group(payload, self.candidates, self.settings.max_group_bytes)
        if self.reserved.get(group.task.identity) != group.task:
            raise ValueError("Unreserved, duplicated or incorrectly versioned group")
        del self.reserved[group.task.identity]
        self.ready[group.task.identity] = payload
        self.completed_intervals.append((group.started_at, group.finished_at))

    def evict_stale(self, update: int) -> int:
        """Evict whole groups; selected data cannot change during optimization."""
        if self.selected:
            raise RuntimeError("Cannot evict during an uncommitted training update")
        stale = []
        for identity, payload in self.ready.items():
            group = decode_group(payload, self.candidates, self.settings.max_group_bytes)
            ages = [update - source for source in group.source_updates]
            if min(ages) < 0:
                raise ValueError("Experience comes from a future training update")
            if max(ages) > self.settings.max_policy_lag_updates:
                stale.append(identity)
        for identity in stale:
            del self.ready[identity]
        self.dropped_groups += len(stale)
        return len(stale)

    def select(self, count: int) -> tuple[SampleGroup, ...]:
        """Lease a complete global batch without releasing its capacity."""
        if self.selected or len(self.ready) < count:
            raise RuntimeError("No complete batch or an update is already selected")
        self.selected = tuple(list(self.ready)[:count])
        return tuple(decode_group(self.ready[key], self.candidates, self.settings.max_group_bytes)
                     for key in self.selected)

    def commit(self) -> None:
        """Release selected identities only after all training ranks succeed."""
        if not self.selected:
            raise RuntimeError("No selected update to commit")
        for identity in self.selected:
            del self.ready[identity]
        self.consumed_groups += len(self.selected)
        self.selected = ()

    def overlap_seconds(self, started: float, finished: float) -> float:
        """Return a lower bound on task/update overlap from received completion timestamps."""
        overlap = max((max(0.0, min(end, finished) - max(begin, started))
                       for begin, end in self.completed_intervals), default=0.0)
        self.completed_intervals.clear()
        return overlap

    def state_dict(self) -> dict[str, Any]:
        """Export a quiescent buffer; no in-flight or partial update is resumable."""
        if self.reserved or self.selected:
            raise RuntimeError("Checkpoint requires a drained producer and committed update")
        return {"next_identity": self.next_identity, "ready": list(self.ready.values()),
                "consumed_groups": self.consumed_groups, "dropped_groups": self.dropped_groups}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a snapshot and validate capacity, identities and accounting."""
        if self.reserved or self.ready or self.selected:
            raise RuntimeError("Restore requires an empty buffer")
        next_identity = state["next_identity"]
        ready = OrderedDict()
        for payload in state["ready"]:
            group = decode_group(payload, self.candidates, self.settings.max_group_bytes)
            identity = group.task.identity
            if identity in ready or identity >= next_identity:
                raise ValueError("Invalid restored group identity")
            ready[identity] = payload
        consumed, dropped = state["consumed_groups"], state["dropped_groups"]
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
               for value in (next_identity, consumed, dropped)):
            raise ValueError("Invalid restored buffer counters")
        if len(ready) > self.settings.max_buffer_groups or next_identity != consumed + dropped + len(ready):
            raise ValueError("Restored buffer violates capacity or identity conservation")
        self.next_identity, self.ready = next_identity, ready
        self.consumed_groups, self.dropped_groups = consumed, dropped


class AsyncRuntime:
    """Connect a single buffer owner to a CPU-only generation producer."""

    def __init__(
        self, config: Mapping[str, Any], buffer: ExperienceBuffer, source: dict | None = None,
        target: Any = run_rollouter,
    ) -> None:
        self.settings = buffer.settings
        self.buffer = buffer
        self.trace = EventTrace(self.settings.audit_dir, "coordinator")
        context = mp.get_context("spawn")
        self.connection, child = context.Pipe()
        self.results = context.Queue(maxsize=self.settings.max_buffer_groups)
        self.process = context.Process(target=target, args=(config, child, self.results, source),
                                       name="HyperRL-Rollouter")
        self.messages: deque = deque()
        self.closed = False
        self.process.start()
        child.close()
        try:
            self.wait("ready")
        except BaseException:
            self.close()
            raise

    def pump(self) -> None:
        """Receive data and control without queue-size or empty polling assumptions."""
        while True:
            try:
                self.buffer.accept(self.results.get_nowait())
            except queue.Empty:
                break
        while self.connection.poll():
            try:
                message = self.connection.recv()
            except EOFError as error:
                raise RuntimeError("Sampler control channel closed unexpectedly") from error
            if message[0] == "error":
                raise RuntimeError(f"Sampler failed:\n{message[1]}")
            self.messages.append(message)
        if not self.process.is_alive():
            raise RuntimeError(f"Sampler exited unexpectedly: {self.process.exitcode}")

    def wait(self, kind: str) -> Any:
        """Wait for an acknowledgement while continuously draining completed results."""
        deadline = time.monotonic() + self.settings.operation_timeout
        while True:
            self.pump()
            for message in list(self.messages):
                if message[0] == kind:
                    self.messages.remove(message)
                    return message[1]
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Sampler did not acknowledge {kind}")
            time.sleep(0.01)

    def fill(self, version: int, source_update: int) -> None:
        """Grant only pre-reserved tasks; one owner allocates monotonic identities."""
        self.pump()
        while True:
            task = self.buffer.reserve(version, source_update)
            if task is None:
                break
            self.trace.emit("admit", identity=task.identity, policy_version=version, source_update=source_update)
            self.connection.send(("task", task))

    def drain(self) -> None:
        """Stop admission and collect all accepted results before returning."""
        self.connection.send(("drain", None))
        self.wait("drained")
        deadline = time.monotonic() + self.settings.operation_timeout
        while self.buffer.reserved:
            self.pump()
            if time.monotonic() >= deadline:
                raise TimeoutError("Drained producer has outstanding result deliveries")
            time.sleep(0.01)

    def pause(self) -> None:
        """Suspend incomplete candidates without waiting for complete GRPO groups."""
        self.connection.send(("pause", None))
        self.wait("paused")

    def resume(self, version: int) -> None:
        """Verify committed inference identity before allowing further tasks."""
        self.connection.send(("resume", version))
        if self.wait("resumed") != version:
            raise RuntimeError("Sampler acknowledged the wrong policy")

    def snapshot(self) -> dict:
        """Capture the source cursor after drain, including epoch transitions."""
        self.connection.send(("snapshot", None))
        return self.wait("snapshot")

    def close(self, graceful: bool = False) -> None:
        """Stop normally after drain, or terminate a failed job with bounded joins."""
        if self.closed:
            return
        self.closed = True
        if graceful and self.process.is_alive():
            try:
                self.connection.send(("stop", None))
            except (BrokenPipeError, EOFError, OSError):
                # The child may exit after reporting the primary error.
                pass
            self.process.join(timeout=5)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=5)
        self.connection.close()
        self.results.cancel_join_thread()
        self.results.close()
