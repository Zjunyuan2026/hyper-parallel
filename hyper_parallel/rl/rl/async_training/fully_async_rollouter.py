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
"""Independent CPU producer; only complete, behavior-versioned GRPO groups cross IPC."""
import concurrent.futures
import functools
import os
import queue
import time
import threading
import traceback
from dataclasses import replace
from typing import Any, Mapping

from transformers import AutoTokenizer

from rl.agentic.envs.environment import load_agentic_module
from rl.utils.async_audit import EventTrace
from rl.async_training.protocol import AsyncTrainingConfig
from rl.async_training.protocol import SampleGroup, SamplingTask
from rl.config import tokenizer_trust_remote_code
from rl.dataset.data_source import (
    PromptDataset, build_prompt_records, build_train_dataloader, collate_prompt_samples,
)
from rl.roles.rollout.client import AsyncGenerationClient, HTTPGenerationEngine
from rl.roles.rollout.worker import RolloutManager
from rl.weight_sync.vllm_client import committed_policy_version


class FullyAsyncRollouter:
    """Own the prompt iterator and independent task sessions in a spawned process."""

    def __init__(self, config: Mapping[str, Any], connection: Any, results: Any, source: dict | None) -> None:
        self.config = config
        self.settings = AsyncTrainingConfig.from_mapping(config["async_training"])
        self.trace = EventTrace(self.settings.audit_dir, "rollouter")
        self.connection, self.results = connection, results
        self.candidates = config["rollout"]["num_return_sequences"]
        model, data = config["model"], config["data"]
        self.tokenizer = self._load_tokenizer()
        self.task_local = threading.local()
        dataset = PromptDataset(
            parquet_path=data["train_path"], tokenizer=self.tokenizer,
            max_prompt_length=data["max_prompt_length"],
            max_samples=data.get("max_train_samples"),
            **{key: data[key] for key in (
                "row_adapter", "metadata_columns", "chat_template_kwargs", "prompt_column",
                "answer_column", "prompt_instruction",
            ) if key in data},
        )
        self.collate = functools.partial(collate_prompt_samples, pad_token_id=self.tokenizer.pad_token_id)
        self.loader, self.sampler = build_train_dataloader(
            dataset, self.collate, batch_size=1, rank=0, world_size=1,
            seed=config["train"]["seed"], shuffle=data.get("shuffle", True),
            num_workers=0, pin_memory=False,
        )
        if len(self.loader) == 0:
            raise ValueError("Async prompt dataset is empty")
        self.epoch = 0 if source is None else source["epoch"]
        self.sampler.set_epoch(self.epoch)
        if source is not None:
            self.loader.load_state_dict(source["loader"])
        self.iterator = iter(self.loader)
        load_agentic_module(config["agentic"]["module_path"])
        vllm = config["rollout"]["vllm"]
        self.client = AsyncGenerationClient(
            f"http://{vllm.get('host', '127.0.0.1')}:{vllm['port']}", model["registry_name"],
            request_timeout=vllm.get("request_timeout", 600), operation_timeout=self.settings.operation_timeout,
        )
        self.pending: dict[concurrent.futures.Future, tuple[int, int]] = {}
        self.groups: dict[int, tuple[SamplingTask, float, list]] = {}
        self.policy_version = committed_policy_version(self.client)
        self.next_probe = 0.0
        self.draining = False
        self.drain_acknowledged = False

    def _load_tokenizer(self) -> Any:
        """Keep source encoding and concurrent task tokenization on independent instances."""
        model = self.config["model"]
        tokenizer = AutoTokenizer.from_pretrained(
            model["tokenizer_path"], local_files_only=True,
            trust_remote_code=tokenizer_trust_remote_code(model),
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        return tokenizer

    def _next_prompt(self, task: SamplingTask) -> Any:
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.epoch += 1
            self.sampler.set_epoch(self.epoch)
            self.iterator = iter(self.loader)
            batch = next(self.iterator)
        record = build_prompt_records(batch, batch["input_ids"], batch["attention_mask"])[0]
        self.trace.emit("source_read", identity=task.identity, source_prompt_id=str(record.prompt_id),
                        epoch=self.epoch)
        return replace(record, prompt_id=str(task.identity),
                       metadata={**record.metadata, "source_prompt_id": record.prompt_id})

    def _generate_candidate(self, task: SamplingTask, prompt: Any, candidate: int) -> Any:
        rollout, agent = self.config["rollout"], self.config["agentic"]
        if not hasattr(self.task_local, "tokenizer"):
            self.task_local.tokenizer = self._load_tokenizer()
        tokenizer = self.task_local.tokenizer
        engine = HTTPGenerationEngine(self.client, task.policy_version)
        manager = RolloutManager(
            engine=engine, tokenizer=tokenizer, environment_name=agent["environment"],
            num_return_sequences=1, max_turns=1,
            max_observation_tokens=agent.get("max_observation_tokens", 0),
            max_new_tokens=rollout["max_new_tokens"], temperature=1.0, top_p=1.0, top_k=0,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            collect_old_log_probs=True,
            seed=(int(rollout.get("seed") or self.config["train"]["seed"])
                  + task.identity * self.candidates + candidate),
            ignore_eos=rollout.get("ignore_eos", False),
            max_episode_tokens=agent.get("max_episode_tokens"),
            environment_settings=dict(agent), interaction_mode="single_turn",
        )
        result = manager.generate([prompt], policy_version=task.policy_version)
        if len(result.trajectories) != 1 or not engine.segments:
            raise ValueError("Async candidate must contain sampled actions")
        row = result.trajectories[0]
        version = engine.segments[-1]["policy_version"]
        return replace(
            row, trajectory_id=f"{task.identity}:{candidate}",
            policy_version=version, worker_policy_version=version,
            metadata={**row.metadata, "source_prompt_id": prompt.metadata["source_prompt_id"],
                      "behavior_segments": tuple(engine.segments), "partial_interruptions": engine.interruptions},
        )

    def _accept_candidate(self, future: concurrent.futures.Future) -> None:
        """Publish one group only after all its candidate futures have succeeded."""
        identity, candidate = self.pending.pop(future)
        task, started, rows = self.groups[identity]
        rows[candidate] = future.result()
        if any(row is None for row in rows):
            return
        finished = time.monotonic()
        self.trace.emit("sampling_end", identity=identity, policy_version=task.policy_version,
                        source_update=task.source_update, candidates=len(rows))
        payload = SampleGroup(task, tuple(rows), finished - started, started, finished).encode(
            self.candidates, self.settings.max_group_bytes,
        )
        deadline = time.monotonic() + self.settings.operation_timeout
        while True:
            try:
                self.results.put(payload, timeout=0.1)
                break
            except queue.Full:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Result transport blocked during group completion")
        del self.groups[identity]

    def run(self) -> None:
        """Run task admission and completion handling, keeping control separate from data."""
        self.connection.send(("ready", self.policy_version))
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.settings.max_inflight_groups * self.candidates,
        )
        try:
            while True:
                if self.connection.poll(0.02):
                    command, value = self.connection.recv()
                    if command == "stop":
                        if self.groups:
                            raise RuntimeError("Cannot stop with in-flight groups")
                        break
                    if command == "task":
                        if (self.draining or len(self.groups) >= self.settings.max_inflight_groups
                                or value.policy_version != self.policy_version):
                            raise RuntimeError("Invalid sampling admission")
                        prompt = self._next_prompt(value)
                        self.groups[value.identity] = (value, time.monotonic(), [None] * self.candidates)
                        self.trace.emit("sampling_start", identity=value.identity, policy_version=value.policy_version,
                                        source_update=value.source_update)
                        for index in range(self.candidates):
                            future = executor.submit(self._generate_candidate, value, prompt, index)
                            self.pending[future] = (value.identity, index)
                    elif command == "drain":
                        self.draining, self.drain_acknowledged = True, False
                    elif command == "pause":
                        self.draining, self.drain_acknowledged = True, True
                        self.client.pause()
                        self.connection.send(("paused", None))
                    elif command == "resume":
                        self.client.resume(value)
                        self.policy_version = value
                        self.draining, self.drain_acknowledged = False, False
                        self.connection.send(("resumed", value))
                    elif command == "snapshot":
                        if not self.draining or self.groups:
                            raise RuntimeError("Source snapshot requires quiescence")
                        self.connection.send(("snapshot", {"epoch": self.epoch, "loader": self.loader.state_dict()}))
                    elif command != "task":
                        raise ValueError(f"Unknown sampler command: {command}")
                for future in list(self.pending):
                    if not future.done():
                        continue
                    self._accept_candidate(future)
                if self.trace.path is not None and time.monotonic() >= self.next_probe:
                    vllm = self.config["rollout"]["vllm"]
                    self.trace.probe_generation(f"http://{vllm.get('host', '127.0.0.1')}:{vllm['port']}")
                    self.next_probe = time.monotonic() + 2.
                if self.draining and not self.groups and not self.drain_acknowledged:
                    self.connection.send(("drained", None))
                    self.drain_acknowledged = True
        finally:
            self.client.close()
            executor.shutdown(wait=False, cancel_futures=True)


def run_rollouter(config: Mapping[str, Any], connection: Any, results: Any, source: dict | None) -> None:
    """Spawn entry: clear inherited launcher identity and report errors over the control pipe."""
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        os.environ.pop(key, None)
    try:
        FullyAsyncRollouter(config, connection, results, source).run()
    except BaseException:
        connection.send(("error", traceback.format_exc()))
        raise
    finally:
        connection.close()
