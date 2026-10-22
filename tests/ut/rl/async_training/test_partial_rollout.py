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
"""Independent continuation, group coverage, stale eviction and publication checks."""
import asyncio
import json
import math
import tempfile
import threading
import unittest
from dataclasses import asdict, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import torch
import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from rl.algorithm import GRPOAlgorithm, GRPOConfig
from rl.async_training.runtime import ExperienceBuffer
from rl.async_training.protocol import AsyncTrainingConfig
from rl.async_training.fully_async_trainer import FullyAsyncTrainer
from rl.roles.rollout.client import AsyncGenerationClient, HTTPGenerationEngine
from rl.async_training.protocol import SamplingTask, decode_group
from rl.async_training.runtime import AsyncRuntime
from rl.async_training.vllm_server import PartialRequestService
from rl.dataset.batch_builder import ExperiencePreparer, build_experience_batch
from rl.roles.rollout.base import GenerationRequest, GenerationSettings
from rl.roles.rollout.vllm import VLLMGenerationEngine
from tests.ut.rl.async_training.support import group


class TestPartialGeneration(unittest.TestCase):
    """Check actual preserved prefixes, raw probabilities and cumulative budgets."""

    def test_repeated_interruptions_keep_original_probabilities(self):
        for prefix_length in (1, 2048):
            with self.subTest(prefix_length=prefix_length):
                prefix = list(range(7, 7 + prefix_length))
                probabilities = [-.1 - index / 10000 for index in range(prefix_length)]
                client = Mock(operation_timeout=30)
                client.generate_segment.side_effect = [
                    {"token_ids": prefix, "log_probs": probabilities,
                     "finish_reason": "abort", "policy_version": 1},
                    {"token_ids": [], "log_probs": [], "finish_reason": "abort", "policy_version": 2},
                    {"token_ids": [8], "log_probs": [-.8], "finish_reason": "abort", "policy_version": 3},
                    {"token_ids": [9], "log_probs": [-.9], "finish_reason": "abort", "policy_version": 4},
                ]
                engine = HTTPGenerationEngine(client, 1)
                settings = GenerationSettings(prefix_length + 2, 1., 1., 0, True, 0, 2, collect_log_probs=True)
                result = engine.generate(GenerationRequest(torch.tensor([[4, 5]]), torch.ones(1, 2), settings))
                self.assertEqual(result.sequences.tolist(), [[4, 5] + prefix + [8, 9]])
                torch.testing.assert_close(result.rollout_log_probs, torch.tensor([probabilities + [-.8, -.9]]),
                                           rtol=0., atol=0.)
                calls = [call.args[0] for call in client.generate_segment.call_args_list]
                self.assertEqual([call["prompt_token_ids"] for call in calls],
                                 [[4, 5], [4, 5] + prefix, [4, 5] + prefix, [4, 5] + prefix + [8]])
                self.assertEqual([call["max_tokens"] for call in calls], [prefix_length + 2, 2, 2, 1])
                self.assertEqual(engine.segments, [
                    {"start": 0, "end": prefix_length, "policy_version": 1, "source_update": 0},
                    {"start": prefix_length, "end": prefix_length + 1, "policy_version": 3, "source_update": 2},
                    {"start": prefix_length + 1, "end": prefix_length + 2, "policy_version": 4, "source_update": 3},
                ])
                self.assertEqual(engine.interruptions, 4)
                self.assertEqual(result.finish_reasons, ("length",))

    def test_late_http_response_keeps_its_server_version_after_resume(self):
        client = AsyncGenerationClient("http://unused", "model", 5, 5)
        entered, deliver = threading.Event(), threading.Event()
        received = []

        def request(method, path, payload):
            if path == "hyper_async/generate":
                entered.set()
                if not deliver.wait(5):
                    raise TimeoutError("Test did not release the delayed response")
                return {"policy_version": 1, "token_ids": [7], "log_probs": [-.7], "finish_reason": "abort"}
            return {"status": "paused" if path.endswith("pause") else "resumed"}

        worker = threading.Thread(target=lambda: received.append(client.generate_segment({"request_id": "old"})))
        try:
            with patch.object(client, "_request", side_effect=request):
                worker.start()
                self.assertTrue(entered.wait(5))
                client.pause()
                self.assertFalse(client.admission.is_set())
                client.resume(3)
                deliver.set()
                worker.join(timeout=5)
                self.assertFalse(worker.is_alive())
                self.assertEqual(received[0]["policy_version"], 1)
        finally:
            deliver.set()
            worker.join(timeout=5)
            client.close()

    def test_misaligned_partial_probabilities_fail(self):
        client = Mock(operation_timeout=30)
        client.generate_segment.return_value = {
            "token_ids": [7], "log_probs": [], "finish_reason": "stop", "policy_version": 1,
        }
        engine = HTTPGenerationEngine(client, 1)
        settings = GenerationSettings(3, 1., 1., 0, True, 0, 2, collect_log_probs=True)
        with self.assertRaisesRegex(ValueError, "payload"):
            engine.generate(GenerationRequest(torch.tensor([[4]]), torch.ones(1, 1), settings))

    def test_stop_at_budget_is_not_reported_as_length_truncation(self):
        for reason, expected in (("stop", "stop"), ("abort", "length"), ("length", "length")):
            with self.subTest(reason=reason):
                client = Mock(operation_timeout=30)
                client.generate_segment.return_value = {
                    "token_ids": [7], "log_probs": [-.7], "finish_reason": reason, "policy_version": 1,
                }
                engine = HTTPGenerationEngine(client, 1)
                settings = GenerationSettings(1, 1., 1., 0, True, 0, 2, collect_log_probs=True)
                result = engine.generate(GenerationRequest(torch.tensor([[4]]), torch.ones(1, 1), settings))
                self.assertEqual(result.finish_reasons, (expected,))

    def test_async_frontend_owner_does_not_change_default_server_command(self):
        engine = VLLMGenerationEngine.__new__(VLLMGenerationEngine)
        engine._deployment, engine._config = "disjoint", {"data_parallel_size": 2}
        engine._model = SimpleNamespace(weights_path="/model", tokenizer_path="/model", name="qwen3")
        engine._rollout_model = SimpleNamespace(is_adapted=False, family="qwen3")
        with patch.dict("os.environ", {}, clear=True):
            original = engine._server_command("127.0.0.1", 8100)
        with patch.dict("os.environ", {"HYPER_RL_ASYNC_TRAINING": "1"}, clear=True):
            partial = engine._server_command("127.0.0.1", 8100)
        self.assertNotIn("--api-server-count", original)
        position = partial.index("--api-server-count")
        self.assertEqual(partial[position + 1], "1")
        self.assertEqual(original[2], "vllm.entrypoints.cli.main")
        self.assertEqual(partial[2], "rl.async_training.vllm_server")
        partial[2] = original[2]
        self.assertEqual(partial[:position] + partial[position + 2:], original)


class TestMixedBehaviorGroup(unittest.TestCase):
    """Keep synchronous identity strict while async groups carry explicit behavior spans."""

    @staticmethod
    def _mixed_group():
        original = group(SamplingTask(0, 1, 0))
        row = original.trajectories[1]
        mixed = replace(row, policy_version=3, worker_policy_version=3, metadata={"behavior_segments": (
            {"start": 0, "end": 1, "policy_version": 1, "source_update": 0},
            {"start": 1, "end": 2, "policy_version": 3, "source_update": 2},
        )})
        return replace(original, trajectories=(original.trajectories[0], mixed))

    def test_mixed_candidates_roundtrip_and_oldest_segment_eviction(self):
        value = self._mixed_group()
        payload = value.encode(2, 65536)
        decoded = decode_group(payload, 2, 65536)
        self.assertEqual(decoded.source_updates, (0, 0, 2))
        settings = AsyncTrainingConfig(max_policy_lag_updates=2)
        buffer = ExperienceBuffer(settings, 2)
        buffer.reserve(1, 0)
        buffer.accept(payload)
        self.assertEqual(buffer.evict_stale(2), 0)
        self.assertEqual(buffer.evict_stale(3), 1)
        self.assertEqual(buffer.next_identity, buffer.dropped_groups)

    def test_segment_gaps_and_wrong_terminal_version_rejected(self):
        value = self._mixed_group()
        row = value.trajectories[1]
        for segments in (
            ({"start": 1, "end": 2, "policy_version": 3, "source_update": 2},),
            ({"start": 0, "end": 2, "policy_version": 2, "source_update": 1},),
        ):
            with self.subTest(segments=segments), self.assertRaises(ValueError):
                replace(value, trajectories=(value.trajectories[0], replace(
                    row, metadata={"behavior_segments": segments},
                ))).validate(2)

    def test_synchronous_batch_still_rejects_mixed_versions(self):
        settings = GenerationSettings(2, 1., 1., 0, True, 0, 2, collect_log_probs=True)
        with self.assertRaisesRegex(ValueError, "consistent worker policy"):
            build_experience_batch(self._mixed_group().trajectories, 0.1, settings, {})

    def test_checkpoint_roundtrip_preserves_behavior_segments(self):
        buffer = ExperienceBuffer(AsyncTrainingConfig(), 2)
        buffer.reserve(1, 0)
        buffer.accept(self._mixed_group().encode(2, buffer.settings.max_group_bytes))
        restored = ExperienceBuffer(buffer.settings, 2)
        restored.load_state_dict(buffer.state_dict())
        self.assertEqual(restored.select(1)[0].source_updates, (0, 0, 2))

    def test_mixed_behavior_gradient_and_update_match_scalar_reference(self):
        settings = GenerationSettings(2, 1., 1., 0, True, 0, 4, collect_log_probs=True)
        value = self._mixed_group()
        value = replace(value, trajectories=(value.trajectories[0], replace(
            value.trajectories[1], rollout_log_probs=torch.tensor([-.6, -1.4]),
        )))
        batch = build_experience_batch(value.trajectories, .1, settings, {}, mixed_policy_versions=True)
        algorithm = GRPOAlgorithm(GRPOConfig(kl_coef=0.0))
        batch = ExperiencePreparer(algorithm).prepare(batch)
        magnitude = .5 / (math.sqrt(.5) + algorithm.config.advantage_epsilon)
        torch.testing.assert_close(batch.advantages,
                                   torch.tensor([[-magnitude, -magnitude], [magnitude, magnitude]]))
        torch.testing.assert_close(batch.old_log_probs, batch.rollout_log_probs)
        parameters = torch.tensor([.1, -.2, .3, .2, -.1], dtype=torch.float64, requires_grad=True)
        actions = batch.sequences[:, 1:]
        current = parameters.log_softmax(0)[actions]
        loss = algorithm.compute_actor_loss(current, batch.old_log_probs, None,
                                            batch.advantages, batch.loss_action_mask).total_loss_sum / 4
        loss.backward()

        def reference(weights):
            normalizer = math.log(sum(math.exp(weight) for weight in weights))
            result = 0.
            for row in range(2):
                advantage = -magnitude if row == 0 else magnitude
                for column in range(2):
                    ratio = math.exp(weights[int(actions[row, column])] - normalizer
                                     - float(batch.old_log_probs[row, column]))
                    objective = max(-advantage * ratio, -advantage * min(1.2, max(.8, ratio)))
                    result += min(objective, -3. * advantage) if advantage < 0 else objective
            return result / 4

        expected = []
        for index in range(5):
            plus, minus = parameters.detach().tolist(), parameters.detach().tolist()
            plus[index] += 1.e-6
            minus[index] -= 1.e-6
            expected.append((reference(plus) - reference(minus)) / 2.e-6)
        expected = torch.tensor(expected, dtype=torch.float64)
        torch.testing.assert_close(parameters.grad, expected, rtol=2.e-6, atol=1.e-8)
        before = parameters.detach().clone()
        torch.optim.SGD([parameters], lr=.01).step()
        torch.testing.assert_close(parameters, before - .01 * expected, rtol=2.e-6, atol=1.e-8)


class TestPartialService(unittest.IsolatedAsyncioTestCase):
    """Exercise engine output capture and publication exclusion without accelerator mocks."""

    async def test_abort_returns_original_tokens_and_resume_checks_version(self):
        aborted = asyncio.Event()

        class Engine:
            version = 1

            async def collective_rpc(self, method):
                return [{"version": self.version}]

            async def add_request(self, request_id, prompt, params):
                iterator = self.generate(prompt, params, request_id)
                return SimpleNamespace(request_id=request_id, get_nowait=lambda: None,
                                       get=iterator.__anext__, close=Mock())

            async def generate(self, prompt, params, request_id):
                yield SimpleNamespace(finished=False, outputs=[SimpleNamespace(token_ids=[7], logprobs=[
                    {7: SimpleNamespace(logprob=-.7)}], finish_reason=None)])
                await aborted.wait()
                yield SimpleNamespace(finished=True, outputs=[SimpleNamespace(token_ids=[7], logprobs=[
                    {7: SimpleNamespace(logprob=-.7)}], finish_reason="abort")])

            async def pause_generation(self, **kwargs):
                aborted.set()

            async def resume_generation(self):
                pass

        engine = Engine()
        service = PartialRequestService(engine, lambda **kwargs: kwargs)
        task = asyncio.create_task(service.generate({"policy_version": 1, "max_tokens": 4,
                                                     "prompt_token_ids": [4], "request_id": "a"}))
        await asyncio.sleep(0)
        self.assertEqual(await service.pause(), {"status": "paused"})
        result = await task
        self.assertEqual(result["token_ids"], [7])
        self.assertEqual(result["log_probs"], [-.7])
        self.assertEqual(result["policy_version"], 1)
        self.assertEqual(await service.generate({"policy_version": 1, "max_tokens": 4}), {"paused": True})
        with self.assertRaises(ValueError):
            await service.resume({"policy_version": 3})
        engine.version = 3
        self.assertEqual(await service.resume({"policy_version": 3}), {"status": "resumed"})
        # A request delayed across publication receives the actual engine version.
        result = await service.generate({"max_tokens": 4, "prompt_token_ids": [4], "request_id": "late"})
        self.assertEqual(result["policy_version"], 3)

    async def test_pause_before_first_output_and_cancel_cleanup(self):
        """A registered queued request must not hold admission while waiting for output."""
        for mode in ("pause", "cancel", "registration"):
            with self.subTest(mode=mode):
                entered, aborted = asyncio.Event(), asyncio.Event()

                async def get():
                    entered.set()
                    await aborted.wait()
                    return SimpleNamespace(finished=True, outputs=[SimpleNamespace(
                        token_ids=[], logprobs=None, finish_reason="abort")])

                async def generate(*args):
                    yield await get()

                async def add_request(*args):
                    if mode == "registration":
                        entered.set()
                        await aborted.wait()
                    return collector

                collector = SimpleNamespace(request_id="internal", get_nowait=lambda: None, get=get, close=Mock())
                engine = SimpleNamespace(
                    generate=generate, add_request=AsyncMock(side_effect=add_request), abort=AsyncMock(),
                    collective_rpc=AsyncMock(return_value=[{"version": 1}]),
                    pause_generation=AsyncMock(side_effect=lambda **kwargs: aborted.set()),
                )
                service = PartialRequestService(engine, lambda **kwargs: kwargs)
                task = asyncio.create_task(service.generate({"request_id": "queued", "max_tokens": 4,
                                                             "prompt_token_ids": [4]}))
                await asyncio.wait_for(entered.wait(), 1)
                if mode != "pause":
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                    engine.abort.assert_awaited_once_with(
                        "queued" if mode == "registration" else "internal", internal=mode != "registration",
                    )
                else:
                    self.assertEqual(await asyncio.wait_for(service.pause(), 1), {"status": "paused"})
                    self.assertEqual((await task)["token_ids"], [])
                    engine.abort.assert_not_awaited()
                if mode == "registration":
                    collector.close.assert_not_called()
                else:
                    collector.close.assert_called_once()
                self.assertEqual(service.active, 0)
                self.assertTrue(service.idle.is_set())

    def test_empty_abort_keeps_already_emitted_prefix(self):
        marker = SimpleNamespace(outputs=[SimpleNamespace(token_ids=[], logprobs=None, finish_reason="abort")])
        tokens, scores, reason = PartialRequestService._capture(marker, [7, 8], [-.7, -.8])
        self.assertEqual((tokens, scores, reason), ([7, 8], [-.7, -.8], "abort"))

    def test_engine_must_not_rewrite_an_emitted_prefix(self):
        output = SimpleNamespace(outputs=[SimpleNamespace(
            token_ids=[8], logprobs=[{8: SimpleNamespace(logprob=-.8)}], finish_reason="stop",
        )])
        with self.assertRaisesRegex(ValueError, "prefix"):
            PartialRequestService._capture(output, [7], [-.7])


class TestPartialPublication(unittest.TestCase):
    """Assert safe transaction ordering and complete-group checkpoint boundaries."""

    @staticmethod
    def _trainer():
        trainer = FullyAsyncTrainer.__new__(FullyAsyncTrainer)
        trainer.settings = AsyncTrainingConfig()
        trainer.roles = SimpleNamespace(state=SimpleNamespace(global_step=2),
                                        publish_policy=Mock(), tracker=Mock())
        trainer.sampler, trainer.trace = Mock(), Mock()
        return trainer

    def test_periodic_publication_pauses_and_checkpoint_publication_drains(self):
        for resume in (True, False):
            trainer = self._trainer()
            order = Mock()
            order.attach_mock(trainer.sampler.pause, "pause")
            order.attach_mock(trainer.sampler.drain, "drain")
            order.attach_mock(trainer.roles.publish_policy, "publish")
            order.attach_mock(trainer.sampler.resume, "resume")
            with self.subTest(resume=resume), patch(
                "rl.async_training.fully_async_trainer.coordinator_call", side_effect=lambda name, call: call(),
            ):
                trainer._publish(resume=resume)
            self.assertEqual([call[0] for call in order.mock_calls],
                             ["pause", "publish", "resume"] if resume else ["drain", "publish"])
            trainer.roles.publish_policy.assert_called_once_with(3)

    def test_failed_publication_never_releases_continuations(self):
        trainer = self._trainer()
        trainer.roles.publish_policy.side_effect = RuntimeError("transfer failed")
        with patch("rl.async_training.fully_async_trainer.coordinator_call", side_effect=lambda name, call: call()):
            with self.assertRaisesRegex(RuntimeError, "transfer failed"):
                trainer._publish()
        trainer.sampler.resume.assert_not_called()


class InterruptedHandler(BaseHTTPRequestHandler):
    """Force one candidate to finish before publication and one to cross it."""

    def log_message(self, *args):
        pass

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/collective_rpc":
            result = {"results": [{"version": self.server.version}]}
        elif self.path == "/hyper_async/pause":
            self.server.aborted.set()
            result = {"status": "paused"}
        elif self.path == "/hyper_async/resume":
            result = {"status": "resumed"}
        elif self.path == "/hyper_async/generate":
            version = self.server.version
            if version == 1:
                with self.server.lock:
                    ordinal = self.server.requests
                    self.server.requests += 1
                if ordinal == 0:
                    result = {"token_ids": [2, 3], "log_probs": [-.7, -.8], "finish_reason": "stop"}
                else:
                    self.server.started.set()
                    if not self.server.aborted.wait(20):
                        self.send_error(500)
                        return
                    result = {"token_ids": [2], "log_probs": [-.7], "finish_reason": "abort"}
            else:
                self.server.continuations.append(payload)
                result = {"token_ids": [3], "log_probs": [-.9], "finish_reason": "stop"}
            result["policy_version"] = version
        else:
            self.send_error(404)
            return
        content = json.dumps(result).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


class TestPartialProducer(unittest.TestCase):
    """Use the real spawned producer, HTTP client, AgentRunner and group transport."""

    def test_group_mixes_completed_and_resumed_candidates(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), InterruptedHandler)
        server.version, server.requests = 1, 0
        server.lock, server.started, server.aborted = threading.Lock(), threading.Event(), threading.Event()
        server.continuations = []
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
                    {"[PAD]": 0, "[EOS]": 1, "####": 2, "4": 3, "[UNK]": 4}, unk_token="[UNK]")),
                    pad_token="[PAD]", eos_token="[EOS]", unk_token="[UNK]")
                tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
                tokenizer.save_pretrained(root)
                pd.DataFrame({"prompt": ["question"], "answer": ["4"]}).to_parquet(root / "train.parquet")
                settings = AsyncTrainingConfig(max_inflight_groups=1,
                                               max_buffer_groups=2, max_group_bytes=65536,
                                               max_buffer_bytes=131072, operation_timeout=60)
                config = {
                    "async_training": asdict(settings),
                    "model": {"name": "qwen3", "registry_name": "served-qwen", "tokenizer_path": str(root)},
                    "data": {"train_path": str(root / "train.parquet"), "max_prompt_length": 32,
                             "prompt_column": "prompt", "answer_column": "answer", "shuffle": False},
                    "train": {"seed": 123},
                    "rollout": {"num_return_sequences": 2, "max_new_tokens": 8,
                                "vllm": {"port": server.server_port, "request_timeout": 30}},
                    "agentic": {"module_path": "examples.gsm8k.agent", "environment": "gsm8k_tools",
                                "max_turns": 1, "interaction_mode": "single_turn", "apply_chat_template": False},
                }
                buffer = ExperienceBuffer(settings, 2)
                runtime = AsyncRuntime(config, buffer)
                try:
                    runtime.resume(1)
                    runtime.fill(1, 0)
                    self.assertTrue(server.started.wait(30), "Producer never reached interruptible generation")
                    runtime.pause()
                    self.assertEqual(len(buffer.ready), 0, "Incomplete groups must not enter the ready buffer")
                    server.version = 3
                    runtime.resume(3)
                    runtime.drain()
                    value = buffer.select(1)[0]
                    self.assertEqual(sorted(row.policy_version for row in value.trajectories), [1, 3])
                    for row in value.trajectories:
                        self.assertEqual(row.token_ids[-2:].tolist(), [2, 3])
                    self.assertEqual(server.continuations[0]["prompt_token_ids"][-1], 2)
                    self.assertEqual(server.continuations[0]["max_tokens"], 7)
                    resumed = next(row for row in value.trajectories if row.policy_version == 3)
                    torch.testing.assert_close(resumed.rollout_log_probs[-2:], torch.tensor([-.7, -.9]))
                    self.assertEqual([s["policy_version"] for s in resumed.metadata["behavior_segments"]], [1, 3])
                    buffer.commit()
                    self.assertIn("loader", runtime.snapshot())
                finally:
                    runtime.close(graceful=True)
        finally:
            server.aborted.set()
            server.shutdown()
            server.server_close()
            worker.join(timeout=5)
