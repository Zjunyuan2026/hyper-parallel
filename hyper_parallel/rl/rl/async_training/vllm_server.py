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
"""Opt-in vLLM routes that return aborted tokens instead of dropping partial output."""
import asyncio
import math
import os
from functools import partial, wraps
from importlib.metadata import version as package_version
from typing import Any

from rl.utils.async_audit import EventTrace


class PartialRequestService:
    """Serialize engine admission with pause and capture cumulative engine output."""

    def __init__(self, engine: Any, sampling_factory: Any, trace: EventTrace | None = None) -> None:
        self.engine, self.sampling_factory = engine, sampling_factory
        self.trace = trace or EventTrace(None, "inference")
        self.lock = asyncio.Lock()
        self.closed = False
        self.active = 0
        self.committed_version: int | None = None
        self.idle = asyncio.Event()
        self.idle.set()

    async def version(self) -> int:
        """Require agreement across every inference worker."""
        results = await self.engine.collective_rpc("get_policy_version")
        if not results:
            raise ValueError("Inference policy query returned no worker results")
        versions = {row["version"] for row in results}
        if len(versions) != 1 or next(iter(versions)) < 1:
            raise ValueError("Inference workers have no common committed policy")
        return next(iter(versions))

    async def generate(self, payload: dict) -> dict:
        """Register under admission, then collect output without blocking publication."""
        params = self.sampling_factory(
            n=1, temperature=1.0, top_p=1.0, top_k=-1, logprobs=1,
            max_tokens=payload["max_tokens"], seed=payload.get("seed"),
            ignore_eos=payload.get("ignore_eos", False),
            stop_token_ids=[] if payload.get("ignore_eos") else payload.get("stop_token_ids", []),
        )
        collector = None
        registered = finished = False
        try:
            async with self.lock:
                if self.closed:
                    return {"paused": True}
                if self.committed_version is None:
                    self.committed_version = await self.version()
                version = self.committed_version
                self.trace.emit("segment_start", segment_id=payload["request_id"], policy_version=version)
                self.active += 1
                registered = True
                self.idle.clear()
                # Registration must precede pause; waiting for a scheduled token must not.
                collector = await self.engine.add_request(
                    payload["request_id"], {"prompt_token_ids": payload["prompt_token_ids"]}, params,
                )
            tokens, scores, reason = [], [], None
            while not finished:
                output = collector.get_nowait() or await collector.get()
                finished = output.finished
                tokens, scores, reason = self._capture(output, tokens, scores)
            return {"token_ids": tokens, "log_probs": scores,
                    "finish_reason": reason, "policy_version": version}
        finally:
            try:
                try:
                    if registered and not finished:
                        await self.engine.abort(
                            collector.request_id if collector is not None else payload["request_id"],
                            internal=collector is not None,
                        )
                finally:
                    if collector is not None:
                        collector.close()
            finally:
                if registered:
                    self.trace.emit("segment_end", segment_id=payload["request_id"], policy_version=version)
                    self.active -= 1
                    if self.active == 0:
                        self.idle.set()

    @staticmethod
    def _capture(output: Any, previous_tokens: list, previous_scores: list) -> tuple:
        """Check cumulative output while retaining the prefix on an empty abort marker."""
        if len(output.outputs) != 1:
            raise ValueError("Partial request did not return one candidate")
        completion = output.outputs[0]
        tokens, probabilities = list(completion.token_ids), completion.logprobs
        reason = completion.finish_reason
        if not tokens and reason in ("abort", "aborted"):
            return previous_tokens, previous_scores, reason
        if probabilities is None or len(probabilities) != len(tokens):
            raise ValueError("Engine partial probabilities are missing or misaligned")
        scores = [float(values[token].logprob) for token, values in zip(tokens, probabilities)]
        if any(not math.isfinite(score) or score > 1.e-5 for score in scores):
            raise ValueError("Engine returned invalid behavior probabilities")
        if tokens[:len(previous_tokens)] != previous_tokens or scores[:len(previous_scores)] != previous_scores:
            raise ValueError("Engine changed an already emitted token/probability prefix")
        return tokens, scores, reason

    async def pause(self) -> dict:
        """Close admission, abort running generation and await captured partial output."""
        async with self.lock:
            self.closed = True
            self.committed_version = None
            await self.engine.pause_generation(mode="abort", clear_cache=True)
        await self.idle.wait()
        return {"status": "paused"}

    async def resume(self, payload: dict) -> dict:
        """Verify the publication before reopening segment admission."""
        async with self.lock:
            if self.active or await self.version() != payload["policy_version"]:
                raise ValueError("Cannot resume partial requests before publication completes")
            await self.engine.resume_generation()
            self.committed_version = payload["policy_version"]
            self.closed = False
        return {"status": "resumed"}


def install_async_routes() -> None:
    """Register routes only in explicitly enabled asynchronous vLLM services."""
    runtime = (package_version("vllm").split("+", maxsplit=1)[0],
               package_version("vllm-ascend").split("+", maxsplit=1)[0])
    if runtime != ("0.23.0", "0.23.0.post1"):
        raise ValueError(f"Partial rollout requires pinned vLLM 0.23.0 / Ascend 0.23.0.post1, got {runtime}")
    # Optional vLLM/FastAPI dependencies exist only in the inference service.
    from fastapi import Request  # pylint: disable=C0415
    from vllm import SamplingParams  # pylint: disable=C0415
    from vllm.entrypoints import serve  # pylint: disable=C0415
    from vllm.sampling_params import RequestOutputKind  # pylint: disable=C0415

    original = serve.register_vllm_serve_api_routers
    if getattr(original, "hyper_async_routes", False):
        return

    @wraps(original)
    def register(app: Any) -> None:
        original(app)

        def service(request: Any) -> PartialRequestService:
            if not hasattr(request.app.state, "hyper_async_service"):
                request.app.state.hyper_async_service = PartialRequestService(
                    request.app.state.engine_client, partial(SamplingParams, output_kind=RequestOutputKind.CUMULATIVE),
                    EventTrace(os.environ.get("HYPER_RL_ASYNC_AUDIT_DIR"), "inference"),
                )
            return request.app.state.hyper_async_service

        async def generate(request: Request) -> dict:
            return await service(request).generate(await request.json())

        async def pause(request: Request) -> dict:
            return await service(request).pause()

        async def resume(request: Request) -> dict:
            return await service(request).resume(await request.json())

        app.add_api_route("/hyper_async/generate", generate, methods=["POST"])
        app.add_api_route("/hyper_async/pause", pause, methods=["POST"])
        app.add_api_route("/hyper_async/resume", resume, methods=["POST"])

    register.hyper_async_routes = True
    serve.register_vllm_serve_api_routers = register


if __name__ in ("__main__", "__mp_main__"):
    # Spawned inference workers must register the same hooks before deserializing their targets.
    # vLLM is optional outside this explicitly selected inference entry.
    from rl.roles.rollout.vllm_plugin import register_hyper_models

    os.environ["HYPER_RL_ASYNC_TRAINING"] = "1"
    register_hyper_models()
    if __name__ == "__main__":
        from vllm.entrypoints.cli.main import main

        main()
