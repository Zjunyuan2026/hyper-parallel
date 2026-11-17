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
"""Real HTTP regression for long-generation and control deadline separation."""

import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import unittest
from unittest.mock import patch

import yaml

from hyper_parallel.rl.examples.consistency.scripts.configure_run import configure
from rl.rollout.vllm import VLLMHTTPClient


class _DelayedCompletion(BaseHTTPRequestHandler):
    """Return a complete response after the shorter control budget has expired."""

    def do_POST(self) -> None:  # pylint: disable=invalid-name
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        time.sleep(0.2)
        body = json.dumps({"choices": [{"index": 0, "token_ids": [17],
                                        "logprobs": {"token_logprobs": [-0.5]}}]}).encode()
        try:
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except OSError:
            pass  # A deliberately timed-out test client has already disconnected.

    def log_message(self, unused_format: str, *args) -> None:
        """Keep request logging out of test output."""


class TestGenerationTimeout(unittest.TestCase):
    """Keep a finite generation deadline without extending weight-control calls."""

    def setUp(self) -> None:
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _DelayedCompletion)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.thread.join, 2)
        self.addCleanup(self.server.shutdown)
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def request(self, generation_timeout=None):
        """Exercise the actual persistent aiohttp transport against localhost."""
        client = VLLMHTTPClient(None, self.url, "model", 0.03, generation_timeout=generation_timeout)
        self.addCleanup(client._close_async_runtime)  # pylint: disable=protected-access
        loop = client._ensure_async_runtime(1)  # pylint: disable=protected-access
        future = asyncio.run_coroutine_threadsafe(
            client._request_async("POST", "v1/completions", {"prompt": [[1]]}), loop,  # pylint: disable=protected-access
        )
        return client, future

    def test_long_generation_outlives_control_budget(self) -> None:
        client, future = self.request(2)
        result = future.result(timeout=5)
        self.assertEqual(result["choices"][0]["token_ids"], [17])
        self.assertEqual(result["choices"][0]["logprobs"]["token_logprobs"], [-0.5])
        self.assertEqual(client._request_timeout, 0.03)  # pylint: disable=protected-access
        with patch("rl.rollout.vllm.urllib_request.urlopen", side_effect=OSError("control probe")) as opening:
            with self.assertRaisesRegex(OSError, "control probe"):
                client._request("POST", "collective_rpc", {})  # pylint: disable=protected-access
        self.assertEqual(opening.call_args.kwargs["timeout"], 0.03)

    def test_omitted_generation_budget_preserves_existing_timeout(self) -> None:
        _, future = self.request()
        with self.assertRaisesRegex(RuntimeError, "request_timeout=0.03s") as caught:
            future.result(timeout=5)
        self.assertIsInstance(caught.exception.__cause__, asyncio.TimeoutError)

    def test_generation_deadline_is_still_enforced(self) -> None:
        _, future = self.request(0.05)
        with self.assertRaisesRegex(RuntimeError, "request_timeout=0.05s"):
            future.result(timeout=5)

    def test_invalid_generation_budget_is_rejected(self) -> None:
        for value in (0, -1, float("nan"), float("inf"), True):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite and positive"):
                VLLMHTTPClient(None, self.url, "model", 600, generation_timeout=value)

    def test_a3_entry_exports_default_and_explicit_generation_budgets(self) -> None:
        root = Path(__file__).resolve().parents[4]
        for value, expected in ((None, "1800"), ("2400", "2400")):
            env = dict(os.environ)
            env.pop("GENERATION_TIMEOUT", None)
            if value is not None:
                env["GENERATION_TIMEOUT"] = value
            result = subprocess.run(["bash", str(root / "run_a3.sh"), "--dry-run"],
                                    env=env, text=True, capture_output=True, check=True)
            self.assertIn(f"GENERATION_TIMEOUT={expected}\n", result.stdout)

    def test_recipe_keeps_control_timeout_and_sampling_settings(self) -> None:
        root = Path(__file__).resolve().parents[4]
        path = root / "hyper_parallel/rl/examples/consistency/configs/qwen3_moe_4l_rl.yaml"
        with patch.dict(os.environ, {"RUN_MODEL_PATH": "/models/full", "GENERATION_TIMEOUT": "1800"}, clear=True):
            config = configure(yaml.safe_load(path.read_text()), 16)
        self.assertEqual(config["rollout"]["vllm"]["generation_timeout"], 1800)
        self.assertEqual(config["rollout"]["vllm"]["request_timeout"], 600)
        self.assertEqual(config["rollout"]["max_new_tokens"], 1024)
