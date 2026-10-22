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
"""Actual producer, AgentRunner and HTTP parsing against a local scripted completion server."""
import json
import tempfile
import threading
import unittest
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from rl.async_training.runtime import ExperienceBuffer
from rl.async_training.protocol import AsyncTrainingConfig
from rl.async_training.protocol import decode_group
from rl.async_training.runtime import AsyncRuntime


class CompletionHandler(BaseHTTPRequestHandler):
    """Serve deterministic tokens and committed identity, without a vLLM installation."""

    def log_message(self, *args):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        if self.path == "/collective_rpc":
            response = {"results": [{"version": 1}]}
        elif self.path == "/hyper_async/generate":
            response = {"token_ids": [2, 3], "log_probs": [-.7, -.8],
                        "finish_reason": "stop", "policy_version": 1}
        elif self.path == "/hyper_async/pause":
            response = {"status": "paused"}
        elif self.path == "/hyper_async/resume":
            response = {"status": "resumed"}
        else:
            self.send_error(404)
            return
        content = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


class TestRealRollouter(unittest.TestCase):
    """Do not replace the production producer or agent orchestration in this test."""

    def test_complete_groups_keep_tokens_probabilities_and_source_cursor(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), CompletionHandler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                vocabulary = {"[PAD]": 0, "[EOS]": 1, "####": 2, "4": 3, "[UNK]": 4}
                backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
                backend.pre_tokenizer = Whitespace()
                tokenizer = PreTrainedTokenizerFast(
                    tokenizer_object=backend, pad_token="[PAD]", eos_token="[EOS]", unk_token="[UNK]",
                )
                tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
                tokenizer.save_pretrained(root)
                pd.DataFrame({"prompt": ["What is 2+2?", "What is 1+3?", "What is 3+1?"],
                              "answer": ["4", "4", "4"]}).to_parquet(root / "train.parquet")
                settings = AsyncTrainingConfig(max_inflight_groups=2, max_buffer_groups=2,
                                               max_group_bytes=65536, max_buffer_bytes=131072,
                                               operation_timeout=30)
                config = {
                    "async_training": asdict(settings),
                    "model": {"name": "qwen3", "registry_name": "served-qwen", "tokenizer_path": str(root)},
                    "data": {"train_path": str(root / "train.parquet"), "max_prompt_length": 32,
                             "prompt_column": "prompt", "answer_column": "answer", "shuffle": False},
                    "train": {"seed": 123},
                    "rollout": {"num_return_sequences": 2, "max_new_tokens": 8,
                                "vllm": {"port": server.server_port, "request_timeout": 10}},
                    "agentic": {"module_path": "examples.gsm8k.agent", "environment": "gsm8k_tools",
                                "max_turns": 1, "interaction_mode": "single_turn",
                                "apply_chat_template": False},
                }
                buffer = ExperienceBuffer(settings, 2)
                runtime = AsyncRuntime(config, buffer)
                try:
                    runtime.resume(1)
                    runtime.fill(1, 0)
                    runtime.drain()
                    self.assertEqual(len(buffer.ready), 2)
                    snapshot = runtime.snapshot()
                    self.assertIn("loader", snapshot)
                    for payload in buffer.ready.values():
                        group = decode_group(payload, 2, 65536)
                        for row in group.trajectories:
                            self.assertEqual(row.token_ids[-2:].tolist(), [2, 3])
                            self.assertEqual(row.worker_policy_version, 1)
                            self.assertEqual(row.rollout_log_probs[-2:].tolist(),
                                             [-0.699999988079071, -0.800000011920929])
                    buffer.select(2)
                    buffer.commit()
                    runtime.close(graceful=True)
                    runtime = AsyncRuntime(config, buffer, source=snapshot)
                    runtime.resume(1)
                    runtime.fill(1, 0)
                    runtime.drain()
                    restored = {identity: decode_group(payload, 2, 65536)
                                for identity, payload in buffer.ready.items()}
                    self.assertEqual(str(restored[2].trajectories[0].metadata["source_prompt_id"]), "2")
                    self.assertEqual(str(restored[3].trajectories[0].metadata["source_prompt_id"]), "0")
                finally:
                    runtime.close(graceful=True)
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=5)
