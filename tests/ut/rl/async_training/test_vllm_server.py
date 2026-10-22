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
"""Exercise the inference entry and its actual spawn child without installing a package."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from rl.async_training import vllm_server


class TestInferenceEntry(unittest.TestCase):
    """Check process-local registration before vLLM dispatch and failure propagation."""

    def test_spawn_workers_register_without_distribution_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            modules = {
                "rl/async_training/vllm_server.py": Path(vllm_server.__file__).read_text(encoding="utf-8"),
                "rl/utils/async_audit.py": "class EventTrace: pass\n",
                "rl/roles/rollout/vllm_plugin.py": (
                    "import os\n"
                    "def register_hyper_models():\n"
                    "    if os.environ.get('FAIL_REGISTER'): raise RuntimeError('registration failed')\n"
                    "    os.environ['REGISTERED_PID'] = str(os.getpid())\n"
                ),
                "vllm/entrypoints/cli/main.py": (
                    "import json, multiprocessing, os\n"
                    "def worker():\n"
                    "    with open(os.environ['RECORD'], 'a') as stream:\n"
                    "        stream.write(json.dumps({'pid': os.getpid(), 'registered': "
                    "os.environ.get('REGISTERED_PID')}) + '\\n')\n"
                    "def main():\n"
                    "    worker()\n"
                    "    child = multiprocessing.get_context('spawn').Process(target=worker)\n"
                    "    child.start()\n"
                    "    child.join(20)\n"
                    "    if child.is_alive(): child.kill(); raise RuntimeError('spawn timeout')\n"
                    "    if child.exitcode: raise RuntimeError('spawn failed')\n"
                ),
            }
            for name, content in modules.items():
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
                for parent in target.parents:
                    if parent == root:
                        break
                    (parent / "__init__.py").touch()
            record = root / "records.jsonl"
            environment = {**os.environ, "PYTHONPATH": str(root), "RECORD": str(record)}
            command = [sys.executable, "-m", "rl.async_training.vllm_server"]
            subprocess.run(command, cwd=root, env=environment, check=True, capture_output=True, timeout=30)
            records = [json.loads(line) for line in record.read_text().splitlines()]
            self.assertEqual(len(records), 2)
            self.assertNotEqual(records[0]["pid"], records[1]["pid"])
            self.assertTrue(all(str(row["pid"]) == row["registered"] for row in records))
            record.unlink()
            environment["FAIL_REGISTER"] = "1"
            failed = subprocess.run(command, cwd=root, env=environment, capture_output=True, timeout=30)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn(b"registration failed", failed.stderr)
            self.assertFalse(record.exists())

    def test_launcher_preserves_native_pythonpath(self):
        launcher = Path(vllm_server.__file__).parents[2] / "examples/async_training/scripts/run_async_training.sh"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake_python = root / "python"
            fake_python.write_text(
                '#!/bin/bash\nprintf "%s\\n" "$PYTHONPATH" "$PYTHONDONTWRITEBYTECODE" "$@" > "$RECORD"\n',
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            for original in ("", "/cann/python:/cann/opp"):
                environment = {**os.environ, "PYTHONPATH": original, "PATH": f"{root}:{os.environ['PATH']}",
                               "RECORD": str(root / "command")}
                subprocess.run(["bash", str(launcher), "/external/config.yaml", "--train.max_steps=2"],
                               env=environment, check=True, capture_output=True, timeout=10)
                lines = (root / "command").read_text().splitlines()
                prefix = f"{launcher.parents[3]}:{launcher.parents[5]}"
                self.assertEqual(lines[0], prefix + (f":{original}" if original else ""))
                self.assertEqual(lines[1:4], ["1", "-m", "torch.distributed.run"])
                self.assertEqual(lines[-2:], ["/external/config.yaml", "--train.max_steps=2"])
