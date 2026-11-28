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
"""Exercise the shared-path shell setup without hardware or existing shared storage."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "hyper_parallel/rl/examples/gsm8k/scripts/configure_qwen3_shared_paths.sh"
BASH = ("C:/Program Files/Git/bin/bash.exe" if os.name == "nt" else shutil.which("bash"))


@unittest.skipUnless(BASH and Path(BASH).is_file(), "Bash is required for shell path contracts")
class SharedPathTests(unittest.TestCase):
    def run_setup(self, root, overrides=None):
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith("GSM8K_")}
        environment.update(overrides or {})
        environment["GSM8K_CACHE_NODE"] = "node101"
        command = (
            'set -eu; if command -v cygpath >/dev/null; then '
            'export GSM8K_SHARE_ROOT=$(cygpath -u "$1"); else export GSM8K_SHARE_ROOT="$1"; fi; '
            'source "$2"; '
            'for name in GSM8K_SHARE_ROOT GSM8K_MODEL GSM8K_DATA GSM8K_RESULTS GSM8K_IMAGES '
            'GSM8K_COMPARE_WITH PIP_CACHE_DIR VLLM_CACHE_ROOT ASCEND_CACHE_PATH; do '
            'printf "%s=%s\\n" "$name" "${!name-}"; done'
        )
        return subprocess.run([BASH, "-c", command, "paths-test", str(root), str(SCRIPT)],
                              env=environment, capture_output=True, text=True, check=False)

    def test_defaults_and_legacy_paths_share_new_root(self):
        with tempfile.TemporaryDirectory(dir=ROOT.parent) as temporary:
            root = Path(temporary)
            result = self.run_setup(root, {"GSM8K_DATA": "/mnt/share/zjy/datasets/existing",
                "GSM8K_COMPARE_WITH": "/mnt/share/zjy/rl-cluster-runs/previous"})
            self.assertEqual(result.returncode, 0, f"status={result.returncode}, stderr={result.stderr}")
            values = dict(line.split("=", 1) for line in result.stdout.splitlines())
            shared = values["GSM8K_SHARE_ROOT"]
            self.assertEqual(values["GSM8K_DATA"], shared + "/datasets/existing")
            self.assertEqual(values["GSM8K_COMPARE_WITH"], shared + "/rl-cluster-runs/previous")
            self.assertEqual(values["GSM8K_MODEL"], shared + "/models/Qwen3-30B-A3B")
            self.assertEqual(values["GSM8K_IMAGES"], shared + "/images")
            self.assertEqual(values["VLLM_CACHE_ROOT"], shared + "/cache/node101/vllm")
            self.assertTrue((root / "cache/node101/ascend").is_dir())
            self.assertTrue((root / "cache/pip").is_dir())

    def test_missing_share_is_rejected_without_creating_local_fallback(self):
        with tempfile.TemporaryDirectory(dir=ROOT.parent) as temporary:
            missing = Path(temporary) / "unmounted"
            result = self.run_setup(missing)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Shared directory is unavailable", result.stderr)
            self.assertFalse(missing.exists())
