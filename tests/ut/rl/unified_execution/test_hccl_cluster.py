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
"""CPU-only checks for cluster probe evidence and failure cleanup."""

import argparse
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import shlex
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[4] / "hyper_parallel/rl/examples/consistency/scripts/run_hccl_cluster.py"
SPEC = importlib.util.spec_from_file_location("hccl_controller", SOURCE)
CONTROLLER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTROLLER)


class ClusterProbeTests(unittest.TestCase):
    """Reject incomplete evidence and preserve cleanup after partial startup."""

    def test_every_rank_and_message_size_is_required(self) -> None:
        """Missing/duplicate ranks and invalid timing must never become success."""
        records = [{"rank": rank, "local_rank": rank % 16, "world_size": 128, "passed": True,
                    "measurements": [{"bytes": size, "mean_seconds": 0.01}
                                     for size in (4, 4194304, 67108864)]} for rank in range(128)]

        def collect(rows: list[dict]) -> dict:
            return CONTROLLER.collect_results(["HCCL_PROBE_RESULT " + json.dumps(row) for row in rows], 128, 16)

        self.assertTrue(collect(records)["passed"])
        self.assertFalse(collect(records[:-1])["passed"])
        self.assertFalse(collect(records + [records[0]])["passed"])
        for key, value in (("world_size", 8), ("local_rank", 5), ("passed", False)):
            changed = copy.deepcopy(records)
            changed[0][key] = value
            self.assertFalse(collect(changed)["passed"])
        changed = copy.deepcopy(records)
        changed[0]["measurements"][1]["mean_seconds"] = float("nan")
        self.assertFalse(collect(changed)["passed"])

    def test_docker_launch_preserves_worker_payload_and_global_topology(self) -> None:
        """Shell quoting must preserve the Python entry and all sixteen local workers."""
        args = argparse.Namespace(hosts=["192.0.2.1", "192.0.2.2"], nproc_per_node=16,
                                  master_addr="198.51.100.1", master_port=29651, interface="data0.172")
        command = CONTROLLER.docker_command(args, 1, "198.51.100.2", "sha256:example", "owned", "cGF5bG9hZA==")
        self.assertIn("HCCL_NPU_SOCKET_PORT_RANGE=auto", command)
        self.assertIn("HP_HCCL_WORKER=cGF5bG9hZA==", command)
        entry = shlex.split(command[-1].split("; exec ", 1)[1])
        self.assertIn("--nproc_per_node=16", entry)
        self.assertIn("--nnodes=2", entry)
        self.assertIn("--node_rank=1", entry)
        self.assertEqual(entry[-2], "-c")
        compile(entry[-1], "probe_entry", "exec")

    def test_busy_node_aborts_preflight(self) -> None:
        """Any observed NPU process prevents startup."""
        args = argparse.Namespace(hosts=["192.0.2.1"], interface="data0.172", master_addr="198.51.100.1",
                                  image="image")
        outputs = ["192.0.2.1", "sha256:example",
                   "Process id\n| 0 0 123 other_task |\nNo running processes found in NPU 1"]
        with patch.object(CONTROLLER, "execute", side_effect=[SimpleNamespace(stdout=text) for text in outputs]):
            with self.assertRaisesRegex(RuntimeError, "NPUs busy"):
                CONTROLLER.preflight(args)

    def test_log_collection_failure_does_not_skip_container_cleanup(self) -> None:
        """A partially launched run removes only the generated names even if logs fail."""
        commands = []

        def execute(host: str, local: bool, command: list[str]) -> SimpleNamespace:
            del host, local
            commands.append(command)
            if command[1] in ("run", "logs"):
                raise subprocess.CalledProcessError(1, command)
            return SimpleNamespace(stdout="", stderr="")

        with tempfile.TemporaryDirectory() as temporary:
            argv = [str(SOURCE), "--hosts", "192.0.2.1", "--master-addr", "198.51.100.1",
                    "--interface", "data0.172", "--result-base", temporary]
            with patch("sys.argv", argv), patch.object(CONTROLLER, "preflight",
                                                       return_value=("sha256:example", ["198.51.100.1"])), \
                 patch.object(CONTROLLER, "execute", side_effect=execute), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(subprocess.CalledProcessError):
                    CONTROLLER.main()
            removals = [command for command in commands if command[:3] == ["docker", "rm", "-f"]]
            self.assertEqual(len(removals), 1)
            self.assertTrue(removals[0][-1].startswith("hp-hccl-"))
            summary = json.loads(next(Path(temporary).glob("*/summary.json")).read_text())
            self.assertFalse(summary["passed"])


if __name__ == "__main__":
    unittest.main()
