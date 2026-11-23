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
"""CPU-only launcher regressions for selected-node RL training."""

import argparse
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "hyper_parallel/rl/examples/consistency/scripts/run_qwen3_moe_cluster.py"
with patch.object(sys, "path", [str(SCRIPT.parent), *sys.path]):
    SPEC = importlib.util.spec_from_file_location("rl_cluster_controller", SCRIPT)
    CONTROLLER = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(CONTROLLER)


def arguments(directory: Path) -> argparse.Namespace:
    """Use documentation-only addresses and an isolated result directory."""
    return argparse.Namespace(hosts=["192.0.2.1", "192.0.2.5"], select_nodes=2,
                              master_addr="192.0.2.1", master_port=29751, interface="data0.172",
                              model_root=directory / "model", data_root=directory / "data",
                              shared_root=directory, wheel=directory / "wheel", cpu_offload=False,
                              steps=3, max_new_tokens=1024, timeout=3600)


class ClusterLauncherTests(unittest.TestCase):
    """Validate placement, completeness, and bounded failure cleanup without NPU use."""

    def test_container_uses_full_model_and_node_local_rollout(self) -> None:
        """Selected host order drives rank assignment; global and local sizes remain distinct."""
        args = arguments(Path("/shared/run"))
        with patch.object(os, "getuid", return_value=0, create=True), \
             patch.object(os, "getgid", return_value=0, create=True):
            command = CONTROLLER.docker_command(args, 1, "172.27.3.105", "sha256:abc", "owned", Path("/shared/run"))
        env = dict(value.split("=", 1) for index, value in enumerate(command) if index and command[index - 1] == "-e")
        self.assertEqual(env["NODE_RANK"], "1")
        self.assertEqual(env["MODEL_LAYERS"], "0")
        self.assertEqual(env["NNODES"], "2")
        self.assertEqual(env["ROLLOUT_LOCAL_WORLD_SIZE"], "16")
        self.assertEqual(env["CPU_OFFLOAD"], "false")
        self.assertEqual(env["GLOO_SOCKET_IFNAME"], "data0.172")
        self.assertEqual(command[-1], "--track-updates")

    def test_startup_failure_removes_only_its_own_containers(self) -> None:
        """Failure on a later node releases the first node and preserves a failure summary."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = arguments(root)
            commands = []

            def stage(directory, _wheel):
                return directory / "source", "revision"

            def execute(host, _local, command):
                commands.append(command)
                if command[:2] == ["docker", "run"] and host == args.hosts[1]:
                    raise RuntimeError("second node launch failed")
                output = Path(command[1]).read_text() if command[0] == "cat" else ""
                return SimpleNamespace(stdout=output, stderr="")

            with patch.object(CONTROLLER, "preflight", return_value=("sha256:abc", args.hosts)), \
                 patch.object(CONTROLLER, "stage_source", side_effect=stage), \
                 patch.object(CONTROLLER, "execute", side_effect=execute), \
                 patch.object(CONTROLLER, "require_idle"), \
                 patch.object(os, "getuid", return_value=0, create=True), \
                 patch.object(os, "getgid", return_value=0, create=True):
                with self.assertRaisesRegex(RuntimeError, "second node"):
                    CONTROLLER.run(args)
            removed = [command for command in commands if command[:3] == ["docker", "rm", "-f"]]
            self.assertEqual(len(removed), 2)
            self.assertTrue(all(command[3].startswith("hp-rl-") for command in removed))
            summary = next(root.glob("qwen3-cluster-*/summary.json"))
            self.assertFalse(json.loads(summary.read_text())["passed"])

    def test_dry_run_does_not_contact_remote_hosts(self) -> None:
        """Print the 64-device plan without importing Torch or starting subprocess SSH."""
        command = [str(SCRIPT), "--hosts", "192.0.2.1,192.0.2.2,192.0.2.3,192.0.2.4",
                   "--master-addr", "172.27.3.101", "--dry-run"]
        output = io.StringIO()
        with patch.object(sys, "argv", command), contextlib.redirect_stdout(output), \
             patch.object(CONTROLLER, "execute", side_effect=AssertionError("unexpected remote call")):
            CONTROLLER.main()
        plan = json.loads(output.getvalue())
        self.assertEqual(plan["world_size"], 64)
        self.assertEqual(plan["training"], "TP4/CP2/DP8/EP4/EDP16")
        self.assertEqual(plan["model_layers"], "full")
        self.assertFalse(plan["cpu_offload"])

    def test_success_requires_complete_steps_and_successful_exits_on_all_nodes(self) -> None:
        """A rank-zero success file alone must not hide partial training or a failed peer."""
        cases = ((3, True, 0, True), (2, True, 0, False), (3, False, 0, False), (3, True, 1, False))
        for completed, reported_pass, peer_exit, expected in cases:
            with self.subTest(completed=completed, reported_pass=reported_pass, peer_exit=peer_exit), \
                 tempfile.TemporaryDirectory() as temporary:
                args = arguments(Path(temporary))
                active_directory = []

                def stage(directory, _wheel):
                    active_directory.append(directory)
                    return directory / "source", "revision"

                def execute(host, _local, command):
                    output = ""
                    if command[0] == "cat":
                        output = Path(command[1]).read_text()
                    if command[:2] == ["docker", "inspect"]:
                        report = {"passed": reported_pass, "completed_steps": completed,
                                  "checks": [{}] * completed}
                        (active_directory[0] / "node0/run/consistency.json").write_text(json.dumps(report))
                        output = json.dumps({"Running": False, "ExitCode": 0 if host == args.hosts[0] else peer_exit})
                    return SimpleNamespace(stdout=output, stderr="")

                with patch.object(CONTROLLER, "preflight", return_value=("sha256:abc", args.hosts)), \
                     patch.object(CONTROLLER, "stage_source", side_effect=stage), \
                     patch.object(CONTROLLER, "execute", side_effect=execute), \
                     patch.object(CONTROLLER, "require_idle"), \
                     patch.object(os, "getuid", return_value=0, create=True), \
                     patch.object(os, "getgid", return_value=0, create=True):
                    if expected:
                        CONTROLLER.run(args)
                    else:
                        with self.assertRaisesRegex(RuntimeError, "Cluster training failed"):
                            CONTROLLER.run(args)
                summary = json.loads((active_directory[0] / "summary.json").read_text())
                self.assertEqual(summary["passed"], expected)
