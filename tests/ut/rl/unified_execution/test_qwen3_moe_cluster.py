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
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
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
                              steps=3, max_new_tokens=1024, timeout=3600,
                              global_prompt_batch_size=None, micro_batch_size=1, response_mini_batch_size=None,
                              full_data=False, plot=False, formal_gsm8k=False, max_prompt=512)


class ClusterLauncherTests(unittest.TestCase):
    """Validate placement, completeness, and bounded failure cleanup without NPU use."""

    def test_native_baseline_selects_non_strict_runner(self) -> None:
        """Native pilots retain the idle-node selector without claiming numerical parity."""
        command = [str(SCRIPT), "--hosts", "192.0.2.1,192.0.2.2,192.0.2.3,192.0.2.4",
                   "--master-addr", "198.51.100.1", "--native-baseline", "--micro-batch-size", "2", "--dry-run"]
        stream = io.StringIO()
        with patch.object(sys, "argv", command), patch.object(CONTROLLER, "preflight") as probe, \
             contextlib.redirect_stdout(stream):
            CONTROLLER.main()
        result = json.loads(stream.getvalue())
        self.assertEqual(result["steps"], 2)
        self.assertFalse(result["strict_consistency"])
        self.assertFalse(result["cpu_offload"])
        self.assertEqual(result["batch"]["global_prompt_batch_size"], 128)
        probe.assert_not_called()
        args = arguments(Path("/shared/run"))
        args.native_baseline = True
        args.hosts = ["192.0.2.1", "192.0.2.2", "192.0.2.3", "192.0.2.4"]
        args.global_prompt_batch_size = 128
        args.micro_batch_size = 2
        with patch.object(os, "getuid", return_value=0, create=True), \
             patch.object(os, "getgid", return_value=0, create=True):
            docker = CONTROLLER.docker_command(args, 0, "198.51.100.1", "image", "owned", Path("/shared/run"))
        env = dict(value.split("=", 1) for index, value in enumerate(docker) if index and docker[index - 1] == "-e")
        self.assertEqual(env["NATIVE_BASELINE"], "true")
        self.assertEqual(env["CPU_OFFLOAD"], "false")
        self.assertEqual(env["TRAIN_SP"], "false")
        self.assertNotIn("--track-updates", docker)

    def test_baseline_pilot_dry_run_and_node_environment(self) -> None:
        """The pilot has eight answers, local rollout and independent immutable data storage."""
        command = [str(SCRIPT), "--hosts", "192.0.2.1,192.0.2.2,192.0.2.3,192.0.2.4",
                   "--master-addr", "198.51.100.1", "--baseline-5step", "--micro-batch-size", "2", "--dry-run"]
        stream = io.StringIO()
        with patch.object(sys, "argv", command), patch.object(CONTROLLER, "preflight") as probe, \
             contextlib.redirect_stdout(stream):
            CONTROLLER.main()
        result = json.loads(stream.getvalue())
        self.assertEqual(result["steps"], 5)
        self.assertEqual(result["batch"]["global_prompt_batch_size"], 128)
        self.assertEqual(result["batch"]["global_response_batch_size"], 1024)
        self.assertEqual(result["max_num_batched_tokens"], 1024)
        self.assertEqual(result["batch"]["micro_batch_size"], 2)
        probe.assert_not_called()
        stream = io.StringIO()
        with patch.object(sys, "argv", command + ["--steps", "50"]), \
             patch.object(CONTROLLER, "preflight") as probe, contextlib.redirect_stdout(stream):
            CONTROLLER.main()
        full = json.loads(stream.getvalue())
        self.assertEqual(full["steps"], 50)
        self.assertEqual(full["timeout"], 86400)
        self.assertTrue(full["strict_consistency"])
        self.assertEqual(full["batch"], result["batch"])
        probe.assert_not_called()
        args = arguments(Path("/shared/run"))
        args.baseline_5step = True
        args.hosts = ["192.0.2.1", "192.0.2.2", "192.0.2.3", "192.0.2.4"]
        args.global_prompt_batch_size = 128
        args.micro_batch_size = 2
        with patch.object(os, "getuid", return_value=0, create=True), \
             patch.object(os, "getgid", return_value=0, create=True):
            docker = CONTROLLER.docker_command(args, 0, "198.51.100.1", "image", "owned", Path("/shared/run"))
        env = dict(value.split("=", 1) for index, value in enumerate(docker) if index and docker[index - 1] == "-e")
        self.assertEqual(env["BASELINE_5STEP"], "true")
        self.assertEqual(env["ROLLOUT_EP"], "16")
        self.assertEqual(env["PREFILL_TOKENS"], "1024")
        self.assertEqual(env["CANDIDATES"], "8")
        self.assertEqual(env["ROLLOUT_LOCAL_WORLD_SIZE"], "16")
        self.assertIn("/shared/run/baseline:/baseline-run:rw", [value.replace("\\", "/") for value in docker])

    def test_formal_defaults_are_explicit_and_dry_run_allocates_nothing(self) -> None:
        """The formal flag retains measured settings and cannot masquerade as a short pilot."""
        command = [str(SCRIPT), "--hosts", "192.0.2.1,192.0.2.2,192.0.2.3,192.0.2.4",
                   "--master-addr", "198.51.100.1", "--formal-gsm8k", "--dry-run"]
        stream = io.StringIO()
        with patch.object(sys, "argv", command), patch.object(CONTROLLER, "preflight") as probe, \
             contextlib.redirect_stdout(stream):
            CONTROLLER.main()
        result = json.loads(stream.getvalue())
        self.assertEqual(result["steps"], 50)
        self.assertEqual(result["batch"]["global_prompt_batch_size"], 136)
        self.assertEqual(result["batch"]["micro_batch_size"], 1)
        self.assertEqual(result["timeout"], 86400)
        self.assertEqual(result["max_prompt_tokens"], 512)
        self.assertEqual(result["max_new_tokens"], 1024)
        self.assertEqual(result["max_model_len"], 1536)
        self.assertTrue(result["plot"])
        probe.assert_not_called()
        for extra in (["--micro-batch-size", "2"], ["--steps", "5"], ["--cpu-offload"]):
            with patch.object(sys, "argv", command + extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    CONTROLLER.main()

    def test_target_batch_accumulates_into_one_optimizer_update(self) -> None:
        """136 questions span eight DP ranks; 68 micro-batches form one local mini-batch."""
        args = arguments(Path("/shared/run"))
        args.hosts = ["192.0.2.1", "192.0.2.2", "192.0.2.3", "192.0.2.4"]
        args.global_prompt_batch_size = 136
        args.full_data = True
        args.plot = True
        batch = CONTROLLER.resolve_batch(args, 4)
        self.assertEqual(batch["global_response_batch_size"], 544)
        self.assertEqual(batch["prompts_per_dp_rank"], 17)
        self.assertEqual(batch["response_mini_batch_size"], 68)
        self.assertEqual(batch["micro_batch_size"], 1)
        self.assertEqual(batch["optimizer_updates_per_step"], 1)
        with patch.object(os, "getuid", return_value=0, create=True), \
             patch.object(os, "getgid", return_value=0, create=True):
            command = CONTROLLER.docker_command(args, 0, "172.27.3.101", "image", "owned", Path("/shared/run"))
        env = dict(value.split("=", 1) for index, value in enumerate(command) if index and command[index - 1] == "-e")
        expected = {"GLOBAL_PROMPT_BATCH_SIZE": "136", "PROMPTS_PER_RANK": "17", "CANDIDATES": "4",
                    "MICRO_BATCH_SIZE": "1", "RESPONSE_MINI_BATCH_SIZE": "68", "FULL_DATASET": "true",
                    "PLOT_TRAINING": "true", "MODEL_LAYERS": "0"}
        self.assertEqual({key: env[key] for key in expected}, expected)

    def test_invalid_batch_is_rejected_before_preflight(self) -> None:
        """Uneven DP work and impossible accumulation cannot allocate remote devices."""
        base = [str(SCRIPT), "--hosts", "192.0.2.1,192.0.2.2,192.0.2.3,192.0.2.4",
                "--master-addr", "172.27.3.101", "--global-prompt-batch-size", "136"]
        for extra in (["--micro-batch-size", "8"], ["--response-mini-batch-size", "3"],
                      ["--global-prompt-batch-size", "135"], ["--micro-batch-size", "0"]):
            with self.subTest(extra=extra), patch.object(sys, "argv", base + extra), \
                 patch.object(CONTROLLER, "preflight") as preflight, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    CONTROLLER.main()
                self.assertEqual(error.exception.code, 2)
                preflight.assert_not_called()

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

    def test_timing_summary_excludes_first_step_and_preserves_missing_data(self) -> None:
        """Startup outliers and incomplete stage counters must not create a speedup claim."""
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            metrics = directory / "node0/run/steps.jsonl"
            metrics.parent.mkdir(parents=True)
            rows = [{"perf/time_per_step": 900, "timing_s/gen": 800},
                    {"perf/time_per_step": 60, "timing_s/gen": 10, "timing_s/weight_sync": 30},
                    {"perf/time_per_step": 70, "timing_s/gen": 20}]
            metrics.write_text("\n".join(json.dumps(row) for row in rows))
            result = CONTROLLER.summarize_timings(directory)
        self.assertEqual(result["warm_step_samples"], 2)
        self.assertEqual(result["mean_seconds"], {"perf/time_per_step": 65, "timing_s/gen": 15})
        self.assertIsNone(result["reported_peak_allocated_gb"])
        self.assertIsNone(result["rollout_summary"]["mean_response_tokens"])

    def test_length_and_memory_summary_preserves_measurement_scope(self) -> None:
        """Response means are weighted by response count; startup allocation peaks remain visible."""
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            metrics = directory / "node0/run/steps.jsonl"
            metrics.parent.mkdir(parents=True)
            rows = [{"rollout/generated_tokens": 900, "system/max_memory_allocated_gb": 42},
                    {"rollout/generated_tokens": 100, "rollout/sequence_count": 10,
                     "rollout/response_length_min": 2, "rollout/response_length_max": 40,
                     "rollout/truncated_ratio": 0.1, "system/max_memory_allocated_gb": 40},
                    {"rollout/generated_tokens": 300, "rollout/sequence_count": 20,
                     "rollout/response_length_min": 3, "rollout/response_length_max": 50,
                     "rollout/truncated_ratio": 0.2, "system/max_memory_allocated_gb": 41}]
            metrics.write_text("\n".join(json.dumps(row) for row in rows))
            result = CONTROLLER.summarize_timings(directory)
        self.assertEqual(result["reported_peak_allocated_gb"], 42)
        self.assertIn("not whole-device HBM", result["memory_scope"])
        lengths = result["rollout_summary"]
        self.assertEqual(lengths["warm_response_count"], 30)
        self.assertAlmostEqual(lengths["mean_response_tokens"], 400 / 30)
        self.assertAlmostEqual(lengths["truncated_ratio"], 5 / 30)
        self.assertEqual((lengths["min_response_tokens"], lengths["max_response_tokens"]), (2, 50))

    def test_summary_mode_reads_results_without_ssh_or_docker(self) -> None:
        """Previously saved A3 runs remain readable without any runtime dependencies."""
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "summary.json").write_text(json.dumps({"passed": False, "hosts": ["192.0.2.1"]}))
            output = io.StringIO()
            with patch.object(sys, "argv", [str(SCRIPT), "--summarize", str(directory)]), \
                 patch.object(CONTROLLER, "execute", side_effect=AssertionError("unexpected remote call")), \
                 contextlib.redirect_stdout(output):
                CONTROLLER.main()
            result = json.loads(output.getvalue())
        self.assertFalse(result["passed"])
        self.assertEqual(result["warm_step_samples"], 0)
        self.assertEqual(result["mean_seconds"], {})

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
