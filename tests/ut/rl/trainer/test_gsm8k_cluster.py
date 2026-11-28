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
"""Mock SSH and Docker to check node selection, image loading and owned cleanup."""

import importlib.util
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "hyper_parallel/rl/examples/gsm8k/scripts/run_qwen3_moe_cluster.py"
SPEC = importlib.util.spec_from_file_location("gsm8k_cluster", SCRIPT)
cluster = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cluster)
IDLE = "| Process id | Process name |\n| No running processes found |"


class ClusterTests(unittest.TestCase):
    def test_steps_are_arbitrary_positive_integers(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = cluster.parse_args(["--share-root", temporary, "--steps", "7"])
            self.assertEqual(args.steps, 7)
            self.assertEqual(args.mode, "unified")
            self.assertEqual(len(args.hosts), 7)
            with self.assertRaises(SystemExit):
                cluster.parse_args(["--steps", "0"])

    def test_busy_and_unrecognized_device_tables_are_rejected(self):
        cluster.require_idle(IDLE, "host")
        for text in ("unrecognized", "| Process id |\n| 0 | 1 | 123 | python |", IDLE + "\n| 0 | 123 |"):
            with self.assertRaises(RuntimeError):
                cluster.require_idle(text, "host")

    def test_selection_skips_busy_nodes_and_chooses_its_own_master(self):
        args = cluster.parse_args([])

        def execute(host, command, **_kwargs):
            if command[0] == "npu-smi":
                return Mock(stdout=IDLE if host != args.hosts[0] else "| Process id |\n| 0 | 123 |")
            if command[0] == "ip":
                return Mock(stdout="inet 172.27.3." + host.split(".")[-1] + "/24")
            return Mock(stdout="")

        with patch.object(cluster, "execute", side_effect=execute), \
                patch.object(cluster, "ensure_image", return_value="sha256:same"):
            hosts, addresses, image = cluster.select_hosts(args)
        self.assertEqual(hosts, args.hosts[1:5])
        self.assertEqual(addresses[0], "172.27.3.102")
        self.assertEqual(image, "sha256:same")

    def test_archive_discovery_reads_manifest_without_extracting_layers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "image.tar"
            data = json.dumps([{"RepoTags": [cluster.IMAGE]}]).encode()
            with tarfile.open(path, "w") as archive:
                entry = tarfile.TarInfo("manifest.json")
                entry.size = len(data)
                archive.addfile(entry, io.BytesIO(data))
            self.assertEqual(cluster.find_archive(root, cluster.IMAGE, None), path)

    def test_missing_image_is_loaded_then_verified(self):
        args = cluster.parse_args([])
        calls = []

        def execute(_host, command, **_kwargs):
            calls.append(command)
            if command[:3] == ["docker", "image", "inspect"]:
                count = sum(cmd[:3] == ["docker", "image", "inspect"] for cmd in calls)
                return Mock(returncode=int(count == 1), stdout="sha256:same")
            return Mock(returncode=0, stdout="")

        with patch.object(cluster, "execute", side_effect=execute), \
                patch.object(cluster, "find_archive", return_value=Path("/shared/image.tar")):
            self.assertEqual(cluster.ensure_image("host", args, None), "sha256:same")
        self.assertIn(["docker", "load", "-i", str(Path("/shared/image.tar"))], calls)
        self.assertTrue(any(cmd[:2] == ["docker", "rm"] and cmd[-1].startswith("hp-rl-check-")
                            for cmd in calls))

    def test_failed_partial_launch_collects_logs_and_removes_only_owned_containers(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = cluster.parse_args(["--share-root", temporary, "--steps", "7"])
            hosts = args.hosts[:4]
            calls = []

            def execute(_host, command, **_kwargs):
                calls.append(command)
                if command[0] == "cat":
                    return Mock(stdout=Path(command[1]).read_text())
                if command[0] == "npu-smi":
                    return Mock(stdout=IDLE)
                if command[:3] == ["docker", "run", "-d"]:
                    if command[command.index("--name") + 1].endswith("-1"):
                        raise RuntimeError("launch failed")
                return Mock(stdout="retained log", stderr="")

            with patch.object(cluster, "select_hosts", return_value=(hosts, ["172.27.3.101"] * 4, "sha256:same")), \
                    patch.object(cluster, "stage_source", return_value="revision"), \
                    patch.object(cluster, "prepare_data"), patch.object(cluster, "execute", side_effect=execute):
                with self.assertRaisesRegex(RuntimeError, "launch failed"):
                    cluster.run(args)
            directory = next(args.results.glob("qwen3-cluster-*"))
            self.assertEqual((directory / "node0.log").read_text(), "retained log")
            self.assertFalse(json.loads((directory / "cluster-summary.json").read_text())["passed"])
            removed = [cmd[-1] for cmd in calls if cmd[:3] == ["docker", "rm", "-f"]]
            self.assertEqual(len(removed), 3)
            self.assertTrue(all(name.startswith("hp-rl-") for name in removed))
            self.assertFalse(any("prune" in cmd for cmd in calls))
