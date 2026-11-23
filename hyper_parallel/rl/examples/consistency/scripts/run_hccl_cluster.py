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
"""Run from the first host; launch temporary Docker HCCL probes via SSH.

Requires only Python's standard library on the controller. Worker source is sent
through the container environment, so other hosts need no checkout or shared code.
"""

import argparse
import base64
import ipaddress
import json
import math
from pathlib import Path
import re
import shlex
import signal
import subprocess
import tempfile
import time
from typing import Optional
import uuid


def execute(host: str, local: bool, command: list[str]) -> subprocess.CompletedProcess:
    """Execute locally for rank zero, otherwise use noninteractive SSH."""
    if not local:
        command = ["ssh", "-oBatchMode=yes", "-oConnectTimeout=8", host, shlex.join(command)]
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=30, check=True)
    except subprocess.CalledProcessError as error:
        print(f"{host}: {error.stderr}", flush=True)
        raise


def docker_command(args: argparse.Namespace, node_rank: int, address: str,
                   image_id: str, name: str, worker: str) -> list[str]:
    """Create an isolated container command using an immutable local image ID."""
    entry = "import base64,os;exec(base64.b64decode(os.environ['HP_HCCL_WORKER']))"
    launch = ["python", "-m", "torch.distributed.run", f"--nnodes={len(args.hosts)}",
              f"--nproc_per_node={args.nproc_per_node}", f"--node_rank={node_rank}",
              f"--master_addr={args.master_addr}", f"--master_port={args.master_port}",
              f"--local-addr={address}", "--max_restarts=0", "--no-python", "python", "-c", entry]
    command = ["docker", "run", "-d", "--name", name, "--label", "hyper-rl.hccl-probe=true",
               "--privileged", "--network=host", "--shm-size=1g",
               "-v", "/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro"]
    for key, value in {"HCCL_SOCKET_IFNAME": args.interface, "GLOO_SOCKET_IFNAME": args.interface,
                       "HCCL_HOST_SOCKET_PORT_RANGE": "auto", "HCCL_NPU_SOCKET_PORT_RANGE": "auto",
                       "ASCEND_RT_VISIBLE_DEVICES": ",".join(map(str, range(args.nproc_per_node))),
                       "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1", "HP_HCCL_WORKER": worker}.items():
        command.extend(["-e", f"{key}={value}"])
    return command + [image_id, "bash", "-c",
                      "set -e; source /usr/local/Ascend/cann/set_env.sh; exec " + shlex.join(launch)]


def collect_results(logs: list[str], world: int, nproc: int) -> dict:
    """Require one successful record for every expected global and local rank."""
    records = []
    for log in logs:
        for line in log.splitlines():
            if "HCCL_PROBE_RESULT " in line:
                records.append(json.loads(line.split("HCCL_PROBE_RESULT ", 1)[1]))
    valid = (sorted(row.get("rank", -1) for row in records) == list(range(world))
             and all(row.get("passed") is True and row.get("world_size") == world
                     and row.get("local_rank") == row["rank"] % nproc for row in records))
    measurements = []
    if valid:
        for index, size in enumerate((4, 4194304, 67108864)):
            values = []
            for row in records:
                entries = row.get("measurements", [])
                if len(entries) != 3 or entries[index].get("bytes") != size:
                    valid = False
                    break
                value = entries[index].get("mean_seconds")
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                    valid = False
                    break
                values.append(value)
            if not valid:
                break
            measurements.append({"bytes": size, "min_mean_seconds": min(values),
                                 "max_mean_seconds": max(values)})
    return {"passed": valid, "expected_ranks": world, "reported_ranks": len(records),
            "measurements": measurements if valid else [],
            "scope": "AllReduce correctness and per-rank duration; not RL acceptance or bus bandwidth"}


def require_idle(info: str, host: str) -> None:
    """Reject occupied or unrecognized process tables, including pipe-separated A3 rows."""
    header = re.search(r"(?im)^.*\bprocess\s+id\b.*$", info)
    if header is None:
        raise RuntimeError(f"NPU process table unrecognized on {host}; no probe launched\n{info}")
    section = info[header.end():]
    process = re.search(r"(?m)^\|\s*\d+(?:\s*\|\s*|\s+)\d+(?:\s*\|\s*|\s+)\d+(?:\s*\||\s+).*$", section)
    if process:
        raise RuntimeError(f"NPUs busy on {host}; no probe launched: {process.group(0)}")
    if (not re.search(r"no\s+running\s+process(?:es)?\s+found", section, re.IGNORECASE)
            or re.search(r"(?m)^\|\s*\d", section)):
        raise RuntimeError(f"NPU idle state unconfirmed on {host}; no probe launched\n{info}")


def check_host(args: argparse.Namespace, host: str, local: bool, expected_image: Optional[str]) -> tuple[str, str]:
    """Check one candidate without creating containers or taking a device context."""
    current = execute(host, local, ["docker", "image", "inspect", args.image, "--format", "{{.Id}}"]).stdout.strip()
    if expected_image is not None and current != expected_image:
        raise RuntimeError(f"Image ID mismatch on {host}")
    require_idle(execute(host, local, ["npu-smi", "info"]).stdout, host)
    network = execute(host, local, ["ip", "-o", "-4", "addr", "show", "dev", args.interface]).stdout
    found = re.findall(r"\binet (\d+\.\d+\.\d+\.\d+)/", network)
    if len(found) != 1:
        raise RuntimeError(f"Expected one IPv4 address on {host}/{args.interface}")
    return current, found[0]


def preflight(args: argparse.Namespace) -> tuple[str, list[str]]:
    """Keep the first host mandatory; optionally select eligible peers in candidate order."""
    local_ips = execute(args.hosts[0], True, ["hostname", "-I"]).stdout.split()
    if args.hosts[0] not in local_ips:
        raise ValueError("Run this controller on the first host, outside Docker")
    image_id = None
    addresses = []
    selected = []
    count = getattr(args, "select_nodes", None)
    for index, host in enumerate(args.hosts):
        print(f"PREFLIGHT {host}", flush=True)
        try:
            current, address = check_host(args, host, index == 0, image_id)
        except (RuntimeError, subprocess.SubprocessError, OSError) as error:
            if index == 0 or count is None:
                raise
            print(f"SKIP {host}: {error}", flush=True)
            continue
        image_id = image_id or current
        selected.append(host)
        addresses.append(address)
        if count is not None and len(selected) == count:
            break
    if count is not None and len(selected) != count:
        raise RuntimeError(f"Only {len(selected)}/{count} eligible hosts, including required rank0; no probe launched")
    if addresses[0] != args.master_addr:
        raise ValueError("Master address must match the first host's selected interface")
    args.hosts = selected
    print("SELECTED_HOSTS " + ",".join(selected), flush=True)
    return image_id, addresses


def main() -> None:
    """Preflight all hosts, run the bounded probe, collect logs and remove only owned containers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hosts", required=True, type=lambda text: text.split(","))
    parser.add_argument("--select-nodes", type=int, help="Select this many eligible hosts, always including the first")
    parser.add_argument("--select-only", action="store_true", help="Check/select hosts without launching containers")
    parser.add_argument("--master-addr", required=True)
    parser.add_argument("--interface", required=True)
    parser.add_argument("--image", default="hyper-rl:consistency-v0.30.0-cann9.2.1-a3-runtime-arm64")
    parser.add_argument("--nproc-per-node", type=int, choices=range(1, 17), default=16)
    parser.add_argument("--master-port", type=int, default=29651)
    parser.add_argument("--result-base", type=Path, default=Path.cwd() / "hccl-results")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for address in [*args.hosts, args.master_addr]:
        ipaddress.IPv4Address(address)
    if len(args.hosts) != len(set(args.hosts)) or not 1024 <= args.master_port <= 65535:
        parser.error("Hosts must be unique and master port must be in [1024,65535]")
    if args.select_nodes is not None and not 1 <= args.select_nodes <= len(args.hosts):
        parser.error("--select-nodes must be between 1 and the number of candidate hosts")
    if args.dry_run:
        print(json.dumps({"candidate_hosts": args.hosts, "select_nodes": args.select_nodes,
                          "world_size": (args.select_nodes or len(args.hosts)) * args.nproc_per_node,
                          "interface": args.interface, "image": args.image}, indent=2))
        return
    image_id, addresses = preflight(args)
    if args.select_only:
        return
    for index, host in enumerate(args.hosts):
        require_idle(execute(host, index == 0, ["npu-smi", "info"]).stdout, host)
    args.result_base.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="probe-", dir=args.result_base.resolve()))
    worker = base64.b64encode(Path(__file__).with_name("hccl_probe_worker.py").read_bytes()).decode()
    token = uuid.uuid4().hex[:12]
    launched = []
    logs = []
    result = {"passed": False}
    try:
        for index, host in enumerate(args.hosts):
            name = f"hp-hccl-{token}-{index}"
            launched.append((index, host, name))
            print(f"LAUNCH {host} {name}", flush=True)
            execute(host, index == 0, docker_command(args, index, addresses[index], image_id, name, worker))
        deadline = time.monotonic() + 900
        while True:
            states = [json.loads(execute(host, index == 0, ["docker", "inspect", name,
                                                          "--format", "{{json .State}}"]).stdout)
                      for index, host, name in launched]
            if any(not state["Running"] and state["ExitCode"] != 0 for state in states):
                break
            if all(not state["Running"] for state in states):
                break
            if time.monotonic() > deadline:
                raise TimeoutError("Probe exceeded 900 seconds")
            time.sleep(2)
        for index, host, name in launched:
            output = execute(host, index == 0, ["docker", "logs", name])
            log = output.stdout + output.stderr
            logs.append(log)
            (directory / f"node{index}.log").write_text(log, encoding="utf-8")
        result = collect_results(logs, len(args.hosts) * args.nproc_per_node, args.nproc_per_node)
        result["passed"] = result["passed"] and all(not state["Running"] and state["ExitCode"] == 0
                                                    for state in states)
    finally:
        for index, host, name in launched:
            for command in (["docker", "logs", name], ["docker", "rm", "-f", name]):
                try:
                    output = execute(host, index == 0, command)
                    if command[1] == "logs":
                        (directory / f"node{index}.log").write_text(output.stdout + output.stderr, encoding="utf-8")
                except (subprocess.SubprocessError, OSError) as error:
                    print(f"{command[1]} incomplete for {host}/{name}: {error}", flush=True)
        result.update(hosts=args.hosts, nproc_per_node=args.nproc_per_node, image_id=image_id)
        (directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2), "\nLogs:", directory, flush=True)
    raise SystemExit(0 if result["passed"] else 1)


def interrupt(_signum: int, _frame: object) -> None:
    """Run owned-container cleanup when the controller receives SIGTERM."""
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupt)
    main()
