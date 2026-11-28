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
"""Select four idle Ascend hosts and run paired recipes using temporary Docker containers."""

import argparse
import hashlib
import ipaddress
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import tarfile
import time
import uuid
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
SCRIPTS = "hyper_parallel/rl/examples/gsm8k/scripts"
HOSTS = "80.5.21.101,80.5.21.102,80.5.21.103,80.5.21.104,80.5.21.106,80.5.21.107,80.5.21.108"
IMAGE = "hyper-rl:consistency-v0.30.0-cann9.2.1-a3-runtime-arm64"
DEPENDENCIES = """
import json, sys
from importlib.metadata import version
import matplotlib, pandas, pyarrow, yaml
versions = {name: version(name) for name in ('vllm', 'vllm-ascend', 'transformers')}
for name in ('vllm', 'vllm-ascend'):
    if versions[name].split('.')[:2] != ['0', '30']:
        raise SystemExit('Expected vLLM/vLLM-Ascend 0.30: ' + str(versions))
if sys.argv[1] == 'unified':
    versions['flash-attn-npu'] = version('flash-attn-npu')
    if 'hpbi1' not in versions['flash-attn-npu']:
        raise SystemExit('Unified requires the patched hpbi1 V4 wheel: ' + str(versions))
print(json.dumps(versions))
"""
WORKER = f"""
set -eo pipefail
source /usr/local/Ascend/cann/set_env.sh
cp -a /source/. /workspace/
cd /workspace
export PYTHONDONTWRITEBYTECODE=1
source {SCRIPTS}/configure_qwen3_shared_paths.sh
python -m pip install --no-index --no-build-isolation --no-deps -e . -e hyper_parallel/rl
exec bash {SCRIPTS}/run_qwen3_moe_64npu.sh "$MODE" "$STEPS"
"""


def shared_path(value: str, root: Path) -> Path:
    """Rebase the former shared root without matching unrelated user names."""
    old = "/mnt/share/zjy"
    return root / value[len(old) + 1:] if value.startswith(old + "/") else Path(value)


def require_idle(info: str, host: str) -> None:
    """Use the former cluster launcher's strict process-table idle check."""
    header = re.search(r"(?im)^.*\bprocess\s+id\b.*$", info)
    if header is None:
        raise RuntimeError(f"NPU process table unrecognized on {host}")
    section = info[header.end():]
    if (not re.search(r"no\s+running\s+process(?:es)?\s+found", section, re.IGNORECASE)
            or re.search(r"(?m)^\|\s*\d", section)):
        raise RuntimeError(f"NPUs busy or idle state unconfirmed on {host}")


def execute(host: str, command: list[str], *, timeout: int = 60, check: bool = True):
    """Use SSH consistently so the controller need not be one of the selected hosts."""
    user = os.environ.get("GSM8K_SSH_USER")
    target = f"{user}@{host}" if user else host
    invocation = command if host in local_addresses() else [
        "ssh", "-oBatchMode=yes", "-oConnectTimeout=8", target, shlex.join(command)]
    result = subprocess.run(invocation,
                            capture_output=True, text=True, timeout=timeout, check=False)
    if check and result.returncode:
        raise RuntimeError(f"{host}: {shlex.join(command[:3])}: {result.stderr[-2000:]}")
    return result


@lru_cache(maxsize=1)
def local_addresses() -> frozenset[str]:
    """Avoid requiring SSH-to-self when launched on one of the candidate nodes."""
    result = subprocess.run(["hostname", "-I"], capture_output=True, text=True, check=False, timeout=10)
    return frozenset(result.stdout.split()) if result.returncode == 0 else frozenset()


def find_archive(images: Path, image: str, explicit: Path | None) -> Path:
    """Read Docker archive manifests without extracting large layers onto the controller."""
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(f"Image archive is missing: {explicit}")
        return explicit
    for path in sorted(images.rglob("*")) if images.is_dir() else ():
        if not path.is_file() or not path.name.endswith((".tar", ".tar.gz", ".tgz")):
            continue
        try:
            with tarfile.open(path, "r|*") as archive:
                for member in archive:
                    if member.name.removeprefix("./") == "manifest.json":
                        stream = archive.extractfile(member)
                        if stream is not None and any(image in (row.get("RepoTags") or [])
                                                      for row in json.load(stream)):
                            return path
                        break
        except (tarfile.TarError, OSError, ValueError):
            continue
    raise FileNotFoundError(f"No archive for {image} in {images}; set GSM8K_IMAGE_ARCHIVE")


def ensure_image(host: str, args, expected: str | None) -> str:
    """Load a missing image from shared storage and require identical image content."""
    command = ["docker", "image", "inspect", args.image, "--format", "{{.Id}}"]
    result = execute(host, command, check=False)
    if result.returncode:
        archive = find_archive(args.images, args.image, args.image_archive)
        print(f"LOAD_IMAGE {host} {archive}", flush=True)
        execute(host, ["docker", "load", "-i", str(archive)], timeout=1800)
        result = execute(host, command)
    image_id = result.stdout.strip()
    if not image_id or (expected is not None and expected != image_id):
        raise RuntimeError(f"Image ID mismatch on {host}: {image_id}; expected {expected}")
    probe = f"hp-rl-check-{uuid.uuid4().hex[:12]}"
    try:
        execute(host, ["docker", "run", "--rm", "--name", probe, "--network=none", "--entrypoint=python",
                       image_id, "-c", DEPENDENCIES, args.mode], timeout=180)
    finally:
        execute(host, ["docker", "rm", "-f", probe], check=False)
    return image_id


def select_hosts(args) -> tuple[list[str], list[str], str]:
    """Choose the first four eligible candidates, including an automatically chosen master."""
    selected, addresses, image_id = [], [], None
    for host in args.hosts:
        print(f"PREFLIGHT {host}", flush=True)
        try:
            require_idle(execute(host, ["npu-smi", "info"]).stdout, host)
            address_text = execute(host, ["ip", "-o", "-4", "addr", "show", "dev", args.interface]).stdout
            found = re.findall(r"\binet (\d+\.\d+\.\d+\.\d+)/", address_text)
            if len(found) != 1:
                raise RuntimeError(f"Expected one IPv4 address on {args.interface}")
            execute(host, ["test", "-d", str(args.share_root)])
            current = ensure_image(host, args, image_id)
        except (RuntimeError, OSError, subprocess.SubprocessError) as error:
            print(f"SKIP {host}: {error}", flush=True)
            continue
        selected.append(host)
        addresses.append(found[0])
        image_id = current
        if len(selected) == 4:
            print("SELECTED_HOSTS " + ",".join(selected), flush=True)
            return selected, addresses, image_id
    raise RuntimeError(f"Only {len(selected)}/4 eligible idle hosts; training was not launched")


def stage_source(directory: Path) -> str:
    """Freeze tracked worktree files on shared storage and retain their hashes."""
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    files = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    hashes = {}
    for name in filter(None, files):
        source = ROOT / name
        if not source.is_file():
            raise FileNotFoundError(f"Tracked source is missing: {source}")
        target = directory / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    (directory / "source-manifest.json").write_text(json.dumps(
        {"revision": revision, "sha256": hashes}, indent=2), encoding="utf-8")
    return revision


def docker_command(args, directory: Path, rank: int, host: str, master: str, image_id: str,
                   name: str, revision: str) -> list[str]:
    """Bind shared storage and delegate the worker to the current 64-NPU recipe."""
    command = ["docker", "run", "-d", "--name", name, "--label", "hyper-rl.cluster=true",
               "--privileged", "--network=host", "--shm-size=64g", "-w", "/workspace"]
    for source, target, access in ((str(args.share_root), str(args.share_root), "rw"),
                                   (str(directory / "source"), "/source", "ro"),
                                   ("/usr/local/Ascend/driver", "/usr/local/Ascend/driver", "ro"),
                                   ("/usr/local/dcmi", "/usr/local/dcmi", "ro"),
                                   ("/usr/local/bin/npu-smi", "/usr/local/bin/npu-smi", "ro"),
                                   ("/etc/ascend_install.info", "/etc/ascend_install.info", "ro")):
        command.extend(["-v", f"{source}:{target}:{access}"])
    environment = {"NODE_RANK": str(rank), "MASTER_ADDR": master, "MASTER_PORT": str(args.master_port),
                   "GLOO_SOCKET_IFNAME": args.interface, "HCCL_SOCKET_IFNAME": args.interface,
                   "GSM8K_SHARE_ROOT": str(args.share_root), "GSM8K_MODEL": str(args.model),
                   "GSM8K_DATA": str(args.data), "GSM8K_RESULTS": str(args.results),
                   "GSM8K_CACHE_ROOT": str(args.cache),
                   "GSM8K_RUN_ID": directory.name, "GSM8K_CACHE_NODE": host,
                   "GSM8K_SOURCE_REVISION": revision, "MODE": args.mode, "STEPS": str(args.steps),
                   "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "PYTHONUNBUFFERED": "1"}
    if args.compare_with is not None:
        environment["GSM8K_COMPARE_WITH"] = str(args.compare_with)
    for key, value in environment.items():
        command.extend(["-e", f"{key}={value}"])
    return command + [image_id, "bash", "-c", WORKER]


def prepare_data(args, host: str, image_id: str, directory: Path, cpu_name: str) -> None:
    """Prepare missing samples on CPU and validate both counts without using any NPU."""
    if not args.data.exists() or (args.data.is_dir() and not any(args.data.iterdir())):
        print(f"PREPARE_BASELINE_DATA {args.source_data} -> {args.data}", flush=True)
        command = ["docker", "run", "--rm", "--name", cpu_name, "--network=none",
                   "-v", f"{args.share_root}:{args.share_root}:rw",
                   "-v", f"{directory / 'source'}:/source:ro", "--entrypoint=python", image_id,
                   f"/source/{SCRIPTS}/prepare_baseline_data.py", "--source", str(args.source_data),
                   "--destination", str(args.data)]
        execute(host, command, timeout=1800)
    validate = ("import pandas as pd,sys; from pathlib import Path; p=Path(sys.argv[1]); "
                "counts={s:len(pd.read_parquet(p/(s+'.parquet'))) for s in ('train','test')}; "
                "print(counts); "
                "sys.exit(0 if counts=={'train':6400,'test':500} else 'Expected prepared 6400/500 splits')")
    execute(host, ["docker", "run", "--rm", "--name", cpu_name, "--network=none",
                   "-v", f"{args.share_root}:{args.share_root}:ro", "--entrypoint=python", image_id,
                   "-c", validate, str(args.data)], timeout=180)
    print("BASELINE_DATA_READY 6400/500", flush=True)


def run(args) -> None:
    """Retain logs and clean up only this invocation's uniquely named containers."""
    hosts, addresses, image_id = select_hosts(args)
    if args.select_only:
        return
    if args.mode == "native" and args.compare_with is None:
        args.compare_with = find_reference(args.results, args.steps)
        print(f"COMPARE_WITH {args.compare_with or 'none; native figures only'}", flush=True)
    args.results.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:12]
    directory = args.results / f"qwen3-cluster-{args.mode}-{token}"
    directory.mkdir()
    print(f"RESULT_DIRECTORY {directory}", flush=True)
    launched = []
    states = []
    passed = False
    cpu_name = f"hp-rl-data-{token}"
    started = time.monotonic()
    try:
        revision = stage_source(directory)
        sentinel = directory / "shared-check.txt"
        sentinel.write_text(token, encoding="utf-8")
        for host in hosts:
            if execute(host, ["cat", str(sentinel)]).stdout.strip() != token:
                raise RuntimeError(f"Shared filesystem mismatch on {host}")
            execute(host, ["test", "-r", str(args.model / "config.json")])
        prepare_data(args, hosts[0], image_id, directory, cpu_name)
        for host in hosts:
            require_idle(execute(host, ["npu-smi", "info"]).stdout, host)
        plan = {"hosts": hosts, "addresses": addresses, "master_addr": addresses[0], "image_id": image_id,
                "mode": args.mode, "steps": args.steps, "source_revision": revision, "world_size": 64}
        (directory / "cluster-plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
        for rank, host in enumerate(hosts):
            name = f"hp-rl-{token}-{rank}"
            launched.append((rank, host, name))
            print(f"LAUNCH {host} {name}", flush=True)
            execute(host, docker_command(args, directory, rank, host, addresses[0], image_id, name, revision))
        while True:
            states = [json.loads(execute(host, ["docker", "inspect", name,
                                                "--format", "{{json .State}}"]).stdout)
                      for _, host, name in launched]
            if any(not state["Running"] and state["ExitCode"] != 0 for state in states):
                raise RuntimeError("A training container failed; inspect node*.log and node*/run/train.log")
            if all(not state["Running"] for state in states):
                break
            if time.monotonic() - started > args.timeout:
                raise TimeoutError(f"Cluster exceeded {args.timeout} seconds")
            time.sleep(5)
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        passed = summary.get("passed") is True
        if not passed:
            raise RuntimeError("Training completion or sampled-token diagnostics were not verified")
    finally:
        for rank, host, name in launched:
            try:
                log = execute(host, ["docker", "logs", name], check=False, timeout=120)
                (directory / f"node{rank}.log").write_text(log.stdout + log.stderr, encoding="utf-8")
            except (RuntimeError, OSError, subprocess.SubprocessError) as error:
                print(f"LOG_COLLECTION_FAILED {host}: {error}", flush=True)
            try:
                execute(host, ["docker", "rm", "-f", name], check=False)
            except (RuntimeError, OSError, subprocess.SubprocessError) as error:
                print(f"CLEANUP_FAILED {host}/{name}: {error}", flush=True)
        try:
            execute(hosts[0], ["docker", "rm", "-f", cpu_name], check=False)
        except (RuntimeError, OSError, subprocess.SubprocessError) as error:
            print(f"CLEANUP_FAILED {hosts[0]}/{cpu_name}: {error}", flush=True)
        result = {"passed": passed, "directory": str(directory), "container_states": states,
                  "controller_elapsed_seconds": time.monotonic() - started}
        (directory / "cluster-summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print("CLUSTER_RESULT " + json.dumps(result), flush=True)


def valid_reference(directory: Path, steps: int) -> bool:
    """Require a complete consistency run of the same length before scheduling GPUs."""
    try:
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        rows = [json.loads(line) for line in (directory / "node0/run/steps.jsonl").read_text(
            encoding="utf-8").splitlines() if line.strip()]
        return (summary.get("passed") is True and summary.get("native_baseline") is False
                and [row["step"] for row in rows] == list(range(1, steps + 1)))
    except (OSError, ValueError, KeyError):
        return False


def find_reference(results: Path, steps: int) -> Path | None:
    """Use the most recent successful controller-run consistency experiment of equal length."""
    candidates = sorted(results.glob("qwen3-cluster-unified-*"), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in candidates:
        try:
            cluster = json.loads((path / "cluster-summary.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if cluster.get("passed") is True and valid_reference(path, steps):
            return path
    return None


def parse_args(argv=None):
    """Provide machine defaults without importing Torch, vLLM or YAML on the controller."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("native", "consistency", "unified"), default="consistency")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--hosts", default=os.environ.get("GSM8K_HOSTS", HOSTS))
    parser.add_argument("--interface", default=os.environ.get("GSM8K_INTERFACE", "data0.172"))
    parser.add_argument("--master-port", type=int, default=29751)
    parser.add_argument("--image", default=os.environ.get("GSM8K_IMAGE", IMAGE))
    parser.add_argument("--share-root", type=Path, default=Path(os.environ.get("GSM8K_SHARE_ROOT", "/mnt/share/zjy1")))
    parser.add_argument("--image-archive", default=os.environ.get("GSM8K_IMAGE_ARCHIVE"))
    parser.add_argument("--compare-with", default=os.environ.get("GSM8K_COMPARE_WITH"))
    parser.add_argument("--timeout", type=int, default=86400)
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.mode = "unified" if args.mode == "consistency" else args.mode
    args.hosts = args.hosts.split(",")
    for host in args.hosts:
        ipaddress.IPv4Address(host)
    if args.steps < 1 or len(set(args.hosts)) != len(args.hosts) or len(args.hosts) < 4:
        parser.error("Use a positive step count and at least four unique candidate hosts")
    if not 1024 <= args.master_port <= 65535 or args.timeout < 1:
        parser.error("Invalid port or timeout")
    args.share_root = args.share_root.resolve()
    defaults = {"model": ("GSM8K_MODEL", "models/Qwen3-30B-A3B"),
                "data": ("GSM8K_DATA", "datasets/gsm8k-6400-500"),
                "source_data": ("GSM8K_SOURCE_DATA", "datasets/gsm8k/prepared"),
                "results": ("GSM8K_RESULTS", "rl-cluster-runs"), "images": ("GSM8K_IMAGES", "images"),
                "cache": ("GSM8K_CACHE_ROOT", "cache")}
    for key, (variable, relative) in defaults.items():
        value = os.environ.get(variable, str(args.share_root / relative))
        setattr(args, key, shared_path(value, args.share_root).resolve())
    for key in ("compare_with", "image_archive"):
        value = getattr(args, key)
        if value:
            setattr(args, key, shared_path(value, args.share_root).resolve())
    if args.compare_with is not None and (args.mode != "native" or not valid_reference(args.compare_with, args.steps)):
        parser.error("--compare-with requires native mode and a completed consistency run of equal length")
    # Only this root is mounted on workers, so reject input/output paths outside it early.
    for key in (*defaults, "compare_with", "image_archive"):
        value = getattr(args, key)
        if value is not None and not value.is_relative_to(args.share_root):
            parser.error(f"{key} must be inside the mounted shared root: {value}")
    return args


def interrupt(_signum, _frame):
    """Unwind through owned-container cleanup on termination."""
    raise KeyboardInterrupt


def main() -> None:
    """Launch from one host terminal; remote training remains explicitly requested."""
    args = parse_args()
    if args.dry_run:
        print(json.dumps(vars(args), default=str, indent=2))
        return
    if not args.share_root.is_dir():
        raise FileNotFoundError(f"Shared storage is unavailable: {args.share_root}")
    signal.signal(signal.SIGTERM, interrupt)
    run(args)


if __name__ == "__main__":
    main()
