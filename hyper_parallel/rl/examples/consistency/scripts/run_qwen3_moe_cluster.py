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
"""Launch full Qwen3 MoE synchronous consistency training from rank-0's host.

The selected hosts share models, data and an immutable staged source directory.
Each host runs one temporary container with a local vLLM service and 16 Trainers.
"""

import argparse
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import tarfile
import tempfile
import time
from typing import Optional
import uuid

from run_hccl_cluster import execute, interrupt, preflight, require_idle


ROOT = Path(__file__).resolve().parents[5]
SCRIPTS = "hyper_parallel/rl/examples/consistency/scripts"
WHEEL = "flash_attn_npu-0.5.0+hpbi1-cp312-cp312-linux_aarch64.whl"
WHEEL_SHA256 = "5ef04a872b1be2c812bb0c36c17ba4776a4b5f9b9eabb24eab0a56a139a497c9"


def resolve_batch(args: argparse.Namespace, node_count: int) -> dict:
    """Keep prompt work global and gradient accumulation local to the training DP group."""
    dp_size = node_count * 2
    prompts = args.global_prompt_batch_size if args.global_prompt_batch_size is not None else dp_size
    if prompts < 1 or prompts % dp_size:
        raise ValueError(f"Global prompt batch must be positive and divisible by training DP={dp_size}")
    responses = prompts // dp_size * 4
    mini = args.response_mini_batch_size if args.response_mini_batch_size is not None else responses
    if args.micro_batch_size < 1 or mini < 1 or responses % mini or mini % args.micro_batch_size:
        raise ValueError("Responses per DP rank must divide into mini-batches and mini-batches into micro-batches")
    return {"training_dp_size": dp_size, "global_prompt_batch_size": prompts,
            "prompts_per_dp_rank": prompts // dp_size, "candidates": 4,
            "global_response_batch_size": prompts * 4, "response_mini_batch_size": mini,
            "micro_batch_size": args.micro_batch_size, "optimizer_updates_per_step": responses // mini}


def finite_values(rows: list[dict], key: str) -> Optional[list[float]]:
    """Return complete nonnegative measurements; partial evidence stays missing."""
    values = [row.get(key) for row in rows]
    if not values or not all(isinstance(value, (int, float)) and not isinstance(value, bool)
                             and math.isfinite(value) and value >= 0 for value in values):
        return None
    return values


def summarize_timings(directory: Path) -> dict:
    """Read global-rank-zero metrics without treating missing measurements as zero."""
    path = directory / "node0/run/steps.jsonl"
    rows = ([json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if path.is_file() else [])
    warm = rows[1:]
    keys = sorted({key for row in warm for key in row
                   if key.startswith("timing_s/") or key == "perf/time_per_step"})
    means = {}
    for key in keys:
        values = finite_values(warm, key)
        if values is not None:
            means[key] = statistics.mean(values)
    tokens = finite_values(warm, "rollout/generated_tokens")
    generated = sum(tokens) if tokens is not None else None
    counts = finite_values(warm, "rollout/sequence_count")
    count = sum(counts) if counts is not None else None
    minimum = finite_values(warm, "rollout/response_length_min")
    maximum = finite_values(warm, "rollout/response_length_max")
    truncation = finite_values(warm, "rollout/truncated_ratio")
    memory = finite_values(rows, "system/max_memory_allocated_gb")
    return {"recorded_steps": len(rows), "warm_step_samples": len(warm), "mean_seconds": means,
            "warm_generated_tokens": generated,
            "rollout_summary": {
                "warm_response_count": count,
                "mean_response_tokens": generated / count if count and generated is not None else None,
                "min_response_tokens": min(minimum) if minimum is not None else None,
                "max_response_tokens": max(maximum) if maximum is not None else None,
                "truncated_ratio": (sum(ratio * size for ratio, size in zip(truncation, counts)) / count
                                    if count and truncation is not None else None)},
            "reported_peak_allocated_gb": max(memory) if memory is not None else None,
            "memory_scope": "Trainer rank-0 allocator peak; excludes vLLM and other ranks; not whole-device HBM",
            "timing_scope": "Excludes first step, container setup and model startup; not full-dataset throughput"}


def stage_source(directory: Path, wheel: Path) -> tuple[Path, str]:
    """Copy tracked source and required offline wheels once into shared storage."""
    files = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True,
                           check=True).stdout.decode().split("\0")
    target = directory / "source"
    target.mkdir()
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    if digest != WHEEL_SHA256:
        raise ValueError(f"Unexpected batch-invariant wheel SHA256: {digest}")
    archive = directory / "source.tar"
    with tarfile.open(archive, "w") as bundle:
        for filename in files:
            if filename and not filename.startswith("wheels/"):
                bundle.add(ROOT / filename, arcname=filename, recursive=False)
        bundle.add(wheel, arcname=f"wheels/{WHEEL}", recursive=False)
        for dependency in sorted((ROOT / "wheels/runtime").glob("*.whl")):
            bundle.add(dependency, arcname=f"wheels/runtime/{dependency.name}", recursive=False)
    # The archive contains only this checkout's tracked files and explicit wheels.
    subprocess.run(["tar", "-xf", str(archive), "-C", str(target)], check=True)
    archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    (directory / "source.tar.sha256").write_text(archive_hash + "\n", encoding="utf-8")
    archive.unlink()
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    return target, revision


def docker_command(args: argparse.Namespace, node_rank: int, address: str,
                   image_id: str, name: str, directory: Path) -> list[str]:
    """Use node-local inference meshes alongside one global training mesh."""
    source = directory / "source"
    result = directory / f"node{node_rank}"
    batch = resolve_batch(args, len(args.hosts))
    command = ["docker", "run", "-d", "--name", name, "--label", "hyper-rl.cluster=true",
               "--privileged", "--network=host", "--shm-size=64g", "-w", "/workspace"]
    mounts = [(str(source), "/workspace", "ro"), (str(result), "/results", "rw"),
              (str(args.model_root), "/models/Qwen3-30B-A3B", "ro"),
              (str(args.data_root), "/data/gsm8k", "ro"),
              (str(source / "wheels"), "/wheels", "ro"),
              (str(source / "wheels/runtime"), "/runtime-wheels", "ro"),
              (str(source / "hyper_parallel/rl/examples/consistency/configs/qwen3_moe_4l_cp_rl.yaml"),
               "/run-config.yaml", "ro"),
              ("/usr/local/Ascend/driver", "/usr/local/Ascend/driver", "ro"),
              ("/usr/local/dcmi", "/usr/local/dcmi", "ro"),
              ("/usr/local/bin/npu-smi", "/usr/local/bin/npu-smi", "ro"),
              ("/etc/ascend_install.info", "/etc/ascend_install.info", "ro")]
    for host, container, mode in mounts:
        command.extend(["-v", f"{host}:{container}:{mode}"])
    environment = {
        "DEVICE_COUNT": "16", "MODEL_LAYERS": "0", "NNODES": str(len(args.hosts)),
        "NODE_RANK": str(node_rank), "NODE_ADDR": address,
        "MASTER_ADDR": args.master_addr, "MASTER_PORT": str(args.master_port),
        "ROLLOUT_LOCAL_WORLD_SIZE": "16", "TRAIN_TP": "4", "TRAIN_CP": "2", "TRAIN_EP": "4",
        "TRAIN_SP": "true", "ROLLOUT_TP": "2", "ROLLOUT_PCP": "1", "ROLLOUT_EP": "8",
        "ROLLOUT_SP": "false", "CPU_OFFLOAD": str(args.cpu_offload).lower(), "RUNNER": "internal",
        "STEPS": str(args.steps), "CANDIDATES": "4", "PROMPTS_PER_RANK": str(batch["prompts_per_dp_rank"]),
        "GLOBAL_PROMPT_BATCH_SIZE": str(batch["global_prompt_batch_size"]),
        "MICRO_BATCH_SIZE": str(batch["micro_batch_size"]),
        "RESPONSE_MINI_BATCH_SIZE": str(batch["response_mini_batch_size"]),
        "FULL_DATASET": str(args.full_data).lower(),
        "MAX_PROMPT": "512", "MAX_NEW_TOKENS": str(args.max_new_tokens),
        "MAX_MODEL_LEN": str(512 + args.max_new_tokens), "PREFILL_TOKENS": "512",
        "MAX_NUM_SEQS": "4", "KV_CACHE_MEMORY_BYTES": str(2 * 1024**3),
        "RUN_TIMEOUT": str(args.timeout), "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "ASCEND_RT_VISIBLE_DEVICES": ",".join(map(str, range(16))),
        "HCCL_SOCKET_IFNAME": args.interface, "GLOO_SOCKET_IFNAME": args.interface,
        "HCCL_NPU_SOCKET_PORT_RANGE": "auto", "HCCL_HOST_SOCKET_PORT_RANGE": "auto",
        "HCCL_DETERMINISTIC": "true", "HCCL_BUFFSIZE": "512",
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "auto"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "auto"),
        "HYPER_RL_CPU_RESERVE": os.environ.get("HYPER_RL_CPU_RESERVE", "auto"),
        "HYPER_RL_VLLM_CPU_THREADS": os.environ.get("HYPER_RL_VLLM_CPU_THREADS", "1"),
        "VLLM_HOST_IP": "127.0.0.1", "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "HYPER_RL_DIAGNOSTICS_DIR": "/results/run", "V4_WHEEL_SHA256": WHEEL_SHA256,
        "RESULT_UID": str(os.getuid()), "RESULT_GID": str(os.getgid()), "PLOT_TRAINING": str(args.plot).lower(),
    }
    for key, value in environment.items():
        command.extend(["-e", f"{key}={value}"])
    return command + [image_id, "bash", f"{SCRIPTS}/run_qwen3_moe_worker.sh", "--track-updates"]


def run(args: argparse.Namespace) -> Path:
    """Preflight, stage, launch, collect evidence, and clean up only owned containers."""
    image_id, addresses = preflight(args)
    batch = resolve_batch(args, len(args.hosts))
    args.shared_root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="qwen3-cluster-", dir=args.shared_root.resolve()))
    source, revision = stage_source(directory, args.wheel)
    (source / "wheels/runtime").mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:12]
    sentinel = directory / "shared-check.txt"
    sentinel.write_text(token, encoding="utf-8")
    for index, host in enumerate(args.hosts):
        if execute(host, index == 0, ["cat", str(sentinel)]).stdout != token:
            raise RuntimeError(f"Shared storage mismatch on {host}")
        for path in (args.model_root / "config.json", args.data_root / "train.parquet",
                     args.data_root / "test.parquet", source / f"{SCRIPTS}/run_qwen3_moe_worker.sh"):
            execute(host, index == 0, ["test", "-s", str(path)])
        require_idle(execute(host, index == 0, ["npu-smi", "info"]).stdout, host)
        (directory / f"node{index}/run").mkdir(parents=True)
    plan = {"hosts": args.hosts, "addresses": addresses, "image_id": image_id, "revision": revision,
            "world_size": len(args.hosts) * 16, "training": "TP4/CP2/EP4; DP=world/8; EDP=world/4",
            "rollout_per_node": "TP2/DP8/EP8", "cpu_offload": args.cpu_offload,
            "steps": args.steps, "model_layers": "full", "model_root": str(args.model_root),
            "data_root": str(args.data_root), "batch": batch, "full_data": args.full_data,
            "max_new_tokens": args.max_new_tokens, "plot": args.plot,
            "scope": "Short consistency run; not full GSM8K training/evaluation"}
    (directory / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print("BATCH_PLAN", json.dumps(batch), flush=True)
    launched = []
    result = {**plan, "passed": False}
    print("RESULT_DIRECTORY", directory, flush=True)
    try:
        for index, host in enumerate(args.hosts):
            name = f"hp-rl-{token}-{index}"
            launched.append((index, host, name))
            print("LAUNCH", host, name, flush=True)
            execute(host, index == 0, docker_command(args, index, addresses[index], image_id, name, directory))
        deadline = time.monotonic() + args.timeout + 1800
        while True:
            states = [json.loads(execute(host, index == 0, ["docker", "inspect", name,
                                                          "--format", "{{json .State}}"]).stdout)
                      for index, host, name in launched]
            if any(not state["Running"] and state["ExitCode"] != 0 for state in states):
                break
            if all(not state["Running"] for state in states):
                break
            if time.monotonic() > deadline:
                raise TimeoutError("Cluster run exceeded the controller deadline")
            time.sleep(5)
        result["container_states"] = states
        report = directory / "node0/run/consistency.json"
        if report.exists():
            evidence = json.loads(report.read_text(encoding="utf-8"))
            result["consistency"] = evidence
            result["passed"] = (evidence.get("passed") is True
                                and evidence.get("completed_steps") == args.steps
                                and len(evidence.get("checks", [])) == args.steps
                                and all(not state["Running"] and state["ExitCode"] == 0 for state in states))
    finally:
        for index, host, name in launched:
            for command in (["docker", "logs", name], ["docker", "rm", "-f", name]):
                try:
                    output = execute(host, index == 0, command)
                    if command[1] == "logs":
                        (directory / f"node{index}.log").write_text(output.stdout + output.stderr, encoding="utf-8")
                except (subprocess.SubprocessError, OSError) as error:
                    print(f"{command[1]} incomplete for {host}/{name}: {error}", flush=True)
        try:
            result.update(summarize_timings(directory))
        except (OSError, ValueError) as error:
            result["timing_error"] = str(error)
        (directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        timing_keys = ("warm_step_samples", "mean_seconds", "warm_generated_tokens", "rollout_summary",
                       "reported_peak_allocated_gb", "memory_scope", "timing_error")
        print("CLUSTER_TIMING", json.dumps({key: result[key] for key in timing_keys if key in result}), flush=True)
        print("CLUSTER_RESULT", json.dumps({"passed": result["passed"], "directory": str(directory)}), flush=True)
    if not result["passed"]:
        raise RuntimeError(f"Cluster training failed; inspect {directory}/node*.log and node*/run")
    return directory


def main() -> None:
    """Parse a portable controller command; dry-run never contacts hosts or loads models."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hosts", type=lambda value: value.split(","))
    parser.add_argument("--select-nodes", type=int, default=4)
    parser.add_argument("--master-addr")
    parser.add_argument("--summarize", type=Path, help="Read saved timing and acceptance without launching anything")
    parser.add_argument("--interface", default="data0.172")
    parser.add_argument("--master-port", type=int, default=29751)
    parser.add_argument("--image", default="hyper-rl:consistency-v0.30.0-cann9.2.1-a3-runtime-arm64")
    parser.add_argument("--model-root", type=Path, default=Path("/mnt/share/zjy/models/Qwen3-30B-A3B"))
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/share/zjy/datasets/gsm8k/prepared"))
    parser.add_argument("--shared-root", type=Path, default=Path("/mnt/share/zjy/rl-cluster-runs"))
    parser.add_argument("--wheel", type=Path, default=ROOT / "wheels" / WHEEL)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--global-prompt-batch-size", type=int, help="Global questions per step across training DP")
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--response-mini-batch-size", type=int,
                        help="Responses per training DP mini-batch; default is all local responses")
    parser.add_argument("--full-data", action="store_true",
                        help="Shuffle all training rows; steps need not cover an epoch")
    parser.add_argument("--plot", action="store_true", help="Export rank-zero reward/loss/consistency figures")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--timeout", type=int, default=14400)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.summarize is not None:
        directory = args.summarize.resolve()
        result = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        result = {key: result.get(key) for key in ("passed", "hosts", "revision", "steps", "batch", "full_data")}
        result.update(directory=str(directory), **summarize_timings(directory))
        print(json.dumps(result, indent=2), flush=True)
        return
    if not args.hosts or not args.master_addr:
        parser.error("--hosts and --master-addr are required to launch training")
    for address in [*args.hosts, args.master_addr]:
        ipaddress.IPv4Address(address)
    if (len(set(args.hosts)) != len(args.hosts) or not 2 <= args.select_nodes <= len(args.hosts)
            or not 1024 <= args.master_port <= 65535 or min(args.steps, args.max_new_tokens, args.timeout) < 1):
        parser.error("Use unique hosts, >=2 selected nodes, a valid port and positive budgets")
    try:
        batch = resolve_batch(args, args.select_nodes)
    except ValueError as error:
        parser.error(str(error))
    if args.dry_run:
        print(json.dumps({"candidate_hosts": args.hosts, "required_rank0": args.hosts[0],
                          "selected_nodes": args.select_nodes, "world_size": args.select_nodes * 16,
                          "steps": args.steps, "model_layers": "full", "cpu_offload": args.cpu_offload,
                          "training": f"TP4/CP2/DP{args.select_nodes * 2}/EP4/EDP{args.select_nodes * 4}",
                          "rollout_per_node": "TP2/DP8/EP8", "shared_root": str(args.shared_root),
                          "batch": batch, "full_data": args.full_data, "plot": args.plot}, indent=2))
        return
    run(args)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupt)
    main()
