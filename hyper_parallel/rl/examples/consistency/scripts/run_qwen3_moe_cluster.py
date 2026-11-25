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
import shutil
import signal
import statistics
import subprocess
import tarfile
import tempfile
import time
import uuid
from pathlib import Path

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
    candidates = 8 if (getattr(args, "baseline_5step", False) or getattr(args, "native_baseline", False)) else 4
    responses = prompts // dp_size * candidates
    mini = args.response_mini_batch_size if args.response_mini_batch_size is not None else responses
    if args.micro_batch_size < 1 or mini < 1 or responses % mini or mini % args.micro_batch_size:
        raise ValueError("Responses per DP rank must divide into mini-batches and mini-batches into micro-batches")
    return {"training_dp_size": dp_size, "global_prompt_batch_size": prompts,
            "prompts_per_dp_rank": prompts // dp_size, "candidates": candidates,
            "global_response_batch_size": prompts * candidates, "response_mini_batch_size": mini,
            "micro_batch_size": args.micro_batch_size, "optimizer_updates_per_step": responses // mini}


def finite_values(rows: list[dict], key: str) -> list[float] | None:
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


def stage_source(directory: Path, wheel: Path | None) -> tuple[Path, str]:
    """Copy tracked source and required offline wheels once into shared storage."""
    files = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True,
                           check=True).stdout.decode().split("\0")
    target = directory / "source"
    target.mkdir()
    if wheel is not None:
        digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
        if digest != WHEEL_SHA256:
            raise ValueError(f"Unexpected batch-invariant wheel SHA256: {digest}")
    archive = directory / "source.tar"
    with tarfile.open(archive, "w") as bundle:
        for filename in files:
            if filename and not filename.startswith("wheels/"):
                bundle.add(ROOT / filename, arcname=filename, recursive=False)
        # Explicit recipe additions must be staged even before their first commit.
        baseline_files = (
            "hyper_parallel/rl/examples/gsm8k/configs/qwen3_30b_a3b_gsm8k_unified_5step.yaml",
            "hyper_parallel/rl/examples/gsm8k/scripts/prepare_baseline_data.py",
            "hyper_parallel/rl/examples/gsm8k/scripts/run_qwen3_30b_5step.sh",
            "hyper_parallel/rl/examples/gsm8k/scripts/run_qwen3_30b_native.py",
            "hyper_parallel/rl/examples/gsm8k/configs/qwen3_30b_a3b_gsm8k_native.yaml",
        )
        for filename in baseline_files:
            if filename not in files and (ROOT / filename).is_file():
                bundle.add(ROOT / filename, arcname=filename, recursive=False)
        if wheel is not None:
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
    recipe = ("hyper_parallel/rl/examples/gsm8k/configs/qwen3_30b_a3b_gsm8k_native.yaml"
              if getattr(args, "native_baseline", False) else
              "hyper_parallel/rl/examples/gsm8k/configs/qwen3_30b_a3b_gsm8k_unified_5step.yaml"
              if getattr(args, "baseline_5step", False) else
              "hyper_parallel/rl/examples/consistency/configs/qwen3_moe_4l_cp_rl.yaml")
    command = ["docker", "run", "-d", "--name", name, "--label", "hyper-rl.cluster=true",
               "--privileged", "--network=host", "--shm-size=64g", "-w", "/workspace"]
    mounts = [(str(source), "/workspace", "ro"), (str(result), "/results", "rw"),
              (str(args.model_root), "/models/Qwen3-30B-A3B", "ro"),
              (str(args.data_root), "/data/gsm8k", "ro"),
              (str(source / "wheels"), "/wheels", "ro"),
              (str(source / "wheels/runtime"), "/runtime-wheels", "ro"),
              (str(source / recipe), "/run-config.yaml", "ro"),
              ("/usr/local/Ascend/driver", "/usr/local/Ascend/driver", "ro"),
              ("/usr/local/dcmi", "/usr/local/dcmi", "ro"),
              ("/usr/local/bin/npu-smi", "/usr/local/bin/npu-smi", "ro"),
              ("/etc/ascend_install.info", "/etc/ascend_install.info", "ro")]
    if args.formal_gsm8k:
        mounts.append((str(directory / "formal"), "/formal-run", "rw"))
    if getattr(args, "baseline_5step", False) or getattr(args, "native_baseline", False):
        mounts.append((str(directory / "baseline"), "/baseline-run", "rw"))
    for host, container, mode in mounts:
        command.extend(["-v", f"{host}:{container}:{mode}"])
    environment = {
        "DEVICE_COUNT": "16", "MODEL_LAYERS": "0", "NNODES": str(len(args.hosts)),
        "NODE_RANK": str(node_rank), "NODE_ADDR": address,
        "MASTER_ADDR": args.master_addr, "MASTER_PORT": str(args.master_port),
        "ROLLOUT_LOCAL_WORLD_SIZE": "16", "TRAIN_TP": "4", "TRAIN_CP": "2", "TRAIN_EP": "4",
        "TRAIN_SP": "true", "ROLLOUT_TP": "2", "ROLLOUT_PCP": "1",
        "ROLLOUT_EP": ("16" if getattr(args, "baseline_5step", False)
                       or getattr(args, "native_baseline", False) else "8"),
        "ROLLOUT_SP": "false", "CPU_OFFLOAD": str(args.cpu_offload).lower(), "RUNNER": "internal",
        "STEPS": str(args.steps), "CANDIDATES": str(batch["candidates"]),
        "PROMPTS_PER_RANK": str(batch["prompts_per_dp_rank"]),
        "GLOBAL_PROMPT_BATCH_SIZE": str(batch["global_prompt_batch_size"]),
        "MICRO_BATCH_SIZE": str(batch["micro_batch_size"]),
        "RESPONSE_MINI_BATCH_SIZE": str(batch["response_mini_batch_size"]),
        "FULL_DATASET": str(args.full_data).lower(),
        "FORMAL_GSM8K": str(args.formal_gsm8k).lower(),
        "MAX_PROMPT": str(args.max_prompt), "MAX_NEW_TOKENS": str(args.max_new_tokens),
        "MAX_MODEL_LEN": str(args.max_prompt + args.max_new_tokens), "PREFILL_TOKENS": str(args.max_prompt),
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
    if getattr(args, "baseline_5step", False):
        environment.update(BASELINE_5STEP="true", PREFILL_TOKENS="8192")
    if getattr(args, "native_baseline", False):
        environment.update(NATIVE_BASELINE="true", PREFILL_TOKENS="8192", CPU_OFFLOAD="false",
                           TRAIN_SP="false", TRAIN_CP="2")
    if getattr(args, "baseline_5step", False) or getattr(args, "native_baseline", False):
        environment.pop("KV_CACHE_MEMORY_BYTES", None)
        environment["MAX_NUM_SEQS"] = "64"
        environment["GPU_MEMORY_UTILIZATION"] = str(getattr(args, "gpu_memory_utilization", 0.4))
    for key, value in environment.items():
        command.extend(["-e", f"{key}={value}"])
    options = [] if getattr(args, "native_baseline", False) else ["--track-updates"]
    return command + [image_id, "bash", f"{SCRIPTS}/run_qwen3_moe_worker.sh", *options]


def prepare_formal_data(args: argparse.Namespace, directory: Path, image_id: str) -> dict:
    """Prepare the selected data protocol on CPU before launching training containers."""
    baseline = getattr(args, "baseline_5step", False) or getattr(args, "native_baseline", False)
    shared = directory / ("baseline" if baseline else "formal")
    shared.mkdir()
    name = f"hp-rl-data-{uuid.uuid4().hex[:12]}"
    command = ["docker", "run", "-d", "--name", name, "--network=none", "-w", "/workspace",
               "-e", "PYTHONPATH=/workspace/hyper_parallel/rl:/workspace", "-e", "OMP_NUM_THREADS=1",
               "-e", "MKL_NUM_THREADS=1", "-e", "PYTHONDONTWRITEBYTECODE=1"]
    for host, target, mode in ((directory / "source", "/workspace", "ro"),
                               (shared, "/formal-run", "rw"), (args.data_root, "/data/gsm8k", "ro"),
                               (args.model_root, "/models/Qwen3-30B-A3B", "ro"),
                               ("/usr/local/Ascend/driver", "/usr/local/Ascend/driver", "ro")):
        command.extend(["-v", f"{host}:{target}:{mode}"])
    command.extend([image_id, "bash", "-c",
                    "source /usr/local/Ascend/cann/set_env.sh && exec python "
                    + ("hyper_parallel/rl/examples/gsm8k/scripts/prepare_baseline_data.py "
                       if baseline else f"{SCRIPTS}/full_gsm8k.py ")
                    + "--source /data/gsm8k --destination /formal-run/data "
                    + ("" if baseline else f"--model /models/Qwen3-30B-A3B --max-prompt {args.max_prompt}")])
    print("PREPARE_BASELINE_DATA CPU-only; reproducing ordered 6400/500 samples" if baseline else
          "PREPARE_FORMAL_DATA CPU-only; validating all prompts and split isolation", flush=True)
    try:
        execute(args.hosts[0], True, command)
        deadline = time.monotonic() + 1800
        while True:
            state = json.loads(execute(args.hosts[0], True,
                                      ["docker", "inspect", name, "--format", "{{json .State}}"]).stdout)
            if not state["Running"]:
                if state["ExitCode"] != 0:
                    raise RuntimeError("Formal data preparation failed; inspect prepare-data.log")
                break
            if time.monotonic() > deadline:
                raise TimeoutError("Formal CPU data preparation exceeded 30 minutes")
            time.sleep(2)
    finally:
        for command in (["docker", "logs", name], ["docker", "rm", "-f", name]):
            try:
                output = execute(args.hosts[0], True, command)
                if command[1] == "logs":
                    (directory / "prepare-data.log").write_text(output.stdout + output.stderr, encoding="utf-8")
                    print(output.stdout + output.stderr, flush=True)
            except (subprocess.SubprocessError, OSError) as error:
                print(f"Data container {command[1]} incomplete: {error}", flush=True)
    return json.loads((shared / "data/manifest.json").read_text(encoding="utf-8"))


def run(args: argparse.Namespace) -> Path:
    """Preflight, stage, launch, collect evidence, and clean up only owned containers."""
    controller_started = time.monotonic()
    comparison = getattr(args, "compare_with", None)
    if comparison is not None:
        evidence = json.loads((comparison / "summary.json").read_text(encoding="utf-8"))
        rows = [json.loads(line) for line in (comparison / "node0/run/steps.jsonl").read_text(
            encoding="utf-8").splitlines() if line.strip()]
        if (evidence.get("passed") is not True or evidence.get("native_baseline") is True
                or [row.get("step") for row in rows] != list(range(1, 51))):
            raise ValueError("--compare-with requires an existing successful 50-step Consistency run")
    image_id, addresses = preflight(args)
    batch = resolve_batch(args, len(args.hosts))
    args.shared_root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="qwen3-cluster-", dir=args.shared_root.resolve()))
    source, revision = stage_source(directory, None if getattr(args, "native_baseline", False) else args.wheel)
    print("RESULT_DIRECTORY", directory, flush=True)
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
    if args.formal_gsm8k:
        weight_bytes = sum(path.stat().st_size for path in args.model_root.glob("*.safetensors"))
        free_bytes = shutil.disk_usage(directory).free
        model_copies = 6
        print("CHECKPOINT_STORAGE", json.dumps({"shared_free_gb": free_bytes / 1024**3,
              "model_only_lower_bound_gb": weight_bytes * model_copies / 1024**3,
              "note": "Model/export lower bound only; optimizer/rank state needs additional space"}), flush=True)
        if weight_bytes == 0 or free_bytes < weight_bytes * model_copies:
            raise RuntimeError("Insufficient minimum checkpoint space or missing safetensors model weights")
    prepares_data = (args.formal_gsm8k or getattr(args, "baseline_5step", False)
                     or getattr(args, "native_baseline", False))
    manifest = prepare_formal_data(args, directory, image_id) if prepares_data else None
    # CPU preparation may take minutes; recheck occupancy immediately before launch.
    if prepares_data:
        for index, host in enumerate(args.hosts):
            require_idle(execute(host, index == 0, ["npu-smi", "info"]).stdout, host)
    plan = {"hosts": args.hosts, "addresses": addresses, "image_id": image_id, "revision": revision,
            "world_size": len(args.hosts) * 16,
            "training": "TP4/CP2/EP4; DP=world/8; EDP=world/4",
            "rollout_per_node": ("native TP2/DP8/EP16" if getattr(args, "native_baseline", False)
                                 else "TP2/DP8/EP16" if getattr(args, "baseline_5step", False)
                                 else "TP2/DP8/EP8"), "cpu_offload": args.cpu_offload,
            "steps": args.steps, "model_layers": "full", "model_root": str(args.model_root),
            "data_root": str(args.data_root), "batch": batch, "full_data": args.full_data,
            "max_prompt_tokens": args.max_prompt, "max_new_tokens": args.max_new_tokens,
            "max_model_len": args.max_prompt + args.max_new_tokens, "plot": args.plot,
            "formal_gsm8k": args.formal_gsm8k, "dataset_counts": manifest["counts"] if manifest else None,
            "baseline_5step": getattr(args, "baseline_5step", False),
            "native_baseline": getattr(args, "native_baseline", False),
            "compare_with": str(comparison.resolve()) if comparison is not None else None,
            "gpu_memory_utilization": getattr(args, "gpu_memory_utilization", None),
            "strict_consistency": not getattr(args, "native_baseline", False),
            "shared_free_gb_at_start": shutil.disk_usage(directory).free / 1024**3,
            "scope": ("Native vLLM baseline on 6400/500 samples; strict consistency disabled"
                      if getattr(args, "native_baseline", False) else
                      f"Unified {args.steps}-step run over {args.steps * 128} of 6400 questions; "
                      "fixed 500-question evaluation"
                      if getattr(args, "baseline_5step", False) else
                      "One complete GSM8K epoch, held-out validation and final official test" if manifest
                      else "Short consistency run; not full GSM8K training/evaluation")}
    (directory / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print("BATCH_PLAN", json.dumps(batch), flush=True)
    launched = []
    result = {**plan, "passed": False}
    started = time.monotonic()
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
        native = getattr(args, "native_baseline", False)
        report = directory / ("node0/run/native-training.json" if native else "node0/run/consistency.json")
        if report.exists():
            evidence = json.loads(report.read_text(encoding="utf-8"))
            result["native_training" if native else "consistency"] = evidence
            result["passed"] = (evidence.get("passed") is True
                                and evidence.get("completed_steps") == args.steps
                                and (native or len(evidence.get("checks", [])) == args.steps)
                                and all(not state["Running"] and state["ExitCode"] == 0 for state in states))
        if args.formal_gsm8k:
            formal_path = directory / "node0/run/formal-summary.json"
            result["formal"] = json.loads(formal_path.read_text()) if formal_path.is_file() else {}
            result["passed"] = result["passed"] and result["formal"].get("passed") is True
        if args.plot:
            plots = directory / "node0/run/plots"
            marker = plots / "plot-summary.json"
            evidence = json.loads(marker.read_text()) if marker.is_file() else {}
            figures = evidence.get("figures", [])
            verified = bool(figures) and evidence.get("accepted", evidence.get("passed")) is True
            for kind in figures:
                for extension in ("png", "svg"):
                    path = plots / f"{kind}.{extension}"
                    verified = verified and path.is_file() and path.stat().st_size > 0
            result["plots_verified"] = verified
            result["passed"] = result["passed"] and verified
            if getattr(args, "baseline_5step", False) or getattr(args, "native_baseline", False):
                result["final_test_verified"] = evidence.get("final_test_verified") is True
                result["passed"] = result["passed"] and result["final_test_verified"]
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
        result["elapsed_launch_to_collection_seconds"] = time.monotonic() - started
        result["elapsed_controller_seconds"] = time.monotonic() - controller_started
        (directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        timing_keys = ("warm_step_samples", "mean_seconds", "warm_generated_tokens", "rollout_summary",
                       "reported_peak_allocated_gb", "memory_scope", "timing_error")
        print("CLUSTER_TIMING", json.dumps({key: result[key] for key in timing_keys if key in result}), flush=True)
    if not result["passed"]:
        print("CLUSTER_RESULT", json.dumps({"passed": False, "directory": str(directory)}), flush=True)
        raise RuntimeError(f"Cluster training failed; inspect {directory}/node*.log and node*/run")
    if getattr(args, "native_baseline", False):
        try:
            export_native_figures(args, directory, source, image_id)
            result["native_figures_verified"] = True
            result["comparison_figures_verified"] = getattr(args, "compare_with", None) is not None
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            result.update(passed=False, post_training_plot_error=str(error))
            raise
        finally:
            (directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print("CLUSTER_RESULT", json.dumps({"passed": result["passed"], "directory": str(directory)}), flush=True)
    else:
        print("CLUSTER_RESULT", json.dumps({"passed": True, "directory": str(directory)}), flush=True)
    return directory


def export_native_figures(args: argparse.Namespace, directory: Path, source: Path, image_id: str) -> None:
    """Use the existing runtime image on CPU to export native and optional paired figures."""
    output = directory / "node0/run"
    command = ["docker", "run", "--rm", "--network=none", "--user", f"{os.getuid()}:{os.getgid()}",
               "-e", "MPLCONFIGDIR=/tmp/matplotlib", "--entrypoint", "python",
               "-v", f"{source}:/workspace:ro", "-v", f"{directory}:/native-result:ro",
               "-v", f"{output}:/plots:rw"]
    comparison = getattr(args, "compare_with", None)
    if comparison is not None:
        command.extend(["-v", f"{comparison.resolve()}:/consistency-result:ro"])
    command.extend([image_id, f"/workspace/{SCRIPTS}/plot_comparison.py", "--native", "/native-result",
                    "--expected-steps", str(args.steps), "--output",
                    "/plots/comparison-plots" if comparison is not None else "/plots/native-plots"])
    if comparison is not None:
        command.extend(["--consistency", "/consistency-result", "--single-output", "/plots/native-plots"])
    execute(args.hosts[0], True, command)
    for subdirectory, suffix in (("native-plots", ""), ("comparison-plots", "-comparison")):
        if subdirectory == "comparison-plots" and comparison is None:
            continue
        for name in ("reward", "test-accuracy", "logp-max", "time"):
            for extension in ("png", "svg"):
                path = output / subdirectory / f"{name}{suffix}.{extension}"
                if not path.is_file() or path.stat().st_size == 0:
                    raise RuntimeError(f"Missing native experiment figure: {path}")
    print("NATIVE_FIGURES", str(output / "native-plots"), flush=True)
    if comparison is not None:
        print("COMPARISON_FIGURES", str(output / "comparison-plots"), flush=True)


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
    parser.add_argument("--steps", type=int)
    parser.add_argument("--formal-gsm8k", action="store_true",
                        help="50 steps over 6726 training questions, 747 validation and 1319 final test")
    parser.add_argument("--native-baseline", action="store_true",
                        help="Native vLLM MoE without strict consistency; default 2 steps, up to 50")
    parser.add_argument("--compare-with", type=Path,
                        help="Existing successful Consistency cluster root; native 50-step runs only")
    parser.add_argument("--baseline-5step", action="store_true",
                        help="Unified 6400/500 baseline; default 2-step pilot, --steps 50 for a complete epoch")
    parser.add_argument("--global-prompt-batch-size", type=int, help="Global questions per step across training DP")
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--response-mini-batch-size", type=int,
                        help="Responses per training DP mini-batch; default is all local responses")
    parser.add_argument("--full-data", action="store_true",
                        help="Shuffle all training rows; steps need not cover an epoch")
    parser.add_argument("--plot", action="store_true", help="Export rank-zero reward/loss/consistency figures")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-prompt", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float,
                        help="6400/500 profiles only: vLLM memory fraction; "
                             "default 0.4, lower to reserve Trainer headroom")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--timeout", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.compare_with is not None and (not args.native_baseline or args.steps != 50):
        parser.error("--compare-with requires --native-baseline --steps 50")
    if args.gpu_memory_utilization is not None:
        if not (args.baseline_5step or args.native_baseline):
            parser.error("--gpu-memory-utilization is supported only by the 6400/500 baseline modes")
        if not 0 < args.gpu_memory_utilization <= 1:
            parser.error("--gpu-memory-utilization must be in (0, 1]")
    elif args.baseline_5step or args.native_baseline:
        args.gpu_memory_utilization = 0.4
    if args.native_baseline:
        if args.formal_gsm8k or args.baseline_5step:
            parser.error("--native-baseline cannot be combined with unified baseline or formal modes")
        args.steps = 2 if args.steps is None else args.steps
        args.global_prompt_batch_size = 128 if args.global_prompt_batch_size is None else args.global_prompt_batch_size
        args.response_mini_batch_size = 128 if args.response_mini_batch_size is None else args.response_mini_batch_size
        if (not 1 <= args.steps <= 50 or args.select_nodes != 4 or args.global_prompt_batch_size != 128
                or args.micro_batch_size != 2 or args.response_mini_batch_size != 128 or args.cpu_offload):
            parser.error("Native baseline requires four nodes, 1..50 steps, "
                         "batch 128, micro/mini=2/128 and offload=false")
        args.cpu_offload = False
        args.full_data = False
        args.plot = True
    if args.baseline_5step:
        if args.formal_gsm8k:
            parser.error("--baseline-5step and --formal-gsm8k are mutually exclusive")
        args.steps = 2 if args.steps is None else args.steps
        args.global_prompt_batch_size = 128 if args.global_prompt_batch_size is None else args.global_prompt_batch_size
        args.response_mini_batch_size = 128 if args.response_mini_batch_size is None else args.response_mini_batch_size
        if (args.steps not in (2, 5, 50) or args.select_nodes != 4 or args.global_prompt_batch_size != 128
                or args.micro_batch_size != 2 or args.response_mini_batch_size != 128 or args.cpu_offload):
            parser.error("Baseline requires four nodes, 2, 5 or 50 steps, "
                         "batch 128, micro/mini=2/128 and offload=false")
        args.full_data = False
        args.plot = True
    if args.summarize is not None:
        directory = args.summarize.resolve()
        result = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        result = {key: result.get(key) for key in ("passed", "hosts", "revision", "steps", "batch", "full_data",
                                                  "formal_gsm8k", "formal", "elapsed_launch_to_collection_seconds",
                                                  "elapsed_controller_seconds")}
        result.update(directory=str(directory), **summarize_timings(directory))
        print(json.dumps(result, indent=2), flush=True)
        return
    args.steps = args.steps if args.steps is not None else (50 if args.formal_gsm8k else 3)
    args.timeout = args.timeout if args.timeout is not None else (86400 if args.steps == 50 else 14400)
    if args.formal_gsm8k:
        args.global_prompt_batch_size = 136 if args.global_prompt_batch_size is None else args.global_prompt_batch_size
        args.response_mini_batch_size = 68 if args.response_mini_batch_size is None else args.response_mini_batch_size
        if (args.steps != 50 or args.select_nodes != 4 or args.global_prompt_batch_size != 136
                or args.micro_batch_size != 1 or args.response_mini_batch_size != 68 or args.cpu_offload):
            parser.error("Formal GSM8K retains four nodes, 50 steps, batch 136, micro/mini=1/68 and offload=false")
        args.full_data = args.plot = True
    if not args.hosts or not args.master_addr:
        parser.error("--hosts and --master-addr are required to launch training")
    for address in [*args.hosts, args.master_addr]:
        ipaddress.IPv4Address(address)
    if (len(set(args.hosts)) != len(args.hosts) or not 2 <= args.select_nodes <= len(args.hosts)
            or not 1024 <= args.master_port <= 65535
            or min(args.steps, args.max_new_tokens, args.max_prompt, args.timeout) < 1):
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
                          "rollout_per_node": ("native TP2/DP8/EP16" if args.native_baseline else
                                               "TP2/DP8/EP16" if args.baseline_5step else "TP2/DP8/EP8"),
                          "shared_root": str(args.shared_root),
                          "batch": batch, "full_data": args.full_data, "plot": args.plot,
                          "max_prompt_tokens": args.max_prompt, "max_new_tokens": args.max_new_tokens,
                          "max_model_len": args.max_prompt + args.max_new_tokens,
                          "formal_gsm8k": args.formal_gsm8k, "baseline_5step": args.baseline_5step,
                          "max_num_batched_tokens": (8192 if args.baseline_5step or args.native_baseline
                                                     else args.max_prompt),
                          "native_baseline": args.native_baseline,
                          "compare_with": str(args.compare_with) if args.compare_with is not None else None,
                          "max_num_seqs": 64 if args.baseline_5step or args.native_baseline else 4,
                          "gpu_memory_utilization": (args.gpu_memory_utilization
                                                     if args.baseline_5step or args.native_baseline else 0.35),
                          "kv_cache_auto": args.baseline_5step or args.native_baseline,
                          "strict_consistency": not args.native_baseline,
                          "timeout": args.timeout}, indent=2))
        return
    run(args)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupt)
    main()
