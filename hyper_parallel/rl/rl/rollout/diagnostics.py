# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Bounded, host-only evidence captured before rollout error cleanup."""

import functools
import json
import logging
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional
from urllib import request as urllib_request


def _best_effort(method: Callable) -> Callable:
    """Keep diagnostic I/O errors from changing generation or its original exception."""
    @functools.wraps(method)
    def guarded(*args: Any, **kwargs: Any) -> Any:
        try:
            return method(*args, **kwargs)
        except Exception as error:  # pylint: disable=broad-exception-caught
            logging.getLogger(__name__).warning("ROLLOUT_DIAG unavailable: %s: %r", method.__name__, error)
            return None
    return guarded


def _atomic_json(path: Path, value: dict) -> None:
    """Publish complete snapshots for readers in other trainer processes."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=True, indent=2), encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict:
    """Tolerate missing or partially written optional evidence from other processes."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _log_evidence(path: Path) -> dict:
    """Read a bounded pre-cleanup log tail, excluding earlier diagnostic blocks."""
    if not path.is_file():
        return {"available": False, "errors": [], "stats": []}
    with path.open("rb") as stream:
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(max(0, size - 131072))
        lines = stream.read(131072).decode("utf-8", errors="replace").splitlines()
    lines = [re.sub(r"\x1b\[[0-9;]*m", "", line) for line in lines if "ROLLOUT_DIAG" not in line]
    errors = [line for line in lines if re.search(
        r"\bERROR\b|\b[A-Za-z]+Error\b|ERR\d+|HCCL.*(?:fail|timeout)|device.*error", line, re.I)]
    stats = [line for line in lines if re.search(
        r"Engine \d+|Avg generation throughput|Running:|Waiting:|KV cache usage", line)]
    return {"available": True, "tail_bytes": min(size, 131072),
            "errors": errors[:1] + errors[-3:] if len(errors) > 4 else errors, "stats": stats[-4:]}


def _server_processes(path: Path) -> dict:
    """Inspect only the registered server process group; do not attach or signal it."""
    if not path.is_file():
        return {"registered": False, "members": []}
    server = json.loads(path.read_text(encoding="utf-8"))
    pid = server["pid"]
    leader = Path(f"/proc/{pid}/stat")
    if leader.is_file():
        fields = leader.read_text(encoding="utf-8").rsplit(")", 1)[1].split()
        if fields[19] != server["start_ticks"]:
            return {"registered": True, "pid_reused": True, "members": []}
    members = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="utf-8")
            fields = stat.rsplit(")", 1)[1].split()
            if int(fields[2]) != pid:
                continue
            members.append({"pid": int(entry.name), "name": stat.split("(", 1)[1].rsplit(")", 1)[0],
                            "state": fields[0], "cpu_ticks": int(fields[11]) + int(fields[12]),
                            "wchan": (entry / "wchan").read_text(encoding="utf-8").strip()})
        except (OSError, ValueError, IndexError):
            continue
        if len(members) == 24:
            break
    return {"registered": True, "leader_pid": pid, "members": members}


def _metric_lines(telemetry: dict, now: float) -> list[str]:
    """Keep all reported engines visible instead of truncating one metric family."""
    samples = [sample for sample in telemetry["samples"] if sample["values"]]
    if not samples:
        return ["engine_metrics=UNAVAILABLE; inspect engine_stats above"]
    latest = samples[-1]
    engines: dict[str, list[str]] = {}
    for key, value in latest["values"].items():
        label = re.search(r'(?:engine|engine_id|data_parallel_rank)="([^\"]+)"', key)
        engine = label.group(1) if label else "unlabelled/aggregate"
        field = key.split("{", 1)[0].removeprefix("vllm:").removeprefix("vllm_")
        field = field.replace("num_requests_", "").replace("generation_tokens_total", "tokens")
        detail = f"{field}={value:g}"
        if field == "tokens" and len(samples) > 1 and key in samples[0]["values"]:
            delta = value - samples[0]["values"][key]
            detail += f"(delta={delta:g}/{latest['time'] - samples[0]['time']:.0f}s)"
        engines.setdefault(engine, []).append(detail)
    lines = [f"metrics_age_s={now - latest['time']:.1f} (engine-wide; not per-request)"]
    lines.extend(f"engine_label={engine}: {' '.join(fields)}" for engine, fields in list(engines.items())[:8])
    return lines


class RolloutDiagnostics:
    """Track latest request batches without tokens, collectives, or recovery actions."""

    def __init__(self, endpoint: str, timeout: float) -> None:
        """Bind optional run-local evidence storage to one HTTP client."""
        directory = os.environ.get("HYPER_RL_DIAGNOSTICS_DIR")
        self.directory = Path(directory) if directory else None
        self.endpoint = endpoint
        self.timeout = timeout
        self.rank = os.environ.get("RANK", "unknown")
        self.client_id = f"{self.rank}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self.state: dict = {}
        self._stop = threading.Event()
        self._monitor: Optional[threading.Thread] = None

    @_best_effort
    def register_server(self, pid: int) -> None:
        """Share server identity with non-owner trainer clients before generation."""
        if self.directory is None:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
        _atomic_json(self.directory / "rollout-server.json", {"pid": pid, "start_ticks": fields[19]})
        self._monitor = threading.Thread(target=self._sample_metrics, daemon=True, name="rollout-diagnostics")
        self._monitor.start()

    def close(self) -> None:
        """Stop host-only monitoring without waiting on the inference service."""
        self._stop.set()

    def _sample_metrics(self) -> None:
        """Retain three bounded metrics snapshots; failure never affects the engine."""
        history = []
        while not self._stop.is_set():
            sample: dict = {"time": time.time(), "values": {}}
            try:
                request = urllib_request.Request(self.endpoint + "/metrics")
                # Metrics are local; proxy settings must not redirect this probe.
                opener = urllib_request.build_opener(urllib_request.ProxyHandler({}))
                with opener.open(request, timeout=2) as response:
                    body = response.read(131072).decode("utf-8", errors="replace")
                for line in body.splitlines():
                    if line.startswith("#") or not re.match(
                            r"vllm[:_](?:generation_tokens_total|num_requests_running|num_requests_waiting)(?:\{| )",
                            line):
                        continue
                    key, value = line.rsplit(" ", 1)
                    sample["values"][key] = float(value)
                    if len(sample["values"]) >= 32:
                        break
            except Exception as error:  # pylint: disable=broad-exception-caught
                sample["unavailable"] = f"{type(error).__name__}: {error!r}"
            history = (history + [sample])[-3:]
            try:
                _atomic_json(self.directory / "rollout-metrics.json", {"samples": history})
            except OSError:
                return
            self._stop.wait(30)

    @_best_effort
    def start_batch(self, count: int) -> None:
        """Reset per-client evidence at each generation call, tagged by trainer step."""
        if self.directory is None:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        phase_file = self.directory / f"progress-rank{self.rank}.json"
        phase = _read_json(phase_file)
        self.state = {"client": self.client_id, "rank": self.rank, "endpoint": self.endpoint,
                      "batch": uuid.uuid4().hex[:8], "step": phase.get("step"),
                      "started": time.time(), "expected_requests": count, "requests": {}}
        self._save()

    @_best_effort
    def admit(self, request: Any, max_tokens: int) -> Optional[str]:
        """Record metadata only; the ID identifies the client request, not a DP replica."""
        if self.directory is None or not self.state:
            return None
        identifier = f"{self.client_id}-{self.state['batch']}-{request.ordinal}"
        self.state["requests"][identifier] = {
            "id": identifier, "status": "waiting", "started": time.time(),
            "start_row": request.start_row, "prompt_lengths": [len(prompt) for prompt in request.prompts],
            "choices": request.num_choices, "max_tokens": max_tokens, "seed": request.seed,
        }
        self._save()
        return identifier

    @_best_effort
    def finish(self, identifier: Optional[str], error: Optional[Exception] = None) -> None:
        """Record completion or print one failure snapshot before cancellation and cleanup."""
        if identifier is None or self.directory is None:
            return
        request = self.state["requests"][identifier]
        request.update(status="failed" if error is not None else "completed", ended=time.time())
        if error is not None:
            request.update(error_type=type(error).__name__, error_repr=repr(error))
        self._save()
        if error is not None:
            self._failure(request)

    def _save(self) -> None:
        _atomic_json(self.directory / f"rollout-requests-{self.client_id}.json", self.state)

    def _failure(self, failed: dict) -> None:
        claim = self.directory / "rollout-first-failure.lock"
        try:
            with claim.open("x", encoding="utf-8") as stream:
                stream.write(failed["id"])
        except FileExistsError:
            return
        now = time.time()
        batches = []
        for path in self.directory.glob("rollout-requests-*.json"):
            try:
                state = json.loads(path.read_text())
                if state.get("step") == self.state["step"] and state.get("endpoint") == self.endpoint:
                    batches.append(state)
            except (OSError, ValueError):
                continue
        requests = [request for state in batches for request in state["requests"].values()]
        counts = {status: sum(request["status"] == status for request in requests)
                  for status in ("completed", "waiting", "failed")}
        try:
            evidence = _log_evidence(self.directory / "train.log")
        except OSError as error:
            evidence = {"available": False, "errors": [], "stats": [], "unavailable": repr(error)}
        try:
            server = _server_processes(self.directory / "rollout-server.json")
        except (OSError, ValueError, IndexError) as error:
            server = {"members": [], "unavailable": repr(error)}
        metrics_file = self.directory / "rollout-metrics.json"
        telemetry = _read_json(metrics_file) or {"samples": []}
        launcher = {name: os.environ.get(name, "unknown") for name in (
            "DEVICE_COUNT", "MODEL_LAYERS", "TRAIN_TP", "TRAIN_EP", "ROLLOUT_TP", "ROLLOUT_EP", "EP_CAPACITY")}
        report = {"captured_before_client_cleanup": True, "captured_at": now,
                  "step": self.state["step"], "endpoint": self.endpoint, "timeout_seconds": self.timeout,
                  "failed_request": failed, "counts": counts, "batches": batches,
                  "log_evidence": evidence, "server": server, "metrics": telemetry, "launcher": launcher,
                  "root_cause": "UNDETERMINED", "dp_replica": "UNKNOWN",
                  "scope": "Latest per-client batches for this step; HTTP completion is not token progress."}
        lines = ["=== ROLLOUT_DIAG BEGIN (before client cleanup) ===",
                 f"step={report['step']} rank={self.rank} timeout_s={self.timeout:g}",
                 f"launcher: world={launcher['DEVICE_COUNT']} layers={launcher['MODEL_LAYERS']} "
                 f"train_tp/ep={launcher['TRAIN_TP']}/{launcher['TRAIN_EP']} "
                 f"rollout_tp/ep={launcher['ROLLOUT_TP']}/{launcher['ROLLOUT_EP']} "
                 f"ep_capacity={launcher['EP_CAPACITY']}",
                 f"error={failed['error_type']} {failed['error_repr'][:180]}",
                 f"client_request={failed['id']}",
                 f"elapsed_s={now - failed['started']:.1f} prompt_lengths={failed['prompt_lengths']} "
                 f"max_tokens={failed['max_tokens']}",
                 f"latest_batches: completed={counts['completed']} waiting={counts['waiting']} "
                 f"failed={counts['failed']} clients={len(batches)}",
                 "DP_REPLICA=UNKNOWN; client rank/API server number does not prove engine mapping.",
                 "ROOT_CAUSE=UNDETERMINED; evidence below is not a deadlock diagnosis."]
        for request in requests:
            if request["status"] == "waiting" and len(lines) < 12:
                lines.append(f"pending={request['id']} age_s={now - request['started']:.1f}")
        error_lines = evidence["errors"]
        if len(error_lines) > 3:
            error_lines = error_lines[:1] + error_lines[-2:]
        for line in error_lines:
            lines.append("backend_log: " + line[:220])
        for line in evidence["stats"][-3:]:
            lines.append("engine_stats: " + line[:220])
        if not evidence["errors"]:
            lines.append("backend_log: no error found in last 128 KiB (does not exclude an earlier error).")
        members = server["members"]
        lines.append("server_processes: " + (", ".join(
            f"{member['pid']}:{member['state']}:{member['wchan']}" for member in members[:8]) or "unavailable/exited"))
        lines.extend(_metric_lines(telemetry, now))
        lines.extend(["REQUEST_TOKEN_PROGRESS=UNKNOWN; engine metrics cannot identify the pending request.",
                      f"details={self.directory / 'rollout-failure.json'}",
                      "=== ROLLOUT_DIAG END ==="])
        photo = "\n".join(lines)
        logging.getLogger(__name__).error("%s", photo)
        _atomic_json(self.directory / "rollout-failure.json", report)
        (self.directory / "rollout-failure.txt").write_text(photo + "\n", encoding="utf-8")
